#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Distributed training script for protein-to-mRNA translation model with ESM-2 integration.
Usage: 
    torchrun --nproc_per_node=2 --master_port=49514 actor_train.py > actor_train.txt 2>&1 & disown
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from utils import (
    Tokenizer,
    ActorModel_encoder_esm2,
    LengthAwareDistributedSampler_human
)
from transformers import AutoTokenizer, EsmModel

# Environment Configuration
def setup_env():
    """Set environment variables for reproducibility and distributed training"""
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

def set_seed(seed=42):
    """Initialize random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def init_distributed():
    """Initialize distributed training environment"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    return local_rank, device, world_size

# Data Processing
def read_sequences_from_csv(file_path):
    """Load protein and mRNA sequences from CSV file"""
    data = pd.read_csv(file_path, usecols=['Protein_Sequence', 'CDS_Sequence', 'Protein_Length'])
    return (
        data['Protein_Sequence'].tolist(),
        data['CDS_Sequence'].tolist(),
        data['Protein_Length'].tolist()
    )

def preprocess_data(protein_sequences, mRNA_sequences, tokenizer, max_length):
    """Encode and pad sequences for model input"""
    encoded_proteins = [tokenizer.encode_pro(seq, max_length) for seq in protein_sequences]
    encoded_mrnas = [tokenizer.encode_mrna(seq, max_length) for seq in mRNA_sequences]
    return encoded_proteins, encoded_mrnas, protein_sequences

# Dataset Classes
class ProteinTranslationDataset(Dataset):
    """Custom dataset for protein-to-mRNA translation task"""
    def __init__(self, encoded_proteins, encoded_mrnas, pro_seqs):
        self.encoded_proteins = encoded_proteins
        self.encoded_mrnas = encoded_mrnas
        self.pro_seqs = pro_seqs

    def __len__(self):
        return len(self.pro_seqs)

    def __getitem__(self, idx):
        return (
            self.encoded_proteins[idx],
            self.encoded_mrnas[idx],
            self.pro_seqs[idx]
        )

def collate_fn(batch, tokenizer):
    """Batch processing with dynamic padding"""
    pro_max_length = max(len(item[0]) for item in batch)
    mrna_max_length = max(len(item[1]) for item in batch)
    
    padded_proteins = [tokenizer.pad(item[0], pro_max_length) for item in batch]
    padded_mrnas = [tokenizer.pad(item[1], mrna_max_length) for item in batch]
    
    return (
        torch.tensor(padded_proteins),
        torch.tensor(padded_mrnas),
        [item[2] for item in batch]
    )

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, 
                       tokenizer, num_workers, protein_lengths, world_size, local_rank):
    """Create distributed data loaders with custom samplers"""
    train_sampler = LengthAwareDistributedSampler_human(
        train_dataset, protein_lengths, 1.5, world_size, local_rank
    )
    
    loader_args = {
        'batch_size': batch_size,
        'collate_fn': lambda b: collate_fn(b, tokenizer),
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True
    }
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset), **loader_args)
    test_loader = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset), **loader_args)
    
    return train_loader, val_loader, test_loader, train_sampler

# Training Components
class TrainingManager:
    """Orchestrates training process with mixed precision and distributed training"""
    
    def __init__(self, model, esm2_model, esm2_tokenizer, optimizer, criterion,
                 max_seq_length, device, scaler, entropy_weight, ortho_weight):
        self.model = model
        self.esm2_model = esm2_model
        self.esm2_tokenizer = esm2_tokenizer
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_seq_length = max_seq_length
        self.device = device
        self.scaler = scaler
        self.entropy_weight = entropy_weight
        self.ortho_weight = ortho_weight

    def train_epoch(self, train_loader):
        """Single training epoch with mixed precision"""
        self.model.train()
        metrics = {'loss': 0.0, 'ce_loss': 0.0, 'entropy_loss': 0.0, 'ortho_loss': 0.0}
        
        for src, tgt, pro_seqs in train_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            esm2_features = self._get_esm2_features(pro_seqs)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs, router_logits, entropy_loss = self.model.module(src, esm2_features)
                loss, losses = self._calculate_loss(outputs, tgt, entropy_loss)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            for k, v in losses.items():
                metrics[k] += v.item()
        
        return {k: v/len(train_loader) for k, v in metrics.items()}

    def _get_esm2_features(self, pro_seqs):
        """Extract ESM-2 protein features"""
        with torch.no_grad():
            inputs = self.esm2_tokenizer(
                pro_seqs, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)
            return self.esm2_model(**inputs).last_hidden_state

    def _calculate_loss(self, outputs, targets, entropy_loss):
        """Calculate combined loss function"""
        ce_loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        entropy_loss = torch.exp(-entropy_loss) * self.entropy_weight
        ortho_loss = self._calculate_orthogonal_loss() * self.ortho_weight
        total_loss = ce_loss + entropy_loss + ortho_loss
        
        return total_loss, {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'entropy_loss': entropy_loss,
            'ortho_loss': ortho_loss
        }

    def _calculate_orthogonal_loss(self):
        """Calculate MOE layer orthogonal regularization"""
        return torch.mean(torch.stack([
            layer.moe.orthogonal_loss() 
            for layer in self.model.module.encoder.layers
        ]))

# Main Training Flow
def main():
    # Environment setup
    setup_env()
    set_seed(42)
    local_rank, device, world_size = init_distributed()
    
    # Model configuration
    config = {
        'd_model': 768,
        'nhead': 8,
        'num_encoder_layers': 8,
        'dim_feedforward': 1536,
        'dropout': 0.3,
        'num_experts': 6,
        'top_k_experts': 2,
        'max_seq_length': 1310,
        'esm2_dim': 1280,
        'batch_size': 32,
        'lr': 3e-5,
        'epochs': 6,
        'entropy_weight': 1.0,
        'ortho_weight': 10000.0
    }
    
    # Initialize models
    tokenizer = Tokenizer()
    esm2_tokenizer = AutoTokenizer.from_pretrained("./esm2_model_t33_650M_UR50D", local_files_only=True)
    esm2_model = EsmModel.from_pretrained("./esm2_model_t33_650M_UR50D", local_files_only=True).to(device).eval()
    
    model = ActorModel_encoder_esm2(
        vocab_size=len(tokenizer.tokens),
        **{k: config[k] for k in ['d_model', 'nhead', 'num_encoder_layers', 
                                'dim_feedforward', 'dropout', 'num_experts', 'top_k_experts']}
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # Data loading
    train_data = read_sequences_from_csv('../example_data/example_train.csv')
    val_data = read_sequences_from_csv('../example_data/example_val.csv')
    test_data = read_sequences_from_csv('../example_data/example_test.csv')
    
    datasets = {
        'train': ProteinTranslationDataset(*preprocess_data(*train_data[:2], tokenizer, config['max_seq_length'])),
        'val': ProteinTranslationDataset(*preprocess_data(*val_data[:2], tokenizer, config['max_seq_length'])),
        'test': ProteinTranslationDataset(*preprocess_data(*test_data[:2], tokenizer, config['max_seq_length']))
    }
    
    loaders = create_data_loaders(
        datasets['train'], datasets['val'], datasets['test'],
        config['batch_size'], tokenizer, min(8, os.cpu_count()),
        train_data[2], world_size, local_rank
    )
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.padding_idx)
    trainer = TrainingManager(
        model, esm2_model, esm2_tokenizer, optimizer, criterion,
        config['max_seq_length'], device, GradScaler(),
        config['entropy_weight'], config['ortho_weight']
    )
    
    # Training loop
    if local_rank == 0:
        print("Training started...")
        start_time = time.time()
    
    for epoch in range(config['epochs']):
        loaders[-1].set_epoch(epoch)  # Update sampler
        epoch_start = time.time()
        
        # Training phase
        train_metrics = trainer.train_epoch(loaders[0])
        
        # Validation phase
        val_loss = trainer.evaluate(loaders[1])
        test_loss = trainer.evaluate(loaders[2])
        
        # Reporting
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:02d} [{epoch_time:.1f}s]: "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, "
                  f"Ent: {train_metrics['entropy_loss']:.4f}, "
                  f"Ortho: {train_metrics['ortho_loss']:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    # Finalization
    if local_rank == 0:
        torch.save(model.module.state_dict(), "./actor.pt")
        print(f"Training completed in {(time.time()-start_time)/3600:.2f} hours")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
