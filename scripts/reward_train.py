#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Distributed training script for mRNA reward prediction model with MOE architecture.
Usage: 
    torchrun --nproc_per_node=2 --master_port=49516 reward_train.py > reward_train.txt 2>&1 & disown
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
    RewardModel_encoder,
    LengthAwareDistributedSampler_human
)

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

# Reward Calculation Utilities
def calculate_reward_components(cds_mfes, cds_cais, cds_lengths, target_cai=0.8):
    """
    Calculate normalized reward components for MFE and CAI features
    Args:
        cds_mfes: List of minimum free energy values
        cds_cais: List of codon adaptation index values
        cds_lengths: List of sequence lengths
        target_cai: Target CAI value for full reward
    Returns:
        Tuple of (normalized MFE rewards, normalized CAI rewards)
    """
    # Normalize MFE by sequence length
    stand_mfes = [mfe/max(length, 1) for mfe, length in zip(cds_mfes, cds_lengths)]
    
    # Normalize MFE rewards to [0, 1]
    neg_stand_mfes = [-s for s in stand_mfes]
    min_neg = min(neg_stand_mfes) if neg_stand_mfes else 0
    max_neg = max(neg_stand_mfes) if neg_stand_mfes else 0
    mfe_range = max_neg - min_neg or 1.0  # Handle zero division
    
    scaled_mfe = [(n - min_neg)/mfe_range for n in neg_stand_mfes]
    scaled_mfe = [max(0.0, min(1.0, r)) for r in scaled_mfe]
    
    # Calculate CAI rewards with linear scaling
    scaled_cai = [1.0 if cai >= target_cai else max(0.0, cai/target_cai) 
                 for cai in cds_cais]
    
    # Safety check for list lengths
    min_len = min(len(scaled_mfe), len(scaled_cai))
    return scaled_mfe[:min_len], scaled_cai[:min_len]

# Data Processing
class RewardDataset(Dataset):
    """Dataset for mRNA sequence reward prediction"""
    def __init__(self, encoded_sequences, rewards):
        self.sequences = encoded_sequences
        self.rewards = rewards

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return self.sequences[idx], self.rewards[idx]

def preprocess_data(sequences, tokenizer, max_length):
    """Encode and pad mRNA sequences"""
    return [tokenizer.encode_mrna(seq, max_length) for seq in sequences]

def collate_fn(batch, tokenizer):
    """Batch processing with dynamic padding"""
    seqs, rewards = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded_seqs = [tokenizer.pad(s, max_len) for s in seqs]
    return (
        torch.tensor(padded_seqs),
        torch.tensor(rewards, dtype=torch.float32)
    )

def create_data_loaders(train_set, val_set, test_set, batch_size, tokenizer, 
                       num_workers, protein_lengths, world_size, local_rank):
    """Create distributed data loaders with custom samplers"""
    train_sampler = LengthAwareDistributedSampler_human(
        train_set, protein_lengths, 1.5, world_size, local_rank
    )
    
    loader_args = {
        'batch_size': batch_size,
        'collate_fn': lambda b: collate_fn(b, tokenizer),
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True
    }
    
    return (
        DataLoader(train_set, sampler=train_sampler, **loader_args),
        DataLoader(val_set, sampler=DistributedSampler(val_set), **loader_args),
        DataLoader(test_set, sampler=DistributedSampler(test_set), **loader_args),
        train_sampler
    )

# Training Components
class RewardTrainer:
    """Training manager for reward prediction model"""
    
    def __init__(self, model, optimizer, criterion, device, scaler, 
                 entropy_weight, ortho_weight):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.entropy_weight = entropy_weight
        self.ortho_weight = ortho_weight

    def train_epoch(self, train_loader):
        """Single training epoch with mixed precision"""
        self.model.train()
        metrics = {'loss': 0.0, 'mse': 0.0, 'entropy': 0.0, 'ortho': 0.0}
        
        for seqs, rewards in train_loader:
            seqs, rewards = seqs.to(self.device), rewards.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs, routers, entropy_loss = self.model.module(seqs)
                loss, losses = self._compute_loss(outputs, rewards, entropy_loss)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            for k in metrics:
                metrics[k] += losses[k].item()
        
        return {k: v/len(train_loader) for k, v in metrics.items()}

    def _compute_loss(self, outputs, targets, entropy_loss):
        """Calculate combined loss components"""
        mse_loss = self.criterion(outputs.squeeze(), targets)
        entropy_loss = torch.exp(-entropy_loss) * self.entropy_weight
        ortho_loss = self._orthogonal_loss() * self.ortho_weight
        total_loss = mse_loss + entropy_loss + ortho_loss
        
        return total_loss, {
            'loss': total_loss,
            'mse': mse_loss,
            'entropy': entropy_loss,
            'ortho': ortho_loss
        }

    def _orthogonal_loss(self):
        """Calculate MOE layer orthogonal regularization"""
        return torch.mean(torch.stack([
            layer.moe.orthogonal_loss()
            for layer in self.model.module.encoder.layers
        ]))

    def evaluate(self, data_loader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for seqs, rewards in data_loader:
                seqs, rewards = seqs.to(self.device), rewards.to(self.device)
                outputs, _, _ = self.model.module(seqs)
                total_loss += self.criterion(outputs.squeeze(), rewards).item()
        
        return total_loss / len(data_loader)

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
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 10,
        'entropy_weight': 0.1,
        'ortho_weight': 1000.0,
        'target_cai': 0.8,
        'mfe_weight': 2.0,
        'cai_weight': 1.0
    }
    
    # Initialize components
    tokenizer = Tokenizer()
    
    # Load data
    def load_data(path):
        data = pd.read_csv(path, usecols=['CDS_Sequence', 'CDS_Length', 
                                        'CDS_MFE', 'CDS_CAI', 'Protein_Length'])
        mfe, cai = calculate_reward_components(
            data['CDS_MFE'], data['CDS_CAI'], data['CDS_Length'], config['target_cai']
        )
        rewards = [config['mfe_weight']*m + config['cai_weight']*c 
                  for m, c in zip(mfe, cai)]
        return (
            preprocess_data(data['CDS_Sequence'], tokenizer, config['max_seq_length']),
            rewards,
            data['Protein_Length'].tolist()
        )
    
    train_data = load_data('../example_data/example_train.csv')
    val_data = load_data('../example_data/example_val.csv')
    test_data = load_data('../example_data/example_test.csv')
    
    # Create datasets and loaders
    datasets = {
        'train': RewardDataset(*train_data[:2]),
        'val': RewardDataset(*val_data[:2]),
        'test': RewardDataset(*test_data[:2])
    }
    
    loaders = create_data_loaders(
        datasets['train'], datasets['val'], datasets['test'],
        config['batch_size'], tokenizer, min(8, os.cpu_count()),
        train_data[2], world_size, local_rank
    )
    
    # Initialize model
    model = RewardModel_encoder(
        vocab_size=len(tokenizer.tokens),
        **{k: config[k] for k in ['d_model', 'nhead', 'num_encoder_layers',
                                'dim_feedforward', 'dropout', 
                                'num_experts', 'top_k_experts']}
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], 
                                   output_device=local_rank)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    trainer = RewardTrainer(
        model, optimizer, criterion, device, GradScaler(),
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
        
        # Validation
        val_loss = trainer.evaluate(loaders[1])
        test_loss = trainer.evaluate(loaders[2])
        
        # Reporting
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:02d} [{epoch_time:.1f}s]: "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(MSE: {train_metrics['mse']:.4f}, "
                  f"Ent: {train_metrics['entropy']:.4f}, "
                  f"Ortho: {train_metrics['ortho']:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    # Finalization
    if local_rank == 0:
        torch.save(model.module.state_dict(), "./reward.pt")
        print(f"Training completed in {(time.time()-start_time)/3600:.2f} hours")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
