#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Distributed GRPO Training Script with ESM-2 Integration
Usage: 
    torchrun --nproc_per_node=2 --master_port=49515 grpo_train.py > grpo_train.txt 2>&1 & disown
"""

import os
import copy
import math
import random
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, EsmModel

from utils import (
    Tokenizer,
    ActorModel_encoder_esm2,
    RewardModel_encoder,
    LengthAwareDistributedSampler_human
)

# Configuration Constants
class Config:
    """Central configuration class for training parameters and paths."""
    # Environment Settings
    RANDOM_SEED = 42
    CUDA_VISIBLE_DEVICES = "0,1"
    NCCL_SOCKET_IFNAME = 'lo'
    NCCL_DEBUG = "ERROR"

    # Model Architecture
    D_MODEL = 768
    NUM_ENCODER_LAYERS = 8
    NHEAD = 8
    DIM_FEEDFORWARD = 1536  # 768 * 2
    DROPOUT = 0.3
    NUM_EXPERTS = 6
    TOP_K_EXPERTS = 2
    ESM2_DIM = 1280

    # Training Parameters
    NUM_EPOCHS = 12
    BATCH_SIZE = 2
    BASE_LEARNING_RATE = 2.0e-5
    NUM_WORKERS = 8
    MAX_SEQ_LENGTH = 1310

    # GRPO Hyperparameters
    GROUP_SIZE = 16
    KL_COEFFICIENT = 0.01
    CLIP_EPSILON = 0.2
    POLICY_UPDATE_INTERVAL = 50

    # Path Configuration
    DATA_PATHS = {
        'train': '../example_data/example_train.csv',
        'val': '../example_data/example_val.csv',
        'test': '../example_data/example_test.csv'
    }
    MODEL_PATHS = {
        'esm2': "./esm2_model_t33_650M_UR50D",
        'actor': "./actor.pt",
        'reward': "./reward.pt",
        'output_prefix': "./actor_grpo_"
    }

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

def configure_environment(config):
    """Set up environment variables and random seeds."""
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES
    os.environ['NCCL_SOCKET_IFNAME'] = config.NCCL_SOCKET_IFNAME
    os.environ['NCCL_DEBUG'] = config.NCCL_DEBUG
    set_seed(config.RANDOM_SEED)

def set_seed(seed_value):
    """Initialize random generators for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def initialize_distributed():
    """Set up distributed training environment."""
    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
    except KeyError:
        logger.warning("DDP environment variables not found. Using single-process mode.")
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    logger.info(f"Initialized DDP: Rank {rank}/{world_size} (Local {local_rank})")
    return rank, world_size, device

class ProteinDataset(Dataset):
    """Dataset for protein sequence processing."""
    def __init__(self, encoded_sequences, raw_sequences):
        self.encoded_sequences = encoded_sequences
        self.raw_sequences = raw_sequences

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, idx):
        return self.encoded_sequences[idx], self.raw_sequences[idx]

def collate_protein_batch(batch, tokenizer):
    """Batch processing with dynamic padding."""
    encoded_seqs, raw_seqs = zip(*batch)
    max_len = max(len(seq) for seq in encoded_seqs)
    padded_seqs = [tokenizer.pad(seq, max_len) for seq in encoded_seqs]
    return torch.tensor(padded_seqs, dtype=torch.long), raw_seqs

def create_distributed_loaders(datasets, config, rank, world_size):
    """Create distributed data loaders with appropriate samplers."""
    loaders = {}
    sampler_configs = {
        'train': {
            'sampler_class': LengthAwareDistributedSampler_human,
            'shuffle': False,
            'extra_args': (datasets['train'].protein_lengths, 1.0)
        },
        'val': {'sampler_class': DistributedSampler, 'shuffle': False},
        'test': {'sampler_class': DistributedSampler, 'shuffle': False}
    }

    for split in ['train', 'val', 'test']:
        sampler = sampler_configs[split]['sampler_class'](
            datasets[split],
            num_replicas=world_size,
            rank=rank,
            **({'seed': config.RANDOM_SEED} if split == 'train' else {})
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            collate_fn=lambda b: collate_protein_batch(b, tokenizer),
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    return loaders

class GRPOTrainer:
    """Main training handler for GRPO algorithm."""
    def __init__(self, config, device, rank, world_size):
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.global_step = 0
        
        # Initialize components
        self.tokenizer = Tokenizer()
        self.esm_tokenizer, self.esm_model = self.load_esm_model()
        self.actor, self.reward_model = self.initialize_models()
        self.optimizer = optim.AdamW(self.actor.parameters(), lr=config.BASE_LEARNING_RATE)
        self.scaler = GradScaler()

    def load_esm_model(self):
        """Load and configure ESM-2 model."""
        if self.rank == 0:
            logger.info("Loading ESM-2 model...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_PATHS['esm2'])
        model = EsmModel.from_pretrained(self.config.MODEL_PATHS['esm2']).to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return tokenizer, model

    def initialize_models(self):
        """Initialize and configure actor/reward models."""
        # Actor Model
        actor = ActorModel_encoder_esm2(
            vocab_size=len(self.tokenizer.tokens),
            d_model=self.config.D_MODEL,
            nhead=self.config.NHEAD,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            esm2_dim=self.config.ESM2_DIM,
            dropout=self.config.DROPOUT,
            num_experts=self.config.NUM_EXPERTS,
            top_k_experts=self.config.TOP_K_EXPERTS,
            device=self.device
        ).to(self.device)
        
        # Reward Model
        reward_model = RewardModel_encoder(
            len(self.tokenizer.tokens),
            self.config.D_MODEL,
            self.config.NHEAD,
            self.config.NUM_ENCODER_LAYERS,
            self.config.DIM_FEEDFORWARD,
            self.config.DROPOUT,
            self.config.NUM_EXPERTS,
            self.config.TOP_K_EXPERTS,
            self.device
        ).to(self.device)
        reward_model.load_state_dict(torch.load(self.config.MODEL_PATHS['reward']))
        reward_model.eval()
        
        return DDP(actor, device_ids=[self.rank]), reward_model

    def train_epoch(self, train_loader, epoch):
        """Main training loop for one epoch."""
        self.actor.train()
        metrics = {'loss': 0.0, 'policy_loss': 0.0, 'kl_div': 0.0, 'reward': 0.0}
        
        for batch_idx, (pro_tensors, pro_seqs) in enumerate(train_loader):
            # Update old policy periodically
            if self.global_step % self.config.POLICY_UPDATE_INTERVAL == 0:
                self.update_old_policy()
            
            # Forward pass through ESM-2
            esm_embeddings = self.get_esm_embeddings(pro_seqs)
            
            # Sampling and reward calculation
            with torch.no_grad():
                samples, old_logits = self.sample_actions(pro_tensors, esm_embeddings)
                rewards = self.calculate_rewards(samples)
                advantages = self.compute_advantages(rewards)
            
            # Policy optimization step
            loss_components = self.policy_optimization_step(
                pro_tensors, esm_embeddings, samples, old_logits, advantages
            )
            
            # Update metrics
            for key in metrics:
                metrics[key] += loss_components[key]
            self.global_step += 1
            
            # Log intermediate results
            if batch_idx % 50 == 0 and self.rank == 0:
                self.log_batch_stats(epoch, batch_idx, loss_components)
        
        # Normalize metrics and return
        return {k: v/len(train_loader) for k, v in metrics.items()}

    def policy_optimization_step(self, pro_tensors, esm_embeddings, samples, old_logits, advantages):
        """Calculate and optimize policy loss components."""
        self.optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Current policy log probabilities
            new_log_probs = self.calculate_log_probs(pro_tensors, esm_embeddings, samples)
            
            # Policy loss components
            policy_loss = self.calculate_policy_loss(new_log_probs, old_logits, advantages)
            kl_loss = self.calculate_kl_divergence(new_log_probs, old_logits)
            total_loss = policy_loss + self.config.KL_COEFFICIENT * kl_loss
        
        # Backpropagation
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_loss.item(),
            'reward': rewards.mean().item()
        }

    def evaluate(self, data_loader, split_name):
        """Evaluate model performance on a given dataset split."""
        if self.rank != 0:
            return 0.0  # Only evaluate on rank 0
        
        self.actor.eval()
        total_reward = 0.0
        with torch.no_grad():
            for pro_tensors, pro_seqs in data_loader:
                esm_embeddings = self.get_esm_embeddings(pro_seqs)
                samples, _ = self.sample_actions(pro_tensors, esm_embeddings)
                rewards = self.calculate_rewards(samples)
                total_reward += rewards.mean().item()
        
        return total_reward / len(data_loader)

def main():
    """Main execution flow for distributed GRPO training."""
    config = Config()
    rank, world_size, device = initialize_distributed()
    configure_environment(config)
    
    try:
        # Initialize trainer and data loaders
        trainer = GRPOTrainer(config, device, rank, world_size)
        datasets = load_and_preprocess_data(config)
        loaders = create_distributed_loaders(datasets, config, rank, world_size)
        
        # Training loop
        for epoch in range(config.NUM_EPOCHS):
            train_metrics = trainer.train_epoch(loaders['train'], epoch)
            
            # Validation
            val_reward = trainer.evaluate(loaders['val'], 'validation')
            test_reward = trainer.evaluate(loaders['test'], 'test')
            
            # Logging and model saving
            if rank == 0:
                logger.info(f"Epoch {epoch+1} Summary:")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
                logger.info(f"Validation Reward: {val_reward:.4f}")
                logger.info(f"Test Reward: {test_reward:.4f}")
                save_model(trainer.actor.module, config.MODEL_PATHS['output_prefix'], epoch)
                
    except Exception as e:
        logger.exception(f"Training failed: {e}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
