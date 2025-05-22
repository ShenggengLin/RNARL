import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data.distributed import DistributedSampler
import torch.optim.lr_scheduler as lr_scheduler
from transformer_encoder_MoE import Encoder,Encoder_nomoe
from itertools import chain
from torch.nn.parallel import parallel_apply
from typing import List, Dict, Tuple, Optional, Union



class Tokenizer:
    """Tokenizer for biological sequence encoding/decoding (protein and mRNA)."""
    
    def __init__(self):
        
        self.special_tokens = ['[START]', '[END]', '[PAD]', '[UNK]', '[SEG]']
        self.amino_acids = ['A', 'R', 'S', 'I', 'L', 'G', 'V', 'T', 'P', 'N', 
                           'D', 'C', 'Q', 'E', 'H', 'K', 'F', 'Y', 'M', 'W', '*']
        self.protein_alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        self.codons = [''.join([n1, n2, n3]) for n1 in 'UCAG' for n2 in 'UCAG' for n3 in 'UCAG']

        self.tokens = self.special_tokens + self.amino_acids + self.codons
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        self.padding_idx = self.token_to_id['[PAD]']
        self.start_idx = self.token_to_id['[START]']
        self.end_idx = self.token_to_id['[END]']
        self.unk_idx = self.token_to_id['[UNK]']
        self.seg_idx = self.token_to_id['[SEG]']
    
    def encode_pro(self, sequence: str, max_length: int) -> List[int]:
        
        ids = [self.start_idx] + [self.token_to_id.get(token, self.unk_idx) for token in sequence]
        
        if len(ids) < max_length - 1:
            ids.append(self.end_idx)
        else:
            ids = ids[:max_length-1] + [self.end_idx]
        
        return ids
    
    def encode_mrna(self, sequence: str, max_length: int) -> List[int]:
        
        ids = [self.start_idx]
        
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i+3]
            if len(codon) == 3 and codon in self.token_to_id:
                ids.append(self.token_to_id[codon])
            else:
                ids.append(self.unk_idx)
        
        if len(ids) < max_length - 1:
            ids.append(self.end_idx)
        else:
            ids = ids[:max_length-1] + [self.end_idx]
        
        return ids

    def decode(self, ids: List[int]) -> str:

        return ''.join([self.id_to_token.get(id, '[UNK]') for id in ids])
    
    def pad(self, ids: List[int], max_length: int) -> List[int]:

        padding_length = max_length - len(ids)
        if padding_length > 0:
            return ids + [self.padding_idx] * padding_length
        return ids



class BiologicalMappings:
    
    
    @staticmethod
    def get_codon_table() -> Dict[str, str]:
        
        return {
    'GCU':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A', 'CGU':'R', 'CGC':'R',   
    'CGA':'R', 'CGG':'R', 'AGA':'R', 'AGG':'R', 'UCU':'S', 'UCC':'S',
    'UCA':'S', 'UCG':'S', 'AGU':'S', 'AGC':'S', 'AUU':'I', 'AUC':'I',
    'AUA':'I', 'UUA':'L', 'UUG':'L', 'CUU':'L', 'CUC':'L', 'CUA':'L',
    'CUG':'L', 'GGU':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G', 'GUU':'V',
    'GUC':'V', 'GUA':'V', 'GUG':'V', 'ACU':'T', 'ACC':'T', 'ACA':'T',
    'ACG':'T', 'CCU':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P', 'AAU':'N',
    'AAC':'N', 'GAU':'D', 'GAC':'D', 'UGU':'C', 'UGC':'C', 'CAA':'Q',
    'CAG':'Q', 'GAA':'E', 'GAG':'E', 'CAU':'H', 'CAC':'H', 'AAA':'K',
    'AAG':'K', 'UUU':'F', 'UUC':'F', 'UAU':'Y', 'UAC':'Y', 'AUG':'M',
    'UGG':'W','UAG':'*', 'UGA':'*', 'UAA':'*'}
    
    @staticmethod
    def get_amino_acid_to_codon() -> Dict[str, List[str]]:
        
        return {
    'A':['GCU','GCC','GCA','GCG'], 'R':['CGU','CGC','CGA','CGG','AGA','AGG'],
    'S':['UCU','UCC','UCA','UCG','AGU','AGC'],'I':['AUU','AUC','AUA'],
    'L':['UUA','UUG','CUU','CUC','CUA','CUG'],'G':['GGU','GGC','GGA','GGG'],
    'V':['GUU','GUC','GUA','GUG'],'T':['ACU','ACC','ACA','ACG'],
    'P':['CCU','CCC','CCA','CCG'],'N':['AAU','AAC'],'D':['GAU','GAC'],
    'C':['UGU','UGC'],'Q':['CAA','CAG'],'E':['GAA','GAG'],'H':['CAU','CAC'],
    'K':['AAA','AAG'],'F':['UUU','UUC'],'Y':['UAU','UAC'],'M':['AUG'],'W':['UGG'],
    '*':['UAG','UGA','UAA']
}
    
    @staticmethod
    def create_token_mapping(tokenizer: Tokenizer) -> torch.Tensor:
        
        codon_table = BiologicalMappings.get_codon_table()
        token_codon_to_amino_acid = torch.full((len(tokenizer.tokens),), 
                                              tokenizer.unk_idx, 
                                              dtype=torch.long)
        
        for codon, amino_acid in codon_table.items():
            codon_id = tokenizer.token_to_id.get(codon, tokenizer.unk_idx)
            amino_acid_id = tokenizer.token_to_id.get(amino_acid, tokenizer.unk_idx)
            token_codon_to_amino_acid[codon_id] = amino_acid_id
            
        return token_codon_to_amino_acid


class ActorModel_encoder_noesm2(nn.Module):
    
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, dropout: float,
                 num_experts: int, top_k_experts: int, device: torch.device):
        
        super(ActorModel_encoder_noesm2, self).__init__()
        self.device = device
        
        
        self.amino_acid_to_codon = BiologicalMappings.get_amino_acid_to_codon()
        self.precomputed_masks = self._precompute_masks()
        
        
        self.encoder = Encoder(vocab_size, d_model, nhead, num_encoder_layers, 
                              dim_feedforward, dropout, num_experts, top_k_experts)
        
        
        self.mrna_output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, vocab_size)
        )



    def _precompute_masks(self) -> Dict[int, torch.Tensor]:
        
        tokenizer = Tokenizer()
        masks = {}
        
        for amino_acid, codons in self.amino_acid_to_codon.items():
            amino_acid_id = tokenizer.token_to_id.get(amino_acid, tokenizer.unk_idx)
            mask = torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=self.device)
            
            for codon in codons:
                codon_id = tokenizer.token_to_id.get(codon, tokenizer.unk_idx)
                if codon_id != tokenizer.unk_idx:
                    mask[codon_id] = True
                    
            masks[amino_acid_id] = mask
            
        return masks
        
    def forward(self, tokenizer_encoded_proteins: torch.Tensor) -> Tuple[torch.Tensor, list, torch.Tensor]:
        
        tokenizer = Tokenizer()
        src_padding_mask = (tokenizer_encoded_proteins == tokenizer.padding_idx)
        
        
        x, router_logits_list, entropy_loss = self.encoder(
            tokenizer_encoded_proteins, 
            src_key_padding_mask=src_padding_mask
        )
        
        
        batch_size, seq_len = tokenizer_encoded_proteins.shape
        
        amino_acid_to_codon_mask = torch.stack([
            self.precomputed_masks.get(
                tok.item(),
                torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=self.device)
            )
            for tok in tokenizer_encoded_proteins.reshape(-1)
        ]).view(batch_size, seq_len, -1)
        
        mrna_logits = self.mrna_output_layer(x)
        
        
        mrna_logits = mrna_logits.masked_fill(~amino_acid_to_codon_mask, -6.0e4)
        
        return mrna_logits, router_logits_list, entropy_loss

class ActorModel_encoder_esm2(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, esm2_dim: int,dropout: float,
                 num_experts: int, top_k_experts: int, device: torch.device):

        super(ActorModel_encoder_esm2, self).__init__()
        self.device = device
        
        
        self.amino_acid_to_codon = BiologicalMappings.get_amino_acid_to_codon()
        self.precomputed_masks = self._precompute_masks()
        
        self.dim_trans=nn.Linear(esm2_dim, d_model)
        
        self.encoder = Encoder(vocab_size, d_model, nhead, num_encoder_layers, 
                              dim_feedforward, dropout, num_experts, top_k_experts,if_embedding=False,if_pos_encoding=False)
        
        self.mrna_output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, vocab_size)
        )




    def _precompute_masks(self) -> Dict[int, torch.Tensor]:
        
        tokenizer = Tokenizer()
        masks = {}
        
        for amino_acid, codons in self.amino_acid_to_codon.items():
            amino_acid_id = tokenizer.token_to_id.get(amino_acid, tokenizer.unk_idx)
            mask = torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=self.device)
            
            for codon in codons:
                codon_id = tokenizer.token_to_id.get(codon, tokenizer.unk_idx)
                if codon_id != tokenizer.unk_idx:
                    mask[codon_id] = True
                    
            masks[amino_acid_id] = mask
            
        return masks
        
    def forward(self, tokenizer_encoded_proteins,esm2_encoded_proteins) -> Tuple[torch.Tensor, list, torch.Tensor]:

        
        tokenizer = Tokenizer()
        src_padding_mask = (tokenizer_encoded_proteins == tokenizer.padding_idx)
        
        x=self.dim_trans(esm2_encoded_proteins)

        x, router_logits_list, entropy_loss = self.encoder(
            x, 
            src_key_padding_mask=src_padding_mask
        )
        
        
        batch_size, seq_len = tokenizer_encoded_proteins.shape
        
        amino_acid_to_codon_mask = torch.stack([
            self.precomputed_masks.get(
                tok.item(),
                torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=self.device)
            )
            for tok in tokenizer_encoded_proteins.reshape(-1)
        ]).view(batch_size, seq_len, -1)
        
        
        mrna_logits = self.mrna_output_layer(x)
        
        mrna_logits = mrna_logits.masked_fill(~amino_acid_to_codon_mask, -6.0e4)
        
        return mrna_logits, router_logits_list, entropy_loss

    def get_embedding(self, tokenizer_encoded_proteins,esm2_encoded_proteins):

            
            tokenizer = Tokenizer()
            src_padding_mask = (tokenizer_encoded_proteins == tokenizer.padding_idx)
            
            x=self.dim_trans(esm2_encoded_proteins)

            x, router_logits_list, entropy_loss = self.encoder(
                x, 
                src_key_padding_mask=src_padding_mask
            )
            return x
    def get_router_logits(self, tokenizer_encoded_proteins,esm2_encoded_proteins):

            
            tokenizer = Tokenizer()
            src_padding_mask = (tokenizer_encoded_proteins == tokenizer.padding_idx)
            
            x=self.dim_trans(esm2_encoded_proteins)

            x, router_logits_list, entropy_loss = self.encoder(
                x, 
                src_key_padding_mask=src_padding_mask
            )
            return router_logits_list

class ActorModel_encoder_nomoe(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int,  esm2_dim: int,dropout: float, device: torch.device):
        super(ActorModel_encoder_nomoe, self).__init__()
        self.device = device
        
        self.amino_acid_to_codon = BiologicalMappings.get_amino_acid_to_codon()
        self.precomputed_masks = self._precompute_masks()
        
        self.dim_trans=nn.Linear(esm2_dim, d_model)

        self.encoder = Encoder_nomoe(vocab_size, d_model, nhead, num_encoder_layers, 
                              dim_feedforward, dropout,if_embedding=False,if_pos_encoding=False)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, vocab_size)
        )

    def _precompute_masks(self) -> Dict[int, torch.Tensor]:
        
        tokenizer = Tokenizer()
        masks = {}
        
        for amino_acid, codons in self.amino_acid_to_codon.items():
            amino_acid_id = tokenizer.token_to_id.get(amino_acid, tokenizer.unk_idx)
            mask = torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=self.device)
            
            for codon in codons:
                codon_id = tokenizer.token_to_id.get(codon, tokenizer.unk_idx)
                if codon_id != tokenizer.unk_idx:
                    mask[codon_id] = True
                    
            masks[amino_acid_id] = mask
            
        return masks
        
    def forward(self, tokenizer_encoded_proteins,esm2_encoded_proteins):
        
        tokenizer = Tokenizer()
        src_padding_mask = (tokenizer_encoded_proteins == tokenizer.padding_idx)

        x=self.dim_trans(esm2_encoded_proteins)
        
        x= self.encoder(
            x, 
            src_key_padding_mask=src_padding_mask
        )
        
        batch_size, seq_len = tokenizer_encoded_proteins.shape
        
        amino_acid_to_codon_mask = torch.stack([
            self.precomputed_masks.get(
                tok.item(),
                torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=self.device)
            )
            for tok in tokenizer_encoded_proteins.reshape(-1)
        ]).view(batch_size, seq_len, -1)
        
        
        logits = self.output_layer(x)
        
        logits = logits.masked_fill(~amino_acid_to_codon_mask, -6.0e4)
        
        return logits

class RewardModel_encoder(nn.Module):
    def __init__(self, vocab_size,  d_model, nhead, num_encoder_layers,  dim_feedforward,dropout,num_experts,top_k_experts,device):
        super(RewardModel_encoder, self).__init__()
        self.tokenizer=Tokenizer()
        self.device=device
        
        self.encoder = Encoder(vocab_size, d_model, nhead, num_encoder_layers, 
                              dim_feedforward, dropout, num_experts, top_k_experts)
        self.reward_output_layer = nn.Sequential(
                                            nn.Linear(d_model, d_model//2),
                                            nn.LayerNorm(d_model//2),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(d_model//2, 1)
                                        )
        
        
    def forward(self, tokenizer_encoded_mrnas):

        src_padding_mask = (tokenizer_encoded_mrnas==self.tokenizer.padding_idx)

        x,router_logits_list,entropy_loss = self.encoder(tokenizer_encoded_mrnas, src_key_padding_mask=src_padding_mask)
        
        
        reward=self.reward_output_layer(x)
        reward=reward[:,0,:].squeeze()
        
        return reward,router_logits_list,entropy_loss



class LengthAwareDistributedSampler_human(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        weights = np.ones(len(self.lengths))
        weights[np.array(self.lengths) >= 1300] = 85.64*200  
        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 5.02*200
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 4.36*100
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 3.63*100
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 3.15
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 2.20
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 1.64 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 1.36 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.75 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.63
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.60 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.71
        weights[np.array(self.lengths) < 100] = 3.68*100     
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)

class LengthAwareDistributedSampler_Arabidopsis(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        
        weights = np.ones(len(self.lengths))
        weights[np.array(self.lengths) >= 1300] = 630.75*20  
        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 17.05*20
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 11.52*20
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 7.17*10
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 5.56*10
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 3.54
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 2.51 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 1.62 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.68 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.49
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.49 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.49
        weights[np.array(self.lengths) < 100] = 1.23*10   
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)


class LengthAwareDistributedSampler_CR(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        
        weights = np.ones(len(self.lengths))
        weights[np.array(self.lengths) >= 1300] = 61.55*20 
        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 3.66*20
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 2.96*10
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 2.54*10
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 2.11*10
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 1.79
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 1.39 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 1.11 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.82 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.73
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.67 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.66
        weights[np.array(self.lengths) < 100] = 1.18*10     
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)

class LengthAwareDistributedSampler_PC(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        
        weights = np.ones(len(self.lengths))
        weights[np.array(self.lengths) >= 1300] = 318.0*200  
        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 13.98*200
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 10.26*100
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 7.62*100
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 6.14*100
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 3.80
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 2.67 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 1.88 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.88 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.75
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.76 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.83
        weights[np.array(self.lengths) < 100] = 1.87*100     
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)

class LengthAwareDistributedSampler_EscherichiaColi(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        
        weights = np.ones(len(self.lengths))
        weights[np.array(self.lengths) >= 1300] = 211.0*200  
        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 26.38*200
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 15.07*100
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 11.72*100
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 11.11*100
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 4.06
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 2.81 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 2.07 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.46 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.30
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.25 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.25
        weights[np.array(self.lengths) < 100] = 0.47    
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)

class LengthAwareDistributedSampler_TK(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        
        weights = np.ones(len(self.lengths))

        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 12.25*10
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 8.17*10
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 24.5*10
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 8.17*10
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 3.27
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 2.33 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 1.09 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.25 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.17
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.13 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.10
        weights[np.array(self.lengths) < 100] = 0.22   
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)



class LengthAwareDistributedSampler_human_circ(DistributedSampler):
    def __init__(self, dataset, lengths, data_num_rat=None,num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        self.lengths = lengths
        self.weights = self.calculate_weights()
        self.data_num_rat=data_num_rat
        self.total_size = int(len(dataset) * data_num_rat)

    def calculate_weights(self):
        
        weights = np.ones(len(self.lengths))
        weights[np.array(self.lengths) >= 1300] = 89.62*20
        weights[(np.array(self.lengths) >= 1200) & (np.array(self.lengths) < 1300)] = 5.24*20
        weights[(np.array(self.lengths) >= 1100) & (np.array(self.lengths) < 1200)] = 4.58*10
        weights[(np.array(self.lengths) >= 1000) & (np.array(self.lengths) < 1100)] = 3.82*10
        weights[(np.array(self.lengths) >= 900) & (np.array(self.lengths) < 1000)] = 3.30
        weights[(np.array(self.lengths) >= 800) & (np.array(self.lengths) < 900)] = 2.34
        weights[(np.array(self.lengths) >= 700) & (np.array(self.lengths) < 800)] = 1.74 
        weights[(np.array(self.lengths) >= 600) & (np.array(self.lengths) < 700)] = 1.36 
        weights[(np.array(self.lengths) >= 500) & (np.array(self.lengths) < 600)] = 1.0 
        weights[(np.array(self.lengths) >= 400) & (np.array(self.lengths) < 500)] = 0.74 
        weights[(np.array(self.lengths) >= 300) & (np.array(self.lengths) < 400)] = 0.57
        weights[(np.array(self.lengths) >= 200) & (np.array(self.lengths) < 300)] = 0.46 
        weights[(np.array(self.lengths) >= 100) & (np.array(self.lengths) < 200)] = 0.38
        weights[np.array(self.lengths) < 100] = 0.48  
        
        return weights / np.sum(weights)

    def __iter__(self):
        
        indices = np.random.choice(len(self.dataset), self.total_size,  replace=True, p=self.weights)
        
        
        total_size_local = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total_size_local]

        
        indices = indices[self.rank:total_size_local:self.num_replicas]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
