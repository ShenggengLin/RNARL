import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parallel import parallel_apply
from typing import Tuple, List, Optional, Union
import torch.utils.checkpoint as checkpoint


class MultiHeadAttention(nn.Module):
    
    def __init__(self, model_dim: int, n_heads: int):
        super().__init__()
        assert model_dim % n_heads == 0, "model_dim must be divisible by n_heads"
        
        self.model_dim = model_dim
        self.d_k = model_dim // n_heads
        self.n_heads = n_heads
        
        
        self.qkv_linear = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_linear = nn.Linear(model_dim, model_dim, bias=False)
        
        
        nn.init.xavier_uniform_(self.qkv_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = q.size(0)
        
        
        is_self_attention = q.data_ptr() == k.data_ptr() == v.data_ptr()
        if is_self_attention:
            
            qkv = self.qkv_linear(q).chunk(3, dim=-1)
            q, k, v = map(
                lambda x: x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2), 
                qkv
            )
        else:
            
            q = self.qkv_linear(q)[:, :, :self.model_dim]
            k = self.qkv_linear(k)[:, :, self.model_dim:2*self.model_dim]
            v = self.qkv_linear(v)[:, :, 2*self.model_dim:]
            
            q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -6.0e4)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -6.0e4)

        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        output = self.out_linear(context)
        
        return output


class MoE(nn.Module):
    
    
    def __init__(self, d_model: int, num_experts: int, d_ff: int, dropout: float, top_k: int):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.d_model = d_model
        
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.GELU(),                              
                nn.Dropout(dropout),                    
                nn.Linear(d_ff, d_model, bias=False)   
            ) for _ in range(num_experts)
        ])
        
        
        for expert in self.experts:
            nn.init.kaiming_uniform_(expert[0].weight, a=math.sqrt(5))  
            nn.init.zeros_(expert[3].weight)  
            
        nn.init.zeros_(self.gate.weight)  
            
    def orthogonal_loss(self) -> torch.Tensor:
        
        total_loss = 0.0
        num_pairs = 0
        
        
        expert_weights_1 = torch.stack([expert[0].weight for expert in self.experts])
        
        expert_weights_2 = torch.stack([expert[3].weight for expert in self.experts])
        
        
        for i in range(self.num_experts):
            w1_i = expert_weights_1[i]  
            w2_i = expert_weights_2[i]  
            
            for j in range(i+1, self.num_experts):
                w1_j = expert_weights_1[j]  
                w2_j = expert_weights_2[j]  
                
                
                w1_sim = torch.sum((w1_i @ w1_j.T)**2) / (w1_i.size(0) * w1_j.size(0))  
                
                w2_sim = torch.sum((w2_i.T @ w2_j)**2) / (w2_i.size(1) * w2_j.size(1))  
                
                total_loss += (w1_sim + w2_sim) / 2  
                num_pairs += 1
                
        return total_loss / max(num_pairs, 1)  

    def entropy_regularization_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        
        
        log_probs = torch.log(torch.clamp(routing_probs, min=1e-6))  
        
        entropy = -torch.sum(routing_probs * log_probs, dim=-1)  
        return entropy.mean()  

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, d_model = hidden_states.shape
        combined_batch_size = batch_size * seq_len
        
        flat_hidden = hidden_states.reshape(combined_batch_size, d_model)

        
        router_logits = self.gate(flat_hidden)  
        routing_probs = F.softmax(router_logits, dim=-1)
        
        
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        
        
        flat_expert_inputs = [flat_hidden] * self.num_experts
        expert_outputs = parallel_apply(self.experts, flat_expert_inputs)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        
        expert_weights_matrix = torch.zeros(
            combined_batch_size, self.num_experts, device=hidden_states.device
        )
        
        for k in range(self.top_k):
            k_indices = selected_experts[:, k]
            k_weights = routing_weights[:, k].unsqueeze(1)
            
            expert_weights_matrix.scatter_add_(
                1, 
                k_indices.unsqueeze(1),
                k_weights
            )
        
        
        combined_output = torch.bmm(
            expert_weights_matrix.unsqueeze(1),   
            expert_outputs                         
        ).squeeze(1)
        
        
        output = combined_output.reshape(batch_size, seq_len, d_model)  
        
        
        entropy_loss = self.entropy_regularization_loss(routing_probs)
        
        return output, router_logits, entropy_loss



class EncoderLayer(nn.Module):
    
    
    def __init__(self, model_dim: int, n_heads: int, ff_hidden_dim: int, 
                 dropout: float, num_experts: int, top_k: int):
        super().__init__()
        self.model_dim = model_dim
        
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        self.self_attn = MultiHeadAttention(model_dim, n_heads)
        self.moe = MoE(model_dim, num_experts, ff_hidden_dim, dropout, top_k)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.use_projection = False
        if not self.use_projection:
            self.residual_scale = nn.Parameter(torch.ones(1))

    def _sa_block(self, x: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None, 
                 key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = self.self_attn(x, x, x, mask=mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)
    
    def _moe_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return self.moe(x)

    def forward(self, 
                x: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                use_checkpoint: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        normalized_x = self.norm1(x)
        
        
        if use_checkpoint and self.training:
            attn_output = checkpoint.checkpoint(
                self._sa_block, normalized_x, src_mask, src_key_padding_mask
            )
        else:
            attn_output = self._sa_block(normalized_x, src_mask, src_key_padding_mask)
        
        
        x = x + attn_output * self.residual_scale
        
        
        normalized_x = self.norm2(x)
        
        
        if use_checkpoint and self.training:
            moe_output, router_logits, entropy_loss = checkpoint.checkpoint(
                self._moe_block, normalized_x
            )
        else:
            moe_output, router_logits, entropy_loss = self._moe_block(normalized_x)
        
        x = x + self.dropout2(moe_output) * self.residual_scale
        
        return x, router_logits, entropy_loss


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderLayer_nomoe(nn.Module):

    def __init__(self, model_dim: int, n_heads: int, ff_hidden_dim: int, 
                 dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        self.self_attn = MultiHeadAttention(model_dim, n_heads)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_hidden_dim, dropout)

        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        normalized_x = self.norm1(x)
        
        attn_output = self.self_attn(normalized_x, normalized_x, normalized_x, src_mask,src_key_padding_mask)
        
        x = x + self.dropout1(attn_output)
        
        normalized_x = self.norm2(x)
        
        ff_output = self.feed_forward(normalized_x)
        
        x = x + self.dropout2(ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        pos_encoding = self.pe[:, :x.size(1)]
        x = x + pos_encoding
        return self.dropout(x)


class Encoder(nn.Module):
    
    
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 n_heads: int, 
                 num_layers: int, 
                 ff_hidden_dim: int, 
                 dropout: float,
                 num_experts: int,
                 top_k: int,
                 if_embedding: bool = True,
                 if_pos_encoding: bool = True,
                 use_checkpointing: bool = False):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.if_embedding = if_embedding
        self.if_pos_encoding = if_pos_encoding
        self.use_checkpointing = use_checkpointing
        
        
        if if_embedding:
            self.embedding = nn.Embedding(input_dim, model_dim)
            
            nn.init.normal_(self.embedding.weight, mean=0, std=model_dim**-0.5)
        
        
        if if_pos_encoding:
            self.pos_encoding = PositionalEncoding(model_dim, dropout)
        
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                model_dim, n_heads, ff_hidden_dim, dropout, num_experts, top_k
            ) for _ in range(num_layers)
        ])
        
        
        self.final_norm = nn.LayerNorm(model_dim)
        
    def forward(self, 
                src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List, float]:
        
        
        if self.if_embedding:
            x = self.embedding(src) * math.sqrt(self.model_dim)
        else:
            x = src
            
        
        if self.if_pos_encoding:
            x = self.pos_encoding(x)
        
        
        total_entropy_loss = 0.0
        router_logits_list = []
        
        
        for layer in self.layers:
            x, router_logits, entropy_loss = layer(
                x, 
                src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask,
                use_checkpoint=self.use_checkpointing
            )
            total_entropy_loss += entropy_loss
            
            
            if not self.training:
                router_logits_list.append(router_logits.detach().cpu().tolist())
        
        
        x = self.final_norm(x)
        
        
        avg_entropy_loss = total_entropy_loss / self.num_layers
        
        return x, router_logits_list, avg_entropy_loss


class Encoder_nomoe(nn.Module):
    
    
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 n_heads: int, 
                 num_layers: int, 
                 ff_hidden_dim: int, 
                 dropout: float,
                 if_embedding: bool = True,
                 if_pos_encoding: bool = True):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.if_embedding = if_embedding
        self.if_pos_encoding = if_pos_encoding
        
        
        if if_embedding:
            self.embedding = nn.Embedding(input_dim, model_dim)
            
            nn.init.normal_(self.embedding.weight, mean=0, std=model_dim**-0.5)
        
        
        if if_pos_encoding:
            self.pos_encoding = PositionalEncoding(model_dim, dropout)
        
        
        self.layers = nn.ModuleList([
            EncoderLayer_nomoe(
                model_dim, n_heads, ff_hidden_dim, dropout
            ) for _ in range(num_layers)
        ])
        
        
        self.final_norm = nn.LayerNorm(model_dim)
        
    def forward(self, 
                src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List, float]:

        
        if self.if_embedding:
            x = self.embedding(src) * math.sqrt(self.model_dim)
        else:
            x = src
            
        
        if self.if_pos_encoding:
            x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(
                x, 
                src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask
            )
        
        x = self.final_norm(x)
        
        return x