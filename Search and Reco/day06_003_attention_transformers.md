# Day 6.3: Attention Mechanisms and Transformer Models

## Learning Objectives
- Master self-attention mechanisms for recommendation systems
- Implement transformer architectures for sequential recommendations
- Design BERT-style pre-training approaches for RecSys
- Build multi-head attention with positional encodings
- Explore transformer variants optimized for recommendations
- Develop scalable attention mechanisms for large-scale systems

## 1. Self-Attention Mechanisms for Recommendations

### Basic Self-Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SelfAttention(nn.Module):
    """Self-attention mechanism for recommendation systems"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len)
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections
        Q = self.q_linear(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # Add head dimension
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final projection
        output = self.out_proj(attention_output)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for sequential data"""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 5000):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, embed_dim)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)

class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = SelfAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: attention mask
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (SASRec)"""
    
    def __init__(self, n_items: int, embed_dim: int = 64, num_heads: int = 2,
                 num_blocks: int = 2, max_seq_len: int = 200, dropout: float = 0.1):
        super().__init__()
        
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Item embedding (with padding)
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_blocks)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, seq_len) - item sequences
        Returns:
            output: (batch_size, seq_len, n_items) - next item predictions
        """
        batch_size, seq_len = sequences.size()
        device = sequences.device
        
        # Create attention mask (causal mask for autoregressive prediction)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Item embeddings
        item_emb = self.item_embedding(sequences)  # (batch_size, seq_len, embed_dim)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Project to item space (using item embedding as projection matrix)
        output = torch.matmul(x, self.item_embedding.weight[1:].transpose(0, 1))  # Exclude padding
        
        return output

class BERT4Rec(nn.Module):
    """BERT for Sequential Recommendation"""
    
    def __init__(self, n_items: int, embed_dim: int = 64, num_heads: int = 2,
                 num_blocks: int = 2, max_seq_len: int = 200, dropout: float = 0.1,
                 mask_prob: float = 0.15):
        super().__init__()
        
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        # Special tokens
        self.pad_token = 0
        self.mask_token = n_items + 1
        
        # Item embedding (with padding and mask tokens)
        self.item_embedding = nn.Embedding(n_items + 2, embed_dim, padding_idx=0)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_blocks)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Prediction head
        self.prediction_head = nn.Linear(embed_dim, n_items + 1)  # +1 for padding
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def mask_sequence(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masking to sequences"""
        batch_size, seq_len = sequences.size()
        device = sequences.device
        
        # Create copy for masking
        masked_sequences = sequences.clone()
        labels = sequences.clone()
        
        # Create random mask
        rand = torch.rand(batch_size, seq_len, device=device)
        
        # Don't mask padding tokens
        mask_candidates = (sequences != self.pad_token) & (rand < self.mask_prob)
        
        # 80% of the time, replace with [MASK] token
        mask_80 = mask_candidates & (torch.rand(batch_size, seq_len, device=device) < 0.8)
        masked_sequences[mask_80] = self.mask_token
        
        # 10% of the time, replace with random item
        mask_10 = mask_candidates & ~mask_80 & (torch.rand(batch_size, seq_len, device=device) < 0.5)
        random_items = torch.randint(1, self.n_items + 1, (batch_size, seq_len), device=device)
        masked_sequences[mask_10] = random_items[mask_10]
        
        # 10% of the time, keep original
        # (already handled by not changing these positions)
        
        # Only compute loss on masked positions
        labels[~mask_candidates] = -100  # Ignore in loss computation
        
        return masked_sequences, labels
    
    def forward(self, sequences: torch.Tensor, masked: bool = False) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, seq_len) - item sequences
            masked: whether to apply masking during training
        Returns:
            output: (batch_size, seq_len, n_items+1) - item predictions
        """
        batch_size, seq_len = sequences.size()
        device = sequences.device
        
        if masked and self.training:
            sequences, _ = self.mask_sequence(sequences)
        
        # Create attention mask (bidirectional)
        attention_mask = (sequences != self.pad_token).float()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Item embeddings
        item_emb = self.item_embedding(sequences)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Prediction
        output = self.prediction_head(x)
        
        return output
```

## 2. Advanced Transformer Architectures

### Multi-Interest Transformer

```python
class MultiInterestTransformer(nn.Module):
    """Multi-Interest Transformer for diverse user interests"""
    
    def __init__(self, n_items: int, embed_dim: int = 64, num_interests: int = 4,
                 num_heads: int = 2, num_blocks: int = 2, max_seq_len: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        self.max_seq_len = max_seq_len
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks for sequence encoding
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_blocks)
        ])
        
        # Multi-interest extraction
        self.interest_queries = nn.Parameter(torch.randn(num_interests, embed_dim))
        self.interest_attention = SelfAttention(embed_dim, num_heads, dropout)
        
        # Interest fusion
        self.interest_fusion = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequences: torch.Tensor, target_item: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, seq_len)
            target_item: (batch_size,) - for interest fusion
        """
        batch_size, seq_len = sequences.size()
        device = sequences.device
        
        # Create padding mask
        padding_mask = (sequences != 0).float()
        
        # Item and positional embeddings
        item_emb = self.item_embedding(sequences)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = self.layer_norm(x)
        
        # Extract multiple interests
        # Expand interest queries for batch
        interest_queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)
        # (batch_size, num_interests, embed_dim)
        
        # Apply attention to extract interests from sequence
        interest_representations, _ = self.interest_attention(
            interest_queries, x, x, mask=padding_mask.unsqueeze(1).unsqueeze(2)
        )
        # (batch_size, num_interests, embed_dim)
        
        if target_item is not None:
            # Fusion based on target item
            target_emb = self.item_embedding(target_item).unsqueeze(1)  # (batch_size, 1, embed_dim)
            
            fused_interest, _ = self.interest_fusion(
                target_emb.transpose(0, 1),  # (1, batch_size, embed_dim)
                interest_representations.transpose(0, 1),  # (num_interests, batch_size, embed_dim)
                interest_representations.transpose(0, 1)
            )
            
            return fused_interest.transpose(0, 1).squeeze(1)  # (batch_size, embed_dim)
        else:
            # Return all interests
            return interest_representations

class HierarchicalTransformer(nn.Module):
    """Hierarchical Transformer for multi-scale temporal patterns"""
    
    def __init__(self, n_items: int, embed_dim: int = 64, num_heads: int = 2,
                 num_blocks_low: int = 2, num_blocks_high: int = 2,
                 max_seq_len: int = 200, window_size: int = 10, dropout: float = 0.1):
        super().__init__()
        
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        
        # Positional embeddings for different levels
        self.pos_embedding_low = nn.Embedding(window_size, embed_dim)
        self.pos_embedding_high = nn.Embedding(max_seq_len // window_size + 1, embed_dim)
        
        # Low-level transformer (within windows)
        self.low_level_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_blocks_low)
        ])
        
        # High-level transformer (across windows)
        self.high_level_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_blocks_high)
        ])
        
        # Layer normalization
        self.layer_norm_low = nn.LayerNorm(embed_dim)
        self.layer_norm_high = nn.LayerNorm(embed_dim)
        
        # Aggregation
        self.window_aggregation = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, n_items)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, seq_len)
        """
        batch_size, seq_len = sequences.size()
        device = sequences.device
        
        # Pad sequence to be divisible by window_size
        padded_len = ((seq_len - 1) // self.window_size + 1) * self.window_size
        if padded_len > seq_len:
            padding = torch.zeros(batch_size, padded_len - seq_len, dtype=torch.long, device=device)
            sequences = torch.cat([sequences, padding], dim=1)
        
        # Reshape into windows
        num_windows = padded_len // self.window_size
        windowed_sequences = sequences.view(batch_size, num_windows, self.window_size)
        # (batch_size, num_windows, window_size)
        
        # Process each window with low-level transformer
        window_representations = []
        
        for window_idx in range(num_windows):
            window = windowed_sequences[:, window_idx, :]  # (batch_size, window_size)
            
            # Item embeddings
            item_emb = self.item_embedding(window)
            
            # Low-level positional embeddings
            positions = torch.arange(self.window_size, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding_low(positions)
            
            x = item_emb + pos_emb
            x = self.dropout(x)
            
            # Apply low-level transformer blocks
            for block in self.low_level_blocks:
                x = block(x)
            
            x = self.layer_norm_low(x)
            
            # Aggregate window representation (mean pooling)
            window_repr = x.mean(dim=1)  # (batch_size, embed_dim)
            window_representations.append(window_repr)
        
        # Stack window representations
        high_level_input = torch.stack(window_representations, dim=1)  # (batch_size, num_windows, embed_dim)
        
        # High-level positional embeddings
        high_positions = torch.arange(num_windows, device=device).unsqueeze(0).expand(batch_size, -1)
        high_pos_emb = self.pos_embedding_high(high_positions)
        
        high_level_input = high_level_input + high_pos_emb
        high_level_input = self.dropout(high_level_input)
        
        # Apply high-level transformer blocks
        for block in self.high_level_blocks:
            high_level_input = block(high_level_input)
        
        high_level_input = self.layer_norm_high(high_level_input)
        
        # Final representation (last window)
        final_repr = high_level_input[:, -1, :]  # (batch_size, embed_dim)
        
        # Output projection
        output = self.output_proj(final_repr)
        
        return output

class CrossDomainTransformer(nn.Module):
    """Cross-domain Transformer for multi-domain recommendations"""
    
    def __init__(self, domain_vocab_sizes: Dict[str, int], embed_dim: int = 64,
                 num_heads: int = 2, num_blocks: int = 2, max_seq_len: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        
        self.domains = list(domain_vocab_sizes.keys())
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Domain-specific embeddings
        self.domain_embeddings = nn.ModuleDict({
            domain: nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for domain, vocab_size in domain_vocab_sizes.items()
        })
        
        # Domain type embeddings
        self.domain_type_embedding = nn.Embedding(len(self.domains), embed_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Shared transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_blocks)
        ])
        
        # Domain-specific output heads
        self.output_heads = nn.ModuleDict({
            domain: nn.Linear(embed_dim, vocab_size)
            for domain, vocab_size in domain_vocab_sizes.items()
        })
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequences: Dict[str, torch.Tensor],
                domain_types: torch.Tensor, target_domain: str) -> torch.Tensor:
        """
        Args:
            sequences: dict of (batch_size, seq_len) for each domain
            domain_types: (batch_size, seq_len) - domain type indices
            target_domain: target domain for prediction
        """
        batch_size, seq_len = next(iter(sequences.values())).size()
        device = next(iter(sequences.values())).device
        
        # Combine embeddings from all domains
        combined_embeddings = torch.zeros(batch_size, seq_len, self.embed_dim, device=device)
        
        for domain_idx, domain in enumerate(self.domains):
            if domain in sequences:
                domain_mask = (domain_types == domain_idx).float().unsqueeze(-1)
                domain_emb = self.domain_embeddings[domain](sequences[domain])
                combined_embeddings += domain_emb * domain_mask
        
        # Add domain type embeddings
        domain_type_emb = self.domain_type_embedding(domain_types)
        combined_embeddings += domain_type_emb
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        combined_embeddings += pos_emb
        
        x = self.dropout(combined_embeddings)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        
        # Use last position for prediction
        final_repr = x[:, -1, :]  # (batch_size, embed_dim)
        
        # Domain-specific prediction
        output = self.output_heads[target_domain](final_repr)
        
        return output
```

## 3. Scalable Attention Mechanisms

### Linear Attention for Large-Scale Systems

```python
class LinearAttention(nn.Module):
    """Linear attention mechanism for scalable recommendations"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Linear attention: O(n) complexity instead of O(n^2)
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply ELU + 1 for positive values
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention computation
        # Instead of Q @ K^T @ V, compute Q @ (K^T @ V)
        KV = torch.matmul(K.transpose(-2, -1), V)  # (batch_size, num_heads, head_dim, head_dim)
        attention_output = torch.matmul(Q, KV)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Normalize
        normalizer = torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1))
        attention_output = attention_output / (normalizer + 1e-8)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_proj(attention_output)
        
        return output

class SparseAttention(nn.Module):
    """Sparse attention mechanism for handling long sequences"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, block_size: int = 64,
                 num_random_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        
        # Linear projections
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = math.sqrt(self.head_dim)
        
    def create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention pattern"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # Local attention (within blocks)
        for i in range(0, seq_len, self.block_size):
            end = min(i + self.block_size, seq_len)
            mask[i:end, i:end] = 1
        
        # Strided attention (every k-th position)
        stride = max(1, seq_len // (self.block_size * 2))
        for i in range(0, seq_len, stride):
            mask[:, i] = 1
        
        # Random attention
        for _ in range(self.num_random_blocks):
            start = torch.randint(0, max(1, seq_len - self.block_size), (1,)).item()
            end = min(start + self.block_size, seq_len)
            mask[:, start:end] = 1
        
        return mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = query.size()
        device = query.device
        
        # Create sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, device)
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply sparse mask
        attention_scores = attention_scores.masked_fill(
            sparse_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9
        )
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_proj(attention_output)
        
        return output

class AdaptiveAttention(nn.Module):
    """Adaptive attention that adjusts based on sequence characteristics"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Standard attention
        self.standard_attention = SelfAttention(embed_dim, num_heads, dropout)
        
        # Linear attention
        self.linear_attention = LinearAttention(embed_dim, num_heads, dropout)
        
        # Gating mechanism to choose between attention types
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Compute both attention types
        standard_out, _ = self.standard_attention(query, key, value)
        linear_out = self.linear_attention(query, key, value)
        
        # Compute gate values based on query
        gate_values = self.gate(query.mean(dim=1, keepdim=True))  # (batch_size, 1, 1)
        
        # Adaptive combination
        output = gate_values * standard_out + (1 - gate_values) * linear_out
        
        return output
```

## 4. Training and Evaluation Framework

### Transformer Training Pipeline

```python
class TransformerTrainer:
    """Training pipeline for transformer-based recommendation models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'metrics': []
        }
    
    def create_sasrec_data(self, interactions: pd.DataFrame, seq_len: int = 50) -> torch.utils.data.Dataset:
        """Create dataset for SASRec-style training"""
        
        class SASRecDataset(torch.utils.data.Dataset):
            def __init__(self, interactions, seq_len):
                self.interactions = interactions
                self.seq_len = seq_len
                
                # Group by user
                self.user_sequences = interactions.groupby('user_id')['item_id'].apply(list).to_dict()
                self.users = list(self.user_sequences.keys())
                
            def __len__(self):
                return len(self.users)
            
            def __getitem__(self, idx):
                user_id = self.users[idx]
                sequence = self.user_sequences[user_id]
                
                # Truncate or pad sequence
                if len(sequence) > self.seq_len:
                    sequence = sequence[-self.seq_len:]
                else:
                    sequence = [0] * (self.seq_len - len(sequence)) + sequence
                
                # Create input and target
                input_seq = sequence[:-1] if len(sequence) > 1 else sequence
                target = sequence[-1] if len(sequence) > 1 else sequence[0]
                
                # Ensure input_seq has correct length
                if len(input_seq) < self.seq_len - 1:
                    input_seq = [0] * (self.seq_len - 1 - len(input_seq)) + input_seq
                elif len(input_seq) > self.seq_len - 1:
                    input_seq = input_seq[-(self.seq_len - 1):]
                
                return {
                    'sequence': torch.tensor(input_seq, dtype=torch.long),
                    'target': torch.tensor(target, dtype=torch.long)
                }
        
        return SASRecDataset(interactions, seq_len)
    
    def create_bert4rec_data(self, interactions: pd.DataFrame, seq_len: int = 50) -> torch.utils.data.Dataset:
        """Create dataset for BERT4Rec-style training"""
        
        class BERT4RecDataset(torch.utils.data.Dataset):
            def __init__(self, interactions, seq_len):
                self.interactions = interactions
                self.seq_len = seq_len
                
                # Group by user
                self.user_sequences = interactions.groupby('user_id')['item_id'].apply(list).to_dict()
                
                # Create all possible subsequences
                self.sequences = []
                for user_id, sequence in self.user_sequences.items():
                    if len(sequence) >= 3:  # Minimum sequence length
                        for i in range(3, len(sequence) + 1):
                            subseq = sequence[:i]
                            if len(subseq) <= seq_len:
                                # Pad if necessary
                                padded_seq = [0] * (seq_len - len(subseq)) + subseq
                                self.sequences.append(padded_seq)
                
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                sequence = self.sequences[idx]
                return {
                    'sequence': torch.tensor(sequence, dtype=torch.long)
                }
        
        return BERT4RecDataset(interactions, seq_len)
    
    def train_sasrec(self, train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader = None,
                     n_epochs: int = 100, learning_rate: float = 0.001,
                     weight_decay: float = 1e-4) -> Dict[str, List]:
        """Train SASRec model"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(sequences)  # (batch_size, seq_len, n_items)
                
                # Use last position for prediction
                predictions = logits[:, -1, :]  # (batch_size, n_items)
                
                loss = criterion(predictions, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate_sasrec(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.4f}")
        
        return self.training_history
    
    def train_bert4rec(self, train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader = None,
                       n_epochs: int = 100, learning_rate: float = 0.001,
                       weight_decay: float = 1e-4) -> Dict[str, List]:
        """Train BERT4Rec model"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                
                optimizer.zero_grad()
                
                # Apply masking and get predictions
                masked_sequences, labels = self.model.mask_sequence(sequences)
                logits = self.model(masked_sequences, masked=False)
                
                # Compute loss only on masked positions
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")
        
        return self.training_history
    
    def _evaluate_sasrec(self, data_loader: torch.utils.data.DataLoader, criterion) -> float:
        """Evaluate SASRec model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                logits = self.model(sequences)
                predictions = logits[:, -1, :]
                
                loss = criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)

def demonstrate_transformers():
    """Demonstrate transformer models for recommendations"""
    
    print("🤖 Demonstrating Transformer Models for Recommendations...")
    
    # Create sample data
    np.random.seed(42)
    n_users, n_items = 1000, 500
    n_interactions = 10000
    
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(1, n_items + 1, n_interactions),  # 1-indexed items
        'timestamp': pd.date_range('2023-01-01', periods=n_interactions, freq='H')
    })
    
    # Sort by user and timestamp
    interactions = interactions.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    print(f"📊 Created interaction data with {len(interactions)} interactions")
    print(f"👥 Users: {n_users}, 📦 Items: {n_items}")
    
    # Test SASRec
    print("\n🚀 Testing SASRec...")
    sasrec_model = SASRec(n_items, embed_dim=64, num_heads=2, num_blocks=2)
    print(f"✅ SASRec model created with {sum(p.numel() for p in sasrec_model.parameters())} parameters")
    
    # Test BERT4Rec
    print("\n🚀 Testing BERT4Rec...")
    bert4rec_model = BERT4Rec(n_items, embed_dim=64, num_heads=2, num_blocks=2)
    print(f"✅ BERT4Rec model created with {sum(p.numel() for p in bert4rec_model.parameters())} parameters")
    
    # Test Multi-Interest Transformer
    print("\n🚀 Testing Multi-Interest Transformer...")
    multi_interest_model = MultiInterestTransformer(n_items, embed_dim=64, num_interests=4)
    print(f"✅ Multi-Interest Transformer created with {sum(p.numel() for p in multi_interest_model.parameters())} parameters")
    
    # Test Hierarchical Transformer
    print("\n🚀 Testing Hierarchical Transformer...")
    hierarchical_model = HierarchicalTransformer(n_items, embed_dim=64, window_size=10)
    print(f"✅ Hierarchical Transformer created with {sum(p.numel() for p in hierarchical_model.parameters())} parameters")
    
    # Test attention mechanisms
    print("\n🧠 Testing Attention Mechanisms...")
    
    # Test basic self-attention
    self_attn = SelfAttention(64, num_heads=8)
    test_input = torch.randn(32, 20, 64)
    attn_output, attn_weights = self_attn(test_input, test_input, test_input)
    print(f"✅ Self-Attention: Input {test_input.shape} -> Output {attn_output.shape}")
    
    # Test linear attention
    linear_attn = LinearAttention(64, num_heads=8)
    linear_output = linear_attn(test_input, test_input, test_input)
    print(f"✅ Linear Attention: Input {test_input.shape} -> Output {linear_output.shape}")
    
    print("\n🎯 All transformer models successfully created and tested!")

if __name__ == "__main__":
    demonstrate_transformers()
```

## Key Takeaways

1. **Self-Attention Power**: Self-attention mechanisms effectively capture long-range dependencies in user behavior sequences

2. **Transformer Architectures**: Various transformer variants (SASRec, BERT4Rec, etc.) serve different recommendation scenarios

3. **Multi-Interest Modeling**: Transformer architectures can capture diverse user interests through attention mechanisms

4. **Scalability Solutions**: Linear attention and sparse attention address computational challenges for long sequences

5. **Hierarchical Patterns**: Multi-scale transformers capture both short-term and long-term temporal patterns

6. **Cross-Domain Transfer**: Transformers enable effective knowledge transfer across different recommendation domains

## Study Questions

### Beginner Level
1. What are the key components of a transformer architecture?
2. How does self-attention differ from traditional attention mechanisms?
3. What is the purpose of positional encoding in sequential recommendations?
4. How does masking work in BERT4Rec?

### Intermediate Level
1. Compare SASRec and BERT4Rec architectures and their training strategies
2. How do multi-head attention mechanisms improve recommendation quality?
3. What are the computational complexity challenges of standard attention?
4. How can transformers handle multi-interest user modeling?

### Advanced Level
1. Design a transformer architecture that handles both sequential and social graph information
2. Implement a sparse attention pattern optimized for recommendation workloads
3. How would you adapt transformer pre-training for recommendation systems?
4. Design a federated transformer training approach for privacy-preserving recommendations

## Next Session Preview

Tomorrow we'll explore **Graph Neural Networks for Recommendations**, covering:
- Graph Convolutional Networks for collaborative filtering
- GraphSAGE and GAT for recommendation systems
- Heterogeneous graph neural networks
- Knowledge graph embeddings for recommendations
- Graph attention mechanisms
- Scalable graph neural network architectures

We'll implement powerful graph-based models that leverage network structure for enhanced recommendations!