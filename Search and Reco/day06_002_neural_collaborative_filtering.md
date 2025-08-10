# Day 6.2: Neural Collaborative Filtering and Autoencoders

## Learning Objectives
- Master advanced neural collaborative filtering architectures
- Implement denoising and variational autoencoders for recommendations  
- Design convolutional neural networks for sequential recommendations
- Build recurrent neural networks for session-based recommendations
- Explore advanced training techniques and optimization strategies

## 1. Advanced Neural Collaborative Filtering

### Multi-layer Neural Collaborative Filtering

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score, precision_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedNCF(nn.Module):
    """Advanced Neural Collaborative Filtering with multiple interaction layers"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 mlp_dims: List[int] = [128, 64, 32], n_layers: int = 3,
                 dropout_rate: float = 0.2, use_batch_norm: bool = True):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Multiple embedding layers for different interaction types
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(n_users, embedding_dim) for _ in range(n_layers)
        ])
        self.item_embeddings = nn.ModuleList([
            nn.Embedding(n_items, embedding_dim) for _ in range(n_layers)
        ])
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2 * n_layers  # Concatenated embeddings from all layers
        
        for mlp_dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, mlp_dim))
            if use_batch_norm:
                mlp_layers.append(nn.BatchNorm1d(mlp_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = mlp_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with different strategies for each layer"""
        for i, (user_emb, item_emb) in enumerate(zip(self.user_embeddings, self.item_embeddings)):
            # Different initialization for different layers
            if i == 0:
                nn.init.normal_(user_emb.weight, std=0.01)
                nn.init.normal_(item_emb.weight, std=0.01)
            else:
                nn.init.xavier_uniform_(user_emb.weight)
                nn.init.xavier_uniform_(item_emb.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        embeddings = []
        
        # Get embeddings from all layers
        for user_emb, item_emb in zip(self.user_embeddings, self.item_embeddings):
            user_vec = user_emb(user_ids)
            item_vec = item_emb(item_ids)
            embeddings.extend([user_vec, item_vec])
        
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=-1)
        
        # Pass through MLP
        x = self.mlp(x)
        output = self.output_layer(x)
        
        return output.squeeze()

class DeepFM(nn.Module):
    """DeepFM model combining factorization machines with deep learning"""
    
    def __init__(self, feature_dims: List[int], embedding_dim: int = 64,
                 mlp_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        super().__init__()
        
        self.feature_dims = feature_dims  # Dimensions for each categorical feature
        self.embedding_dim = embedding_dim
        self.total_features = len(feature_dims)
        
        # First-order weights (linear part)
        self.first_order_weights = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in feature_dims
        ])
        
        # Embeddings for factorization machine
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in feature_dims
        ])
        
        # Deep component
        mlp_layers = []
        input_dim = self.total_features * embedding_dim
        
        for mlp_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = mlp_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*mlp_layers)
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for first_order in self.first_order_weights:
            nn.init.zeros_(first_order.weight)
        
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: tensor of shape (batch_size, n_features) containing feature indices
        """
        batch_size = features.size(0)
        
        # First-order component (linear)
        first_order = 0
        for i, (feature_idx, weight_emb) in enumerate(zip(features.T, self.first_order_weights)):
            first_order += weight_emb(feature_idx).squeeze()
        
        # Second-order component (factorization machine)
        embeddings = []
        for i, (feature_idx, emb) in enumerate(zip(features.T, self.embeddings)):
            embeddings.append(emb(feature_idx))
        
        embeddings = torch.stack(embeddings, dim=1)  # (batch_size, n_features, embedding_dim)
        
        # FM interaction: 0.5 * ((sum(x))^2 - sum(x^2))
        sum_squared = torch.sum(embeddings, dim=1) ** 2  # (batch_size, embedding_dim)
        squared_sum = torch.sum(embeddings ** 2, dim=1)  # (batch_size, embedding_dim)
        fm_output = 0.5 * torch.sum(sum_squared - squared_sum, dim=1)  # (batch_size,)
        
        # Deep component
        deep_input = embeddings.view(batch_size, -1)  # Flatten
        deep_output = self.deep_layers(deep_input).squeeze()
        
        # Combine all components
        output = self.bias + first_order + fm_output + deep_output
        
        return output

class NeuralBPR(nn.Module):
    """Neural Bayesian Personalized Ranking for implicit feedback"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Neural network for interaction modeling
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # Neural interaction
        output = self.network(x)
        
        return output.squeeze()
    
    def bpr_loss(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, 
                 neg_item_ids: torch.Tensor) -> torch.Tensor:
        """Compute BPR loss"""
        
        # Positive scores
        pos_scores = self.forward(user_ids, pos_item_ids)
        
        # Negative scores  
        neg_scores = self.forward(user_ids, neg_item_ids)
        
        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        return loss

class MultitaskNCF(nn.Module):
    """Multi-task Neural Collaborative Filtering for rating and ranking"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 shared_dims: List[int] = [128, 64], rating_dims: List[int] = [32],
                 ranking_dims: List[int] = [32], dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        # Shared embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Shared layers
        shared_layers = []
        input_dim = embedding_dim * 2
        
        for shared_dim in shared_dims:
            shared_layers.extend([
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = shared_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Task-specific networks
        # Rating prediction head
        rating_layers = []
        rating_input_dim = input_dim
        
        for rating_dim in rating_dims:
            rating_layers.extend([
                nn.Linear(rating_input_dim, rating_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            rating_input_dim = rating_dim
        
        rating_layers.append(nn.Linear(rating_input_dim, 1))
        self.rating_head = nn.Sequential(*rating_layers)
        
        # Ranking head
        ranking_layers = []
        ranking_input_dim = input_dim
        
        for ranking_dim in ranking_dims:
            ranking_layers.extend([
                nn.Linear(ranking_input_dim, ranking_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            ranking_input_dim = ranking_dim
        
        ranking_layers.append(nn.Linear(ranking_input_dim, 1))
        self.ranking_head = nn.Sequential(*ranking_layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                task: str = 'both') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Get shared representation
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        shared_features = self.shared_network(x)
        
        if task == 'rating':
            return self.rating_head(shared_features).squeeze()
        elif task == 'ranking':
            return self.ranking_head(shared_features).squeeze()
        else:  # both
            rating_output = self.rating_head(shared_features).squeeze()
            ranking_output = self.ranking_head(shared_features).squeeze()
            return rating_output, ranking_output
    
    def multitask_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                      ratings: torch.Tensor, ranking_labels: torch.Tensor,
                      rating_weight: float = 0.5) -> torch.Tensor:
        """Compute multi-task loss"""
        
        rating_pred, ranking_pred = self.forward(user_ids, item_ids, task='both')
        
        # Rating prediction loss (MSE)
        rating_loss = F.mse_loss(rating_pred, ratings)
        
        # Ranking loss (BCE)
        ranking_loss = F.binary_cross_entropy_with_logits(ranking_pred, ranking_labels)
        
        # Combined loss
        total_loss = rating_weight * rating_loss + (1 - rating_weight) * ranking_loss
        
        return total_loss, rating_loss, ranking_loss
```

## 2. Autoencoders for Collaborative Filtering

### Denoising Autoencoders

```python
class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder for Collaborative Filtering"""
    
    def __init__(self, n_items: int, hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.5, noise_factor: float = 0.5):
        super().__init__()
        
        self.n_items = n_items
        self.noise_factor = noise_factor
        
        # Encoder
        encoder_layers = []
        input_dim = n_items
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.SELU(),  # SELU activation for autoencoders
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (symmetric to encoder)
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                output_dim = n_items
            else:
                output_dim = hidden_dims[i-1]
            
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], output_dim),
                nn.SELU() if i > 0 else nn.Identity()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for SELU activation"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=np.sqrt(1.0 / module.in_features))
                nn.init.zeros_(module.bias)
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise to input"""
        if self.training:
            # Dropout-based noise
            noise = torch.bernoulli(torch.full_like(x, 1 - self.noise_factor))
            return x * noise
        return x
    
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        if add_noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        
        # Encode
        latent = self.encoder(x_noisy)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation without noise"""
        return self.encoder(x)

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for Collaborative Filtering"""
    
    def __init__(self, n_items: int, latent_dim: int = 128,
                 encoder_dims: List[int] = [512, 256], 
                 decoder_dims: List[int] = [256, 512],
                 dropout_rate: float = 0.5, beta: float = 1.0):
        super().__init__()
        
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.beta = beta  # Beta-VAE parameter
        
        # Encoder
        encoder_layers = []
        input_dim = n_items
        
        for hidden_dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, n_items))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters"""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            return mu + epsilon * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def vae_loss(self, x: torch.Tensor, reconstructed: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss (reconstruction + KL divergence)"""
        
        # Reconstruction loss (multinomial likelihood)
        reconstruction_loss = F.binary_cross_entropy_with_logits(
            reconstructed, x, reduction='sum'
        )
        
        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_divergence
        
        return total_loss, reconstruction_loss, kl_divergence

class ConditionalVAE(nn.Module):
    """Conditional VAE for multi-criteria recommendations"""
    
    def __init__(self, n_items: int, n_conditions: int, latent_dim: int = 128,
                 encoder_dims: List[int] = [512, 256], 
                 decoder_dims: List[int] = [256, 512],
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.n_items = n_items
        self.n_conditions = n_conditions
        self.latent_dim = latent_dim
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(n_conditions, 64)
        
        # Encoder (takes both items and conditions)
        encoder_layers = []
        input_dim = n_items + 64  # Items + condition embedding
        
        for hidden_dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder (takes latent + condition)
        decoder_layers = []
        input_dim = latent_dim + 64
        
        for hidden_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, n_items))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get condition embeddings
        cond_emb = self.condition_embedding(conditions)
        
        # Encode with condition
        encoder_input = torch.cat([x, cond_emb], dim=-1)
        hidden = self.encoder(encoder_input)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        
        # Decode with condition
        decoder_input = torch.cat([z, cond_emb], dim=-1)
        reconstructed = self.decoder(decoder_input)
        
        return reconstructed, mu, logvar
```

## 3. Convolutional Neural Networks for Sequential Recommendations

### CNN-based Sequential Recommendation

```python
class ConvolutionalSequentialRecommender(nn.Module):
    """CNN-based model for sequential recommendations"""
    
    def __init__(self, n_items: int, embedding_dim: int = 64, 
                 sequence_length: int = 20, n_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5], dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.n_filters = n_filters
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, n_filters, kernel_size=filter_size, padding=filter_size//2)
            for filter_size in filter_sizes
        ])
        
        # Max pooling
        self.max_pools = nn.ModuleList([
            nn.AdaptiveMaxPool1d(1) for _ in filter_sizes
        ])
        
        # Fully connected layers
        fc_input_dim = n_filters * len(filter_sizes)
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_input_dim // 2, n_items)
        )
        
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, sequence_length) - item sequences
        """
        # Get embeddings
        embedded = self.item_embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Transpose for Conv1d (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply multiple convolutional filters
        conv_outputs = []
        for conv, max_pool in zip(self.conv_layers, self.max_pools):
            conv_out = F.relu(conv(embedded))  # (batch_size, n_filters, seq_len)
            pooled = max_pool(conv_out).squeeze(-1)  # (batch_size, n_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=-1)  # (batch_size, n_filters * n_filter_sizes)
        
        # Final prediction
        output = self.fc_layers(concatenated)
        
        return output

class HierarchicalCNN(nn.Module):
    """Hierarchical CNN for capturing different temporal patterns"""
    
    def __init__(self, n_items: int, embedding_dim: int = 64,
                 sequence_length: int = 50, local_window: int = 5,
                 n_filters: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.local_window = local_window
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # Local CNN (short-term patterns)
        self.local_conv = nn.Conv1d(embedding_dim, n_filters, kernel_size=local_window, 
                                   padding=local_window//2)
        
        # Global CNN (long-term patterns)
        self.global_conv = nn.Conv1d(embedding_dim, n_filters, kernel_size=sequence_length//4,
                                    padding=sequence_length//8)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(n_filters * 2, num_heads=8, dropout=dropout_rate)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(n_filters * 2, n_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_filters, n_items)
        )
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        embedded = self.item_embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        embedded_transposed = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Local and global convolutions
        local_features = F.relu(self.local_conv(embedded_transposed))  # (batch_size, n_filters, seq_len)
        global_features = F.relu(self.global_conv(embedded_transposed))  # (batch_size, n_filters, seq_len)
        
        # Concatenate features
        combined_features = torch.cat([local_features, global_features], dim=1)  # (batch_size, 2*n_filters, seq_len)
        
        # Transpose for attention (seq_len, batch_size, features)
        combined_features = combined_features.transpose(0, 2).transpose(1, 2)
        
        # Self-attention
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        
        # Global average pooling
        pooled_features = attended_features.mean(dim=0)  # (batch_size, 2*n_filters)
        
        # Final prediction
        output = self.output_layers(pooled_features)
        
        return output
```

## 4. Recurrent Neural Networks for Session-based Recommendations

### GRU-based Session Recommendation

```python
class GRUSessionRecommender(nn.Module):
    """GRU-based model for session-based recommendations"""
    
    def __init__(self, n_items: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 n_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # GRU layers
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout_rate if n_layers > 1 else 0)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, n_items)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, max_seq_len) - padded sequences
            lengths: (batch_size,) - actual sequence lengths
        """
        batch_size, max_seq_len = sequences.size()
        
        # Get embeddings
        embedded = self.item_embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                        batch_first=True, enforce_sorted=False)
        
        # GRU forward pass
        gru_output, hidden = self.gru(embedded)
        
        # Unpack if necessary
        if lengths is not None:
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        
        # Use the last output for each sequence
        if lengths is not None:
            # Get the last valid output for each sequence
            batch_indices = torch.arange(batch_size, device=sequences.device)
            last_outputs = gru_output[batch_indices, lengths - 1]
        else:
            last_outputs = gru_output[:, -1, :]  # Use last timestep
        
        # Apply dropout and get predictions
        last_outputs = self.dropout(last_outputs)
        output = self.output_layer(last_outputs)
        
        return output

class AttentionGRU(nn.Module):
    """GRU with attention mechanism for session recommendations"""
    
    def __init__(self, n_items: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 n_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers,
                         batch_first=True, dropout=dropout_rate if n_layers > 1 else 0)
        
        # Attention mechanism
        self.attention_layer = nn.Linear(hidden_dim, 1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, n_items)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def attention(self, gru_outputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            gru_outputs: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - mask for padded positions
        """
        # Compute attention scores
        attention_scores = self.attention_layer(gru_outputs).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum
        attended_output = torch.sum(gru_outputs * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        
        return attended_output, attention_weights
    
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch_size, max_seq_len = sequences.size()
        
        # Get embeddings
        embedded = self.item_embedding(sequences)
        
        # Create mask for attention
        if lengths is not None:
            mask = torch.arange(max_seq_len, device=sequences.device).expand(
                batch_size, max_seq_len
            ) < lengths.unsqueeze(1)
        else:
            mask = None
        
        # GRU forward pass
        gru_outputs, _ = self.gru(embedded)
        
        # Apply attention
        attended_output, attention_weights = self.attention(gru_outputs, mask)
        
        # Final prediction
        attended_output = self.dropout(attended_output)
        output = self.output_layer(attended_output)
        
        return output

class NARM(nn.Module):
    """Neural Attentive Recommendation Machine"""
    
    def __init__(self, n_items: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # GRU for encoding sequences
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Attention networks
        self.global_attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.local_attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Transformation matrices
        self.A1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.A2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output layer
        self.output_layer = nn.Linear(2 * hidden_dim, n_items)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch_size, max_seq_len = sequences.size()
        
        # Get embeddings
        embedded = self.item_embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Encode with GRU
        gru_outputs, final_hidden = self.encoder_gru(embedded)  # gru_outputs: (batch_size, seq_len, hidden_dim)
        
        # Global representation (last hidden state)
        global_repr = final_hidden.squeeze(0)  # (batch_size, hidden_dim)
        
        # Local attention
        attention_scores = torch.bmm(
            self.local_attention(gru_outputs),  # (batch_size, seq_len, hidden_dim)
            global_repr.unsqueeze(-1)  # (batch_size, hidden_dim, 1)
        ).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if lengths provided
        if lengths is not None:
            mask = torch.arange(max_seq_len, device=sequences.device).expand(
                batch_size, max_seq_len
            ) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        
        # Local representation (attention-weighted sum)
        local_repr = torch.sum(gru_outputs * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        
        # Transform representations
        global_transformed = self.A1(global_repr)
        local_transformed = self.A2(local_repr)
        
        # Combine representations
        combined_repr = torch.cat([global_transformed, local_transformed], dim=-1)  # (batch_size, 2*hidden_dim)
        
        # Final prediction
        combined_repr = self.dropout(combined_repr)
        output = self.output_layer(combined_repr)
        
        return output
```

## 5. Training and Evaluation Framework

### Advanced Training Pipeline

```python
class AdvancedTrainer:
    """Advanced training pipeline for neural collaborative filtering models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []
        }
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion, epoch: int, model_type: str = 'standard') -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Handle different model types
            if model_type == 'autoencoder':
                x = batch['ratings'].to(self.device)
                if hasattr(self.model, 'forward'):
                    if isinstance(self.model, VariationalAutoencoder):
                        reconstructed, mu, logvar = self.model(x)
                        loss, recon_loss, kl_loss = self.model.vae_loss(x, reconstructed, mu, logvar)
                    else:
                        reconstructed = self.model(x)
                        loss = criterion(reconstructed, x)
                else:
                    reconstructed = self.model(x)
                    loss = criterion(reconstructed, x)
            
            elif model_type == 'bpr':
                user_ids = batch['user_id'].to(self.device)
                pos_items = batch['pos_item'].to(self.device)
                neg_items = batch['neg_item'].to(self.device)
                loss = self.model.bpr_loss(user_ids, pos_items, neg_items)
            
            elif model_type == 'multitask':
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                ranking_labels = batch['ranking_label'].to(self.device)
                loss, rating_loss, ranking_loss = self.model.multitask_loss(
                    user_ids, item_ids, ratings, ranking_labels
                )
            
            elif model_type == 'sequential':
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                lengths = batch.get('length', None)
                if lengths is not None:
                    lengths = lengths.to(self.device)
                
                if lengths is not None:
                    predictions = self.model(sequences, lengths)
                else:
                    predictions = self.model(sequences)
                
                loss = criterion(predictions, targets)
            
            else:  # standard
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
            
            # Add L2 regularization if model supports it
            if hasattr(self.model, 'compute_l2_loss'):
                loss += self.model.compute_l2_loss()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader, criterion, 
                model_type: str = 'standard') -> Tuple[float, Dict[str, float]]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                if model_type == 'autoencoder':
                    x = batch['ratings'].to(self.device)
                    if isinstance(self.model, VariationalAutoencoder):
                        reconstructed, mu, logvar = self.model(x)
                        loss, _, _ = self.model.vae_loss(x, reconstructed, mu, logvar)
                    else:
                        reconstructed = self.model(x)
                        loss = criterion(reconstructed, x)
                    
                    predictions = reconstructed.cpu().numpy()
                    targets = x.cpu().numpy()
                
                elif model_type == 'sequential':
                    sequences = batch['sequence'].to(self.device)
                    targets = batch['target'].to(self.device)
                    lengths = batch.get('length', None)
                    if lengths is not None:
                        lengths = lengths.to(self.device)
                        predictions = self.model(sequences, lengths)
                    else:
                        predictions = self.model(sequences)
                    
                    loss = criterion(predictions, targets)
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                
                else:  # standard
                    user_ids = batch['user_id'].to(self.device)
                    item_ids = batch['item_id'].to(self.device)
                    ratings = batch['rating'].to(self.device)
                    
                    predictions = self.model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(ratings.cpu().numpy())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute additional metrics
        metrics = {}
        if all_predictions and all_targets:
            predictions = np.concatenate(all_predictions)
            targets = np.concatenate(all_targets)
            
            # RMSE
            metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
            
            # MAE
            metrics['mae'] = np.mean(np.abs(predictions - targets))
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              n_epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, patience: int = 10,
              model_type: str = 'standard', loss_fn: str = 'mse') -> Dict[str, List]:
        """Full training pipeline"""
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup loss function
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_fn == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience//2, factor=0.5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch, model_type)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader, criterion, model_type)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_metrics'].append(val_metrics)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}")
                    for metric_name, metric_value in val_metrics.items():
                        print(f"  Val {metric_name.upper()}: {metric_value:.4f}")
        
        # Load best model
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.training_history

def demonstrate_advanced_ncf():
    """Demonstrate advanced NCF models"""
    print("🚀 Demonstrating Advanced Neural Collaborative Filtering...")
    
    # Create sample data
    n_users, n_items = 1000, 500
    n_interactions = 10000
    
    # Generate interactions
    np.random.seed(42)
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(0, n_items, n_interactions),
        'rating': np.random.uniform(1, 5, n_interactions)
    })
    
    # Test Advanced NCF
    model = AdvancedNCF(n_users, n_items, embedding_dim=64, n_layers=3)
    print(f"✅ Advanced NCF model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test Denoising Autoencoder
    dae_model = DenoisingAutoencoder(n_items, hidden_dims=[256, 128, 64])
    print(f"✅ Denoising Autoencoder created with {sum(p.numel() for p in dae_model.parameters())} parameters")
    
    # Test Variational Autoencoder
    vae_model = VariationalAutoencoder(n_items, latent_dim=64, encoder_dims=[256, 128])
    print(f"✅ Variational Autoencoder created with {sum(p.numel() for p in vae_model.parameters())} parameters")
    
    # Test Sequential models
    seq_model = GRUSessionRecommender(n_items, embedding_dim=64, hidden_dim=128)
    print(f"✅ GRU Session Recommender created with {sum(p.numel() for p in seq_model.parameters())} parameters")
    
    print("\n🎯 All advanced models successfully created and ready for training!")

if __name__ == "__main__":
    demonstrate_advanced_ncf()
```

## Key Takeaways

1. **Advanced NCF**: Multi-layer architectures and specialized interaction modeling improve recommendation quality

2. **Autoencoders**: VAEs and denoising autoencoders effectively handle sparse and noisy interaction data

3. **Sequential Modeling**: CNNs and RNNs capture temporal patterns in user behavior

4. **Attention Mechanisms**: Attention-based models focus on relevant parts of sequences for better predictions

5. **Multi-task Learning**: Joint optimization of multiple objectives improves overall performance

6. **Training Strategies**: Advanced training techniques including regularization and optimization are crucial

## Study Questions

### Beginner Level
1. What are the advantages of neural collaborative filtering over traditional methods?
2. How do autoencoders handle the sparsity problem in recommendation systems?
3. What is the difference between denoising and variational autoencoders?
4. Why are sequential models important for recommendations?

### Intermediate Level
1. Compare CNN vs RNN approaches for sequential recommendation
2. How does the attention mechanism improve sequential recommendations?
3. What are the challenges in training variational autoencoders for RecSys?
4. How would you handle the cold start problem in neural collaborative filtering?

### Advanced Level
1. Design a multi-modal neural architecture that incorporates text, images, and interactions
2. Implement a hierarchical attention mechanism for long sequences
3. How would you adapt these models for real-time recommendation systems?
4. Design a federated learning approach for neural collaborative filtering

## Next Session Preview

Tomorrow we'll explore **Attention Mechanisms and Transformer Models**, covering:
- Self-attention mechanisms for recommendation systems
- Transformer architectures for sequential recommendations
- BERT-style pre-training for recommendation systems
- Multi-head attention and positional encodings
- Transformer variants optimized for RecSys
- Scalable attention mechanisms for large-scale systems

We'll implement cutting-edge attention-based models that represent the state-of-the-art in neural recommendations!