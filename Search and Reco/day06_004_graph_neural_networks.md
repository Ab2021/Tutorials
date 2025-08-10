# Day 6.4: Graph Neural Networks for Recommendations

## Learning Objectives
- Master Graph Convolutional Networks for collaborative filtering
- Implement GraphSAGE and GAT for recommendation systems
- Design heterogeneous graph neural networks for multi-modal data
- Build knowledge graph embeddings for enhanced recommendations
- Explore graph attention mechanisms and message passing
- Develop scalable graph neural network architectures

## 1. Graph Convolutional Networks for Collaborative Filtering

### Basic Graph Convolutional Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, ndcg_score
import warnings
warnings.filterwarnings('ignore')

class GraphConvLayer(nn.Module):
    """Basic Graph Convolutional Layer"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features (num_nodes, in_features)
            adj_matrix: adjacency matrix (num_nodes, num_nodes)
        Returns:
            output: updated node features (num_nodes, out_features)
        """
        # Linear transformation
        h = self.linear(x)
        
        # Message passing: aggregate neighbor information
        # adj_matrix @ h performs neighborhood aggregation
        output = torch.matmul(adj_matrix, h)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output

class GCNRecommender(nn.Module):
    """Graph Convolutional Network for Recommendation"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [64, 32], dropout: float = 0.1,
                 n_layers: int = 3):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Graph convolutional layers
        self.gcn_layers = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GraphConvLayer(embedding_dim, hidden_dims[0], dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_dims[i], hidden_dims[i+1], dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def create_bipartite_adj_matrix(self, user_item_interactions: torch.Tensor) -> torch.Tensor:
        """
        Create bipartite adjacency matrix from user-item interactions
        Args:
            user_item_interactions: (num_interactions, 2) - [user_id, item_id] pairs
        Returns:
            adj_matrix: normalized adjacency matrix
        """
        device = user_item_interactions.device
        total_nodes = self.n_users + self.n_items
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(total_nodes, total_nodes, device=device)
        
        # User-item edges
        user_indices = user_item_interactions[:, 0]
        item_indices = user_item_interactions[:, 1] + self.n_users  # Offset for items
        
        # Add edges (both directions for undirected graph)
        adj_matrix[user_indices, item_indices] = 1.0
        adj_matrix[item_indices, user_indices] = 1.0
        
        # Add self-loops
        adj_matrix.fill_diagonal_(1.0)
        
        # Normalize adjacency matrix (symmetric normalization)
        degree = adj_matrix.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        normalized_adj = torch.matmul(torch.matmul(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
        
        return normalized_adj
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size,)
            adj_matrix: normalized adjacency matrix
        """
        # Get all node embeddings
        user_embs = self.user_embedding.weight  # (n_users, embedding_dim)
        item_embs = self.item_embedding.weight  # (n_items, embedding_dim)
        
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([user_embs, item_embs], dim=0)  # (n_users + n_items, embedding_dim)
        
        # Apply GCN layers
        x = all_embeddings
        for gcn_layer in self.gcn_layers:
            x = F.relu(gcn_layer(x, adj_matrix))
        
        # Get updated user and item embeddings
        updated_user_embs = x[:self.n_users]  # (n_users, hidden_dim)
        updated_item_embs = x[self.n_users:]  # (n_items, hidden_dim)
        
        # Get embeddings for specific users and items
        user_emb = updated_user_embs[user_ids]  # (batch_size, hidden_dim)
        item_emb = updated_item_embs[item_ids]  # (batch_size, hidden_dim)
        
        # Compute interaction scores
        interaction = user_emb * item_emb  # Element-wise multiplication
        scores = self.output_layer(interaction)
        
        return scores.squeeze()

class LightGCN(nn.Module):
    """LightGCN: Simplifying and Powering Graph Convolution Network"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # User and item embeddings (only initial embeddings)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        LightGCN forward pass - simplified GCN without feature transformation
        """
        # Initial embeddings
        user_embs = self.user_embedding.weight
        item_embs = self.item_embedding.weight
        all_embeddings = torch.cat([user_embs, item_embs], dim=0)
        
        # Store embeddings for each layer
        embeddings_layers = [all_embeddings]
        
        # Message passing for n_layers
        current_embeddings = all_embeddings
        for layer in range(self.n_layers):
            # Simple aggregation: just matrix multiplication with adjacency matrix
            current_embeddings = torch.matmul(adj_matrix, current_embeddings)
            current_embeddings = self.dropout(current_embeddings)
            embeddings_layers.append(current_embeddings)
        
        # Average all layer embeddings (including initial)
        final_embeddings = torch.stack(embeddings_layers, dim=0).mean(dim=0)
        
        # Split back to users and items
        user_final_embs = final_embeddings[:self.n_users]
        item_final_embs = final_embeddings[self.n_users:]
        
        # Get specific user and item embeddings
        user_emb = user_final_embs[user_ids]
        item_emb = item_final_embs[item_ids]
        
        # Compute scores using dot product
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores

class NGCF(nn.Module):
    """Neural Graph Collaborative Filtering"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [64, 64, 64], n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # NGCF layers
        self.ngcf_layers = nn.ModuleList()
        
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.ngcf_layers.append(NGCFLayer(input_dim, hidden_dim, dropout))
            input_dim = hidden_dim
        
        # Prediction layer
        total_embedding_dim = embedding_dim + sum(hidden_dims)
        self.prediction_layer = nn.Linear(total_embedding_dim * 2, 1)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        
        # Initial embeddings
        user_embs = self.user_embedding.weight
        item_embs = self.item_embedding.weight
        all_embeddings = torch.cat([user_embs, item_embs], dim=0)
        
        # Store all layer embeddings
        embeddings_list = [all_embeddings]
        
        # Apply NGCF layers
        current_embeddings = all_embeddings
        for ngcf_layer in self.ngcf_layers:
            current_embeddings = ngcf_layer(current_embeddings, adj_matrix)
            embeddings_list.append(current_embeddings)
        
        # Concatenate all layer embeddings
        final_embeddings = torch.cat(embeddings_list, dim=1)
        
        # Split to users and items
        user_final_embs = final_embeddings[:self.n_users]
        item_final_embs = final_embeddings[self.n_users:]
        
        # Get specific embeddings
        user_emb = user_final_embs[user_ids]
        item_emb = item_final_embs[item_ids]
        
        # Concatenate user and item embeddings for prediction
        combined = torch.cat([user_emb, item_emb], dim=1)
        scores = self.prediction_layer(combined)
        
        return scores.squeeze()

class NGCFLayer(nn.Module):
    """NGCF Layer with message passing and bi-interaction"""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Transformation matrices
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)  # Self-connection
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)  # Neighbor aggregation
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, embeddings: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (num_nodes, in_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        # Self-connection
        self_embeddings = self.W1(embeddings)
        
        # Neighbor aggregation
        neighbor_embeddings = torch.matmul(adj_matrix, embeddings)
        neighbor_embeddings = self.W2(neighbor_embeddings)
        
        # Bi-interaction (element-wise product of self and neighbor embeddings)
        bi_interaction = embeddings * neighbor_embeddings
        
        # Combine all components
        output = self_embeddings + neighbor_embeddings + bi_interaction
        
        # Apply activation and dropout
        output = self.activation(output)
        output = self.dropout(output)
        
        return output
```

## 2. GraphSAGE for Inductive Recommendations

### GraphSAGE Implementation

```python
class GraphSAGELayer(nn.Module):
    """GraphSAGE Layer with different aggregation functions"""
    
    def __init__(self, in_dim: int, out_dim: int, aggregator: str = 'mean',
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        
        # Linear transformations
        if aggregator in ['lstm', 'pool']:
            self.neighbor_linear = nn.Linear(in_dim, out_dim)
        
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.output_linear = nn.Linear(in_dim + out_dim, out_dim)
        
        # Aggregator-specific layers
        if aggregator == 'lstm':
            self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True)
        elif aggregator == 'pool':
            self.pool_linear = nn.Linear(in_dim, out_dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: torch.Tensor, adjacency_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, in_dim)
            adjacency_list: list of neighbor indices for each node
        """
        batch_size = node_features.size(0)
        
        # Aggregate neighbor features
        aggregated_neighbors = []
        
        for i in range(batch_size):
            neighbors = adjacency_list[i]
            
            if len(neighbors) == 0:
                # No neighbors - use zero vector
                neighbor_agg = torch.zeros(self.out_dim, device=node_features.device)
            else:
                neighbor_features = node_features[neighbors]  # (num_neighbors, in_dim)
                
                if self.aggregator == 'mean':
                    neighbor_agg = neighbor_features.mean(dim=0)
                    
                elif self.aggregator == 'max':
                    neighbor_agg, _ = neighbor_features.max(dim=0)
                    
                elif self.aggregator == 'lstm':
                    # LSTM aggregation
                    neighbor_features = neighbor_features.unsqueeze(0)  # Add batch dim
                    lstm_out, _ = self.lstm(neighbor_features)
                    neighbor_agg = lstm_out.squeeze(0)[-1]  # Last output
                    
                elif self.aggregator == 'pool':
                    # Max pooling after linear transformation
                    neighbor_features = F.relu(self.pool_linear(neighbor_features))
                    neighbor_agg, _ = neighbor_features.max(dim=0)
                    
                else:
                    raise ValueError(f"Unknown aggregator: {self.aggregator}")
            
            aggregated_neighbors.append(neighbor_agg)
        
        aggregated_neighbors = torch.stack(aggregated_neighbors)  # (batch_size, out_dim)
        
        # Self features
        self_features = self.self_linear(node_features)
        
        # Concatenate self and neighbor features
        combined = torch.cat([self_features, aggregated_neighbors], dim=1)
        
        # Final transformation
        output = self.output_linear(combined)
        output = F.relu(output)
        output = self.dropout(output)
        
        # L2 normalization
        output = F.normalize(output, p=2, dim=1)
        
        return output

class GraphSAGERecommender(nn.Module):
    """GraphSAGE-based Recommender System"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64], aggregator: str = 'mean',
                 dropout: float = 0.1, n_layers: int = 2):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initial embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        
        # First layer
        self.sage_layers.append(GraphSAGELayer(embedding_dim, hidden_dims[0], aggregator, dropout))
        
        # Additional layers
        for i in range(1, len(hidden_dims)):
            self.sage_layers.append(GraphSAGELayer(hidden_dims[i-1], hidden_dims[i], aggregator, dropout))
        
        # Prediction layer
        self.prediction_layer = nn.Linear(hidden_dims[-1] * 2, 1)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def sample_neighbors(self, nodes: torch.Tensor, adj_matrix: torch.Tensor,
                        num_samples: int = 10) -> List[torch.Tensor]:
        """Sample neighbors for each node"""
        sampled_neighbors = []
        
        for node in nodes:
            # Get all neighbors
            neighbors = torch.nonzero(adj_matrix[node]).squeeze()
            
            if neighbors.numel() == 0:
                sampled_neighbors.append(torch.tensor([], dtype=torch.long))
            elif neighbors.numel() == 1:
                sampled_neighbors.append(neighbors.unsqueeze(0))
            else:
                # Sample with replacement if not enough neighbors
                if len(neighbors) >= num_samples:
                    indices = torch.randperm(len(neighbors))[:num_samples]
                    sampled = neighbors[indices]
                else:
                    # Sample with replacement
                    indices = torch.randint(0, len(neighbors), (num_samples,))
                    sampled = neighbors[indices]
                
                sampled_neighbors.append(sampled)
        
        return sampled_neighbors
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        
        # Get all node embeddings
        user_embs = self.user_embedding.weight
        item_embs = self.item_embedding.weight
        all_embeddings = torch.cat([user_embs, item_embs], dim=0)
        
        # Apply GraphSAGE layers
        current_embeddings = all_embeddings
        
        for layer in self.sage_layers:
            # Sample neighbors for all nodes
            all_nodes = torch.arange(len(current_embeddings))
            neighbor_lists = self.sample_neighbors(all_nodes, adj_matrix)
            
            # Apply GraphSAGE layer
            current_embeddings = layer(current_embeddings, neighbor_lists)
        
        # Get final user and item embeddings
        final_user_embs = current_embeddings[:self.n_users]
        final_item_embs = current_embeddings[self.n_users:]
        
        # Get specific embeddings
        user_emb = final_user_embs[user_ids]
        item_emb = final_item_embs[item_ids]
        
        # Predict interactions
        combined = torch.cat([user_emb, item_emb], dim=1)
        scores = self.prediction_layer(combined)
        
        return scores.squeeze()
```

## 3. Graph Attention Networks (GAT)

### GAT Implementation for Recommendations

```python
class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Dropout and activation
        self.dropout_layer = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: node features (N, in_features)
            adj: adjacency matrix (N, N)
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)  # (N, out_features)
        N = Wh.size()[0]
        
        # Attention mechanism
        # Create all pairs for attention computation
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (N, 1)
        
        # Broadcast to create attention matrix
        e = Wh1 + Wh2.T  # (N, N)
        e = self.leakyrelu(e)
        
        # Mask attention for non-connected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax attention weights
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to node features
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiHeadGATLayer(nn.Module):
    """Multi-head Graph Attention Layer"""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 8,
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        
        self.n_heads = n_heads
        self.concat = concat
        
        # Multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat)
            for _ in range(n_heads)
        ])
        
        # Output layer for concatenated heads
        if concat:
            self.out_proj = nn.Linear(out_features * n_heads, out_features)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Apply all attention heads
        head_outputs = [attn(h, adj) for attn in self.attentions]
        
        if self.concat:
            # Concatenate all heads
            h_prime = torch.cat(head_outputs, dim=1)
            h_prime = self.out_proj(h_prime)
            return h_prime
        else:
            # Average all heads
            h_prime = torch.stack(head_outputs, dim=0).mean(dim=0)
            return h_prime

class GATRecommender(nn.Module):
    """Graph Attention Network for Recommendations"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [64, 32], n_heads: int = 8,
                 dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Initial embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(MultiHeadGATLayer(
            embedding_dim, hidden_dims[0], n_heads, dropout, alpha, concat=True
        ))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.gat_layers.append(MultiHeadGATLayer(
                hidden_dims[i-1], hidden_dims[i], 1, dropout, alpha, concat=False
            ))
        
        # Prediction layer
        self.prediction_layer = nn.Linear(hidden_dims[-1] * 2, 1)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        
        # Get all embeddings
        user_embs = self.user_embedding.weight
        item_embs = self.item_embedding.weight
        all_embeddings = torch.cat([user_embs, item_embs], dim=0)
        
        # Apply GAT layers
        h = all_embeddings
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj_matrix)
        
        # Split back to users and items
        user_final_embs = h[:self.n_users]
        item_final_embs = h[self.n_users:]
        
        # Get specific embeddings
        user_emb = user_final_embs[user_ids]
        item_emb = item_final_embs[item_ids]
        
        # Predict scores
        combined = torch.cat([user_emb, item_emb], dim=1)
        scores = self.prediction_layer(combined)
        
        return scores.squeeze()
```

## 4. Heterogeneous Graph Neural Networks

### Heterogeneous Graph Attention Network

```python
class HeteroGraphAttentionLayer(nn.Module):
    """Heterogeneous Graph Attention Layer"""
    
    def __init__(self, node_types: List[str], edge_types: List[str],
                 in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        
        # Type-specific transformations
        self.node_transforms = nn.ModuleDict({
            node_type: nn.Linear(in_dim, out_dim * n_heads)
            for node_type in node_types
        })
        
        # Edge-type specific attention
        self.edge_attentions = nn.ModuleDict({
            edge_type: nn.Linear(out_dim * 2, n_heads)
            for edge_type in edge_types
        })
        
        # Output projection
        self.output_proj = nn.Linear(out_dim * n_heads, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_indices: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: {node_type: (num_nodes, in_dim)}
            edge_indices: {edge_type: (2, num_edges)}
        """
        # Transform node features
        transformed_features = {}
        for node_type, features in node_features.items():
            transformed = self.node_transforms[node_type](features)
            # Reshape for multi-head attention
            batch_size, _ = features.shape
            transformed = transformed.view(batch_size, self.n_heads, self.out_dim)
            transformed_features[node_type] = transformed
        
        # Aggregate messages for each node type
        aggregated_features = {}
        
        for target_node_type in self.node_types:
            aggregated_messages = []
            
            for edge_type, edge_index in edge_indices.items():
                # Parse edge type (assumes format like "user-item")
                if '-' in edge_type:
                    source_type, target_type = edge_type.split('-')
                else:
                    source_type = target_type = edge_type
                
                if target_type != target_node_type:
                    continue
                
                # Get source and target nodes
                source_nodes = edge_index[0]
                target_nodes = edge_index[1]
                
                if source_type in transformed_features and target_type in transformed_features:
                    source_features = transformed_features[source_type][source_nodes]
                    target_features = transformed_features[target_type][target_nodes]
                    
                    # Compute attention scores
                    combined = torch.cat([source_features, target_features], dim=-1)
                    attention_scores = self.edge_attentions[edge_type](
                        combined.view(-1, self.out_dim * 2)
                    ).view(-1, self.n_heads)
                    
                    attention_weights = F.softmax(attention_scores, dim=0)
                    
                    # Apply attention to source features
                    attended_features = source_features * attention_weights.unsqueeze(-1)
                    
                    # Aggregate by target nodes
                    num_target_nodes = node_features[target_node_type].size(0)
                    aggregated = torch.zeros(num_target_nodes, self.n_heads, self.out_dim,
                                           device=source_features.device)
                    
                    aggregated = aggregated.scatter_add(0, 
                                                      target_nodes.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_heads, self.out_dim),
                                                      attended_features)
                    
                    aggregated_messages.append(aggregated)
            
            if aggregated_messages:
                # Sum messages from different edge types
                total_aggregated = torch.stack(aggregated_messages, dim=0).sum(dim=0)
                # Flatten heads and apply output projection
                total_aggregated = total_aggregated.view(-1, self.n_heads * self.out_dim)
                output = self.output_proj(total_aggregated)
                aggregated_features[target_node_type] = F.relu(self.dropout(output))
            else:
                # No incoming edges, use original features
                original = node_features[target_node_type]
                aggregated_features[target_node_type] = self.node_transforms[target_node_type][:original.size(0)](original)
        
        return aggregated_features

class HeteroGNNRecommender(nn.Module):
    """Heterogeneous Graph Neural Network for Recommendations"""
    
    def __init__(self, node_types: List[str], edge_types: List[str],
                 node_features_dim: Dict[str, int], embedding_dim: int = 64,
                 hidden_dims: List[int] = [64, 32], n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.embedding_dim = embedding_dim
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(1000, embedding_dim)  # Assuming max 1000 nodes per type
            for node_type in node_types
        })
        
        # Feature projection layers
        self.feature_projections = nn.ModuleDict({
            node_type: nn.Linear(node_features_dim.get(node_type, embedding_dim), embedding_dim)
            for node_type in node_types
        })
        
        # Heterogeneous GAT layers
        self.hetero_layers = nn.ModuleList()
        
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.hetero_layers.append(HeteroGraphAttentionLayer(
                node_types, edge_types, input_dim, hidden_dim, n_heads, dropout
            ))
            input_dim = hidden_dim
        
        # Prediction layers
        self.prediction_layer = nn.Linear(hidden_dims[-1] * 2, 1)
        
    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_indices: Dict[str, torch.Tensor],
                user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        
        # Project features to common embedding space
        projected_features = {}
        for node_type, features in node_features.items():
            projected = self.feature_projections[node_type](features)
            projected_features[node_type] = projected
        
        # Apply heterogeneous GAT layers
        h = projected_features
        for hetero_layer in self.hetero_layers:
            h = hetero_layer(h, edge_indices)
        
        # Get user and item embeddings
        user_embs = h['user'][user_ids]
        item_embs = h['item'][item_ids]
        
        # Predict interactions
        combined = torch.cat([user_embs, item_embs], dim=1)
        scores = self.prediction_layer(combined)
        
        return scores.squeeze()
```

## 5. Knowledge Graph Embeddings

### Knowledge-Enhanced Graph Neural Network

```python
class KnowledgeGraphEmbedding(nn.Module):
    """Knowledge Graph Embedding for Enhanced Recommendations"""
    
    def __init__(self, n_entities: int, n_relations: int, embedding_dim: int = 64,
                 margin: float = 1.0):
        super().__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        # Normalize embeddings
        self.entity_embedding.weight.data = F.normalize(
            self.entity_embedding.weight.data, p=2, dim=1
        )
        self.relation_embedding.weight.data = F.normalize(
            self.relation_embedding.weight.data, p=2, dim=1
        )
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        TransE-style knowledge graph embedding
        Args:
            head, relation, tail: entity and relation indices
        """
        head_emb = self.entity_embedding(head)
        relation_emb = self.relation_embedding(relation)
        tail_emb = self.entity_embedding(tail)
        
        # TransE: h + r ≈ t
        score = head_emb + relation_emb - tail_emb
        score = torch.norm(score, p=2, dim=1)
        
        return score
    
    def loss(self, pos_head: torch.Tensor, pos_relation: torch.Tensor, pos_tail: torch.Tensor,
             neg_head: torch.Tensor, neg_relation: torch.Tensor, neg_tail: torch.Tensor) -> torch.Tensor:
        """Margin-based ranking loss"""
        
        pos_score = self.forward(pos_head, pos_relation, pos_tail)
        neg_score = self.forward(neg_head, neg_relation, neg_tail)
        
        loss = F.relu(pos_score - neg_score + self.margin).mean()
        
        return loss

class KGNNRecommender(nn.Module):
    """Knowledge Graph Neural Network for Recommendations"""
    
    def __init__(self, n_users: int, n_items: int, n_entities: int, n_relations: int,
                 embedding_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Knowledge graph embeddings
        self.kg_embedding = KnowledgeGraphEmbedding(n_entities, n_relations, embedding_dim)
        
        # User embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        
        # Attention mechanism for knowledge aggregation
        self.knowledge_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=dropout)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(embedding_dim, embedding_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Prediction layer
        self.prediction_layer = nn.Linear(embedding_dim * 2, 1)
        
        # Initialize user embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        
    def aggregate_knowledge(self, item_entities: torch.Tensor,
                          kg_relations: torch.Tensor, kg_tails: torch.Tensor) -> torch.Tensor:
        """
        Aggregate knowledge information for items
        Args:
            item_entities: (batch_size,) - entity IDs for items
            kg_relations: (batch_size, max_neighbors) - relation IDs
            kg_tails: (batch_size, max_neighbors) - tail entity IDs
        """
        batch_size = item_entities.size(0)
        
        # Get item entity embeddings
        item_embs = self.kg_embedding.entity_embedding(item_entities)  # (batch_size, embedding_dim)
        
        # Get knowledge information
        relation_embs = self.kg_embedding.relation_embedding(kg_relations)  # (batch_size, max_neighbors, embedding_dim)
        tail_embs = self.kg_embedding.entity_embedding(kg_tails)  # (batch_size, max_neighbors, embedding_dim)
        
        # Combine relation and tail information
        knowledge_info = relation_embs + tail_embs  # (batch_size, max_neighbors, embedding_dim)
        
        # Apply attention to aggregate knowledge
        item_query = item_embs.unsqueeze(0)  # (1, batch_size, embedding_dim)
        knowledge_key = knowledge_info.transpose(0, 1)  # (max_neighbors, batch_size, embedding_dim)
        knowledge_value = knowledge_key
        
        aggregated_knowledge, attention_weights = self.knowledge_attention(
            item_query, knowledge_key, knowledge_value
        )
        
        aggregated_knowledge = aggregated_knowledge.squeeze(0)  # (batch_size, embedding_dim)
        
        # Combine with original item embedding
        enhanced_item_emb = item_embs + aggregated_knowledge
        
        return enhanced_item_emb
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                item_entities: torch.Tensor, kg_relations: torch.Tensor,
                kg_tails: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        
        # Get user embeddings
        user_embs = self.user_embedding(user_ids)
        
        # Get knowledge-enhanced item embeddings
        item_embs = self.aggregate_knowledge(item_entities, kg_relations, kg_tails)
        
        # Apply GNN layers to enhance embeddings with collaborative information
        all_user_embs = self.user_embedding.weight
        all_item_embs = self.kg_embedding.entity_embedding.weight[:self.n_items]  # Assuming first n_items are items
        
        all_embeddings = torch.cat([all_user_embs, all_item_embs], dim=0)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            all_embeddings = F.relu(gnn_layer(all_embeddings, adj_matrix))
        
        # Get final embeddings
        final_user_embs = all_embeddings[:self.n_users]
        final_item_embs = all_embeddings[self.n_users:]
        
        user_emb = final_user_embs[user_ids]
        item_emb = final_item_embs[item_ids]
        
        # Combine with knowledge-enhanced embeddings
        item_emb = item_emb + item_embs
        
        # Predict interactions
        combined = torch.cat([user_emb, item_emb], dim=1)
        scores = self.prediction_layer(combined)
        
        return scores.squeeze()
```

## 6. Training and Evaluation Framework

### Graph Neural Network Training Pipeline

```python
class GraphNNTrainer:
    """Training pipeline for Graph Neural Networks"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'val_loss': [], 'metrics': []}
    
    def create_graph_data(self, interactions: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph data from interactions"""
        
        # Create user-item interaction matrix
        user_ids = interactions['user_id'].values
        item_ids = interactions['item_id'].values
        
        # Create adjacency matrix
        n_users = interactions['user_id'].nunique()
        n_items = interactions['item_id'].nunique()
        
        # User-item interactions
        interactions_tensor = torch.tensor(np.column_stack([user_ids, item_ids]), dtype=torch.long)
        
        # Create bipartite adjacency matrix
        adj_matrix = torch.zeros(n_users + n_items, n_users + n_items)
        
        # Add user-item edges
        adj_matrix[user_ids, item_ids + n_users] = 1.0
        adj_matrix[item_ids + n_users, user_ids] = 1.0
        
        # Add self-loops
        adj_matrix.fill_diagonal_(1.0)
        
        # Normalize adjacency matrix
        degree = adj_matrix.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        normalized_adj = torch.matmul(torch.matmul(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
        
        return interactions_tensor, normalized_adj
    
    def negative_sampling(self, interactions: pd.DataFrame, num_negatives: int = 4) -> pd.DataFrame:
        """Generate negative samples for training"""
        
        n_users = interactions['user_id'].nunique()
        n_items = interactions['item_id'].nunique()
        
        # Create set of positive interactions
        positive_pairs = set(zip(interactions['user_id'], interactions['item_id']))
        
        negative_samples = []
        
        for _, row in interactions.iterrows():
            user_id = row['user_id']
            
            # Generate negative items for this user
            negatives_found = 0
            while negatives_found < num_negatives:
                neg_item = np.random.randint(0, n_items)
                
                if (user_id, neg_item) not in positive_pairs:
                    negative_samples.append({
                        'user_id': user_id,
                        'item_id': neg_item,
                        'rating': 0.0  # Negative sample
                    })
                    negatives_found += 1
        
        # Combine positive and negative samples
        positive_samples = interactions.copy()
        positive_samples['rating'] = 1.0
        
        all_samples = pd.concat([
            positive_samples,
            pd.DataFrame(negative_samples)
        ], ignore_index=True)
        
        return all_samples
    
    def train_epoch(self, data_loader: torch.utils.data.DataLoader,
                   adj_matrix: torch.Tensor, optimizer: torch.optim.Optimizer,
                   criterion) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        adj_matrix = adj_matrix.to(self.device)
        
        for batch in data_loader:
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            ratings = batch['rating'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(user_ids, item_ids, adj_matrix)
            
            # Compute loss
            loss = criterion(predictions, ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader,
                adj_matrix: torch.Tensor, criterion) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        adj_matrix = adj_matrix.to(self.device)
        
        with torch.no_grad():
            for batch in data_loader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                predictions = self.model(user_ids, item_ids, adj_matrix)
                loss = criterion(predictions, ratings)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # AUC
        try:
            auc = roc_auc_score(targets, predictions)
        except:
            auc = 0.5
        
        metrics = {
            'auc': auc,
            'rmse': np.sqrt(np.mean((predictions - targets) ** 2))
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, adj_matrix: torch.Tensor,
              n_epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-4) -> Dict[str, List]:
        """Full training pipeline"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, adj_matrix, optimizer, criterion)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader, adj_matrix, criterion)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['metrics'].append(val_metrics)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
                print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        
        return self.training_history

def demonstrate_graph_models():
    """Demonstrate Graph Neural Network models"""
    
    print("🕸️ Demonstrating Graph Neural Networks for Recommendations...")
    
    # Create sample data
    np.random.seed(42)
    n_users, n_items = 1000, 500
    n_interactions = 10000
    
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(0, n_items, n_interactions),
        'rating': np.random.uniform(1, 5, n_interactions)
    })
    
    print(f"📊 Created interaction data: {len(interactions)} interactions")
    print(f"👥 Users: {n_users}, 📦 Items: {n_items}")
    
    # Test different GNN models
    models = {
        'GCN': GCNRecommender(n_users, n_items, embedding_dim=64),
        'LightGCN': LightGCN(n_users, n_items, embedding_dim=64, n_layers=3),
        'NGCF': NGCF(n_users, n_items, embedding_dim=64, hidden_dims=[64, 32, 16]),
        'GraphSAGE': GraphSAGERecommender(n_users, n_items, embedding_dim=64),
        'GAT': GATRecommender(n_users, n_items, embedding_dim=64, n_heads=4)
    }
    
    for model_name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ {model_name} created with {param_count} parameters")
    
    # Test graph data creation
    trainer = GraphNNTrainer(models['LightGCN'])
    interactions_tensor, adj_matrix = trainer.create_graph_data(interactions)
    
    print(f"📈 Graph created: {adj_matrix.shape} adjacency matrix")
    print(f"🔗 Interactions tensor: {interactions_tensor.shape}")
    
    print("\n🎯 All Graph Neural Network models successfully created!")

if __name__ == "__main__":
    demonstrate_graph_models()
```

## Key Takeaways

1. **Graph Structure**: GNNs leverage user-item interaction graphs to capture collaborative patterns more effectively

2. **Message Passing**: Graph neural networks use message passing to aggregate neighborhood information

3. **Scalability**: Models like LightGCN provide simplified architectures for better scalability

4. **Attention Mechanisms**: GAT models use attention to weight neighbor contributions differently

5. **Heterogeneous Graphs**: Heterogeneous GNNs handle multiple node and edge types for richer modeling

6. **Knowledge Integration**: Knowledge graph embeddings enhance recommendations with external knowledge

## Study Questions

### Beginner Level
1. What are the advantages of graph neural networks over traditional collaborative filtering?
2. How does message passing work in graph convolutional networks?
3. What is the difference between GCN and GraphSAGE?
4. How do attention mechanisms work in Graph Attention Networks?

### Intermediate Level
1. Compare LightGCN and NGCF architectures and their computational complexity
2. How would you handle the cold start problem using graph neural networks?
3. What are the challenges in scaling graph neural networks to large datasets?
4. How can knowledge graphs be integrated into graph-based recommendations?

### Advanced Level
1. Design a temporal graph neural network for dynamic recommendation systems
2. Implement a federated graph neural network for privacy-preserving recommendations
3. How would you adapt graph neural networks for multi-modal recommendation data?
4. Design a graph neural network that handles both explicit and implicit feedback

## Next Session Preview

Tomorrow we'll explore **Multi-task Learning for Search and Recommendation**, covering:
- Multi-task neural architectures for recommendations
- Shared and task-specific layers design
- Multi-objective optimization strategies
- Cross-domain knowledge transfer
- Meta-learning for recommendation systems
- Advanced training techniques for multi-task models

We'll implement sophisticated multi-task learning systems that optimize multiple objectives simultaneously!