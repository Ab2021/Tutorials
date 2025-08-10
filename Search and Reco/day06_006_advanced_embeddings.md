# Day 6.6: Advanced Embedding Techniques

## Learning Objectives
- Master dynamic and contextual embeddings for recommendations
- Implement graph-based embedding methods and multi-modal fusion
- Design embedding compression and quantization techniques
- Build privacy-preserving embedding systems
- Develop real-time embedding updates and serving infrastructure

## 1. Dynamic and Contextual Embeddings

### Time-Aware Dynamic Embeddings

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TemporalEmbedding(nn.Module):
    """Time-aware embeddings that evolve over time"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 time_units: int = 24, decay_rate: float = 0.1,
                 update_method: str = 'exponential'):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.time_units = time_units
        self.decay_rate = decay_rate
        self.update_method = update_method
        
        # Base embeddings
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Temporal evolution networks
        if update_method == 'rnn':
            self.temporal_rnn = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
            self.temporal_states = {}  # Store hidden states for each item
        elif update_method == 'attention':
            self.temporal_attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
            self.temporal_memory = {}  # Store temporal memories
        elif update_method == 'neural_ode':
            self.ode_network = self._build_ode_network()
        
        # Time encoding
        self.time_encoding = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.base_embeddings.weight)
        
    def _build_ode_network(self) -> nn.Module:
        """Build Neural ODE network for continuous temporal evolution"""
        return nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.Tanh(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )
    
    def exponential_decay_update(self, base_emb: torch.Tensor, 
                                time_delta: torch.Tensor) -> torch.Tensor:
        """Exponential decay temporal update"""
        decay_factor = torch.exp(-self.decay_rate * time_delta.unsqueeze(-1))
        return base_emb * decay_factor
    
    def rnn_temporal_update(self, item_ids: torch.Tensor, 
                           time_stamps: torch.Tensor) -> torch.Tensor:
        """RNN-based temporal update"""
        batch_size = item_ids.size(0)
        updated_embeddings = []
        
        for i in range(batch_size):
            item_id = item_ids[i].item()
            current_time = time_stamps[i].item()
            
            # Get base embedding
            base_emb = self.base_embeddings(item_ids[i:i+1])
            
            # Initialize hidden state if not exists
            if item_id not in self.temporal_states:
                self.temporal_states[item_id] = torch.zeros(1, 1, self.embedding_dim)
            
            # RNN update
            output, hidden = self.temporal_rnn(base_emb.unsqueeze(0), 
                                             self.temporal_states[item_id])
            
            # Update stored state
            self.temporal_states[item_id] = hidden.detach()
            
            updated_embeddings.append(output.squeeze(0))
        
        return torch.cat(updated_embeddings, dim=0)
    
    def attention_temporal_update(self, item_ids: torch.Tensor,
                                 time_stamps: torch.Tensor) -> torch.Tensor:
        """Attention-based temporal update"""
        batch_size = item_ids.size(0)
        base_embeddings = self.base_embeddings(item_ids)
        
        # Encode time information
        time_features = self.time_encoding(time_stamps.unsqueeze(-1))
        
        # Combine base embeddings with time features
        temporal_queries = base_embeddings + time_features
        
        updated_embeddings = []
        
        for i in range(batch_size):
            item_id = item_ids[i].item()
            query = temporal_queries[i:i+1]
            
            # Get temporal memory for this item
            if item_id not in self.temporal_memory:
                self.temporal_memory[item_id] = query.clone()
            
            memory = self.temporal_memory[item_id]
            
            # Apply temporal attention
            attended_emb, _ = self.temporal_attention(
                query.unsqueeze(0), memory.unsqueeze(0), memory.unsqueeze(0)
            )
            
            # Update memory
            self.temporal_memory[item_id] = torch.cat([memory, query], dim=0)[-10:]  # Keep last 10
            
            updated_embeddings.append(attended_emb.squeeze(0))
        
        return torch.cat(updated_embeddings, dim=0)
    
    def neural_ode_update(self, base_emb: torch.Tensor, 
                         time_delta: torch.Tensor) -> torch.Tensor:
        """Neural ODE temporal update"""
        # Simple Euler method integration
        dt = 0.1
        steps = int(time_delta.max().item() / dt) + 1
        
        current_emb = base_emb
        for step in range(steps):
            derivative = self.ode_network(current_emb)
            current_emb = current_emb + dt * derivative
        
        return current_emb
    
    def forward(self, item_ids: torch.Tensor, time_stamps: torch.Tensor,
                reference_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with temporal evolution"""
        
        if reference_time is None:
            reference_time = time_stamps.max()
        
        # Compute time deltas
        time_deltas = reference_time - time_stamps
        
        if self.update_method == 'exponential':
            base_embeddings = self.base_embeddings(item_ids)
            return self.exponential_decay_update(base_embeddings, time_deltas)
        
        elif self.update_method == 'rnn':
            return self.rnn_temporal_update(item_ids, time_stamps)
        
        elif self.update_method == 'attention':
            return self.attention_temporal_update(item_ids, time_stamps)
        
        elif self.update_method == 'neural_ode':
            base_embeddings = self.base_embeddings(item_ids)
            return self.neural_ode_update(base_embeddings, time_deltas)
        
        else:
            return self.base_embeddings(item_ids)

class ContextualEmbedding(nn.Module):
    """Context-aware embeddings that adapt based on situational factors"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 context_features: List[str], context_dims: Dict[str, int],
                 fusion_method: str = 'attention', dropout_rate: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_features = context_features
        self.context_dims = context_dims
        self.fusion_method = fusion_method
        
        # Base embeddings
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context encoders
        self.context_encoders = nn.ModuleDict()
        for feature, dim in context_dims.items():
            if feature in context_features:
                encoder = nn.Sequential(
                    nn.Linear(dim, embedding_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(embedding_dim // 2, embedding_dim)
                )
                self.context_encoders[feature] = encoder
        
        # Fusion mechanisms
        if fusion_method == 'attention':
            self.context_attention = nn.MultiheadAttention(
                embedding_dim, num_heads=4, dropout=dropout_rate
            )
        elif fusion_method == 'gating':
            total_context_dim = len(context_features) * embedding_dim
            self.gating_network = nn.Sequential(
                nn.Linear(total_context_dim, embedding_dim),
                nn.Sigmoid()
            )
        elif fusion_method == 'mixture':
            self.mixture_weights = nn.Sequential(
                nn.Linear(sum(context_dims[f] for f in context_features), len(context_features) + 1),
                nn.Softmax(dim=-1)
            )
        
        # Contextualization network
        self.contextualization = nn.Sequential(
            nn.Linear(embedding_dim * (len(context_features) + 1), embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.base_embeddings.weight)
    
    def encode_context(self, context_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode context features"""
        encoded_contexts = {}
        
        for feature in self.context_features:
            if feature in context_data and feature in self.context_encoders:
                context_vector = context_data[feature]
                encoded = self.context_encoders[feature](context_vector)
                encoded_contexts[feature] = encoded
        
        return encoded_contexts
    
    def attention_fusion(self, base_emb: torch.Tensor, 
                        encoded_contexts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Attention-based context fusion"""
        # Prepare query, key, value tensors
        query = base_emb.unsqueeze(0)  # (1, batch_size, embedding_dim)
        
        # Stack context embeddings
        context_embeddings = list(encoded_contexts.values())
        if context_embeddings:
            keys = torch.stack(context_embeddings)  # (n_contexts, batch_size, embedding_dim)
            values = keys.clone()
            
            # Apply attention
            attended_emb, attention_weights = self.context_attention(query, keys, values)
            return attended_emb.squeeze(0)
        else:
            return base_emb
    
    def gating_fusion(self, base_emb: torch.Tensor,
                     encoded_contexts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Gating-based context fusion"""
        # Concatenate all context embeddings
        context_embeddings = list(encoded_contexts.values())
        if context_embeddings:
            all_contexts = torch.cat(context_embeddings, dim=-1)
            
            # Compute gates
            gates = self.gating_network(all_contexts)
            
            # Apply gating to base embedding
            return base_emb * gates
        else:
            return base_emb
    
    def mixture_fusion(self, base_emb: torch.Tensor, context_data: Dict[str, torch.Tensor],
                      encoded_contexts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mixture of experts fusion"""
        # Compute mixture weights based on raw context features
        context_features = [context_data[f] for f in self.context_features if f in context_data]
        if context_features:
            combined_context = torch.cat(context_features, dim=-1)
            mixture_weights = self.mixture_weights(combined_context)
            
            # Weighted combination
            all_embeddings = [base_emb] + list(encoded_contexts.values())
            weighted_sum = torch.zeros_like(base_emb)
            
            for i, emb in enumerate(all_embeddings):
                if i < mixture_weights.size(-1):
                    weighted_sum += mixture_weights[:, i:i+1] * emb
            
            return weighted_sum
        else:
            return base_emb
    
    def forward(self, item_ids: torch.Tensor, 
                context_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with contextual adaptation"""
        
        # Get base embeddings
        base_embeddings = self.base_embeddings(item_ids)
        
        # Encode context features
        encoded_contexts = self.encode_context(context_data)
        
        # Fusion based on method
        if self.fusion_method == 'attention':
            fused_embeddings = self.attention_fusion(base_embeddings, encoded_contexts)
        elif self.fusion_method == 'gating':
            fused_embeddings = self.gating_fusion(base_embeddings, encoded_contexts)
        elif self.fusion_method == 'mixture':
            fused_embeddings = self.mixture_fusion(base_embeddings, context_data, encoded_contexts)
        else:
            # Simple concatenation + projection
            all_embeddings = [base_embeddings] + list(encoded_contexts.values())
            concatenated = torch.cat(all_embeddings, dim=-1)
            fused_embeddings = self.contextualization(concatenated)
        
        return fused_embeddings

class AdaptiveEmbedding(nn.Module):
    """Adaptive embeddings that learn to specialize based on usage patterns"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_specialists: int = 4, routing_method: str = 'learned',
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_specialists = num_specialists
        self.routing_method = routing_method
        
        # Specialist embeddings
        self.specialist_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) 
            for _ in range(num_specialists)
        ])
        
        # Routing network
        if routing_method == 'learned':
            self.router = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim // 2, num_specialists),
                nn.Softmax(dim=-1)
            )
            # Base embedding for routing decision
            self.routing_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        elif routing_method == 'frequency':
            # Frequency-based routing (high freq -> specialist 0, low freq -> specialist -1)
            self.frequency_counter = torch.zeros(vocab_size)
            self.frequency_thresholds = torch.linspace(0, 1, num_specialists + 1)[1:-1]
        
        elif routing_method == 'semantic':
            # Semantic clustering for routing
            self.semantic_centroids = nn.Parameter(torch.randn(num_specialists, embedding_dim))
        
        # Initialize all embeddings
        for specialist in self.specialist_embeddings:
            nn.init.xavier_uniform_(specialist.weight)
        
        if routing_method == 'learned':
            nn.init.xavier_uniform_(self.routing_embedding.weight)
    
    def update_frequency(self, item_ids: torch.Tensor):
        """Update frequency counter for frequency-based routing"""
        if self.routing_method == 'frequency':
            unique_ids, counts = torch.unique(item_ids, return_counts=True)
            self.frequency_counter[unique_ids] += counts.float()
    
    def learned_routing(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Learned routing weights"""
        routing_emb = self.routing_embedding(item_ids)
        routing_weights = self.router(routing_emb)
        return routing_weights
    
    def frequency_routing(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Frequency-based routing"""
        batch_size = item_ids.size(0)
        routing_weights = torch.zeros(batch_size, self.num_specialists)
        
        # Normalize frequencies
        total_freq = self.frequency_counter.sum()
        normalized_freq = self.frequency_counter / (total_freq + 1e-8)
        
        for i, item_id in enumerate(item_ids):
            freq = normalized_freq[item_id]
            
            # Find appropriate specialist based on frequency
            specialist_idx = 0
            for j, threshold in enumerate(self.frequency_thresholds):
                if freq > threshold:
                    specialist_idx = j + 1
                else:
                    break
            
            routing_weights[i, specialist_idx] = 1.0
        
        return routing_weights
    
    def semantic_routing(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Semantic similarity-based routing"""
        # Use first specialist embedding as semantic representation
        semantic_emb = self.specialist_embeddings[0](item_ids)
        
        # Compute similarities to centroids
        similarities = torch.matmul(semantic_emb, self.semantic_centroids.T)
        routing_weights = F.softmax(similarities, dim=-1)
        
        return routing_weights
    
    def forward(self, item_ids: torch.Tensor, 
                training: bool = True) -> torch.Tensor:
        """Forward pass with adaptive routing"""
        
        # Update frequency if needed
        if training:
            self.update_frequency(item_ids)
        
        # Get routing weights
        if self.routing_method == 'learned':
            routing_weights = self.learned_routing(item_ids)
        elif self.routing_method == 'frequency':
            routing_weights = self.frequency_routing(item_ids)
        elif self.routing_method == 'semantic':
            routing_weights = self.semantic_routing(item_ids)
        else:
            # Uniform routing
            routing_weights = torch.ones(item_ids.size(0), self.num_specialists) / self.num_specialists
        
        # Get specialist embeddings
        specialist_outputs = []
        for specialist in self.specialist_embeddings:
            specialist_outputs.append(specialist(item_ids))
        
        specialist_embeddings = torch.stack(specialist_outputs, dim=1)  # (batch_size, num_specialists, embedding_dim)
        
        # Weighted combination
        routing_weights = routing_weights.unsqueeze(-1)  # (batch_size, num_specialists, 1)
        adaptive_embeddings = (specialist_embeddings * routing_weights).sum(dim=1)
        
        return adaptive_embeddings, routing_weights.squeeze(-1)
```

## 2. Graph-Based Embedding Methods

### Graph-Structured Embedding Networks

```python
class GraphEmbeddingNetwork(nn.Module):
    """Graph-based embedding using multiple graph structures"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 graph_types: List[str], graph_dims: List[int],
                 aggregation_method: str = 'attention', dropout_rate: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.graph_types = graph_types
        self.graph_dims = graph_dims
        self.aggregation_method = aggregation_method
        
        # Base node embeddings
        self.node_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Graph-specific embedding networks
        self.graph_networks = nn.ModuleDict()
        for graph_type, graph_dim in zip(graph_types, graph_dims):
            network = self._build_graph_network(graph_dim, dropout_rate)
            self.graph_networks[graph_type] = network
        
        # Aggregation mechanisms
        if aggregation_method == 'attention':
            self.graph_attention = nn.MultiheadAttention(
                embedding_dim, num_heads=4, dropout=dropout_rate
            )
        elif aggregation_method == 'weighted':
            self.graph_weights = nn.Parameter(torch.ones(len(graph_types)))
        
        # Final projection
        total_dim = embedding_dim * (len(graph_types) + 1)  # +1 for base embedding
        self.final_projection = nn.Sequential(
            nn.Linear(total_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.node_embeddings.weight)
    
    def _build_graph_network(self, graph_dim: int, dropout_rate: float) -> nn.Module:
        """Build graph-specific embedding network"""
        return nn.Sequential(
            nn.Linear(self.embedding_dim, graph_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(graph_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def message_passing(self, node_embeddings: torch.Tensor, 
                       adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Perform message passing on graph"""
        # Normalize adjacency matrix
        degree = adjacency_matrix.sum(dim=-1, keepdim=True)
        normalized_adj = adjacency_matrix / (degree + 1e-8)
        
        # Message passing: aggregate neighbor information
        messages = torch.matmul(normalized_adj, node_embeddings)
        
        return messages
    
    def forward(self, node_ids: torch.Tensor, 
                graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with multiple graph structures"""
        
        # Base node embeddings
        base_embeddings = self.node_embeddings(node_ids)
        
        # Graph-specific embeddings
        graph_embeddings = []
        
        for graph_type in self.graph_types:
            if graph_type in graph_data:
                adjacency_matrix = graph_data[graph_type]
                
                # Apply graph-specific network
                graph_emb = self.graph_networks[graph_type](base_embeddings)
                
                # Message passing
                if adjacency_matrix is not None:
                    graph_emb = self.message_passing(graph_emb, adjacency_matrix)
                
                graph_embeddings.append(graph_emb)
            else:
                # Use base embedding if graph not available
                graph_embeddings.append(base_embeddings)
        
        # Aggregation
        if self.aggregation_method == 'attention':
            # Stack embeddings for attention
            all_embeddings = [base_embeddings] + graph_embeddings
            stacked_embeddings = torch.stack(all_embeddings, dim=0)  # (num_graphs+1, batch_size, embedding_dim)
            
            # Self-attention across different graph views
            query = base_embeddings.unsqueeze(0)
            attended_emb, _ = self.graph_attention(query, stacked_embeddings, stacked_embeddings)
            final_embeddings = attended_emb.squeeze(0)
            
        elif self.aggregation_method == 'weighted':
            # Weighted sum
            graph_weights = F.softmax(self.graph_weights, dim=0)
            final_embeddings = base_embeddings.clone()
            
            for i, graph_emb in enumerate(graph_embeddings):
                final_embeddings += graph_weights[i] * graph_emb
            
        else:
            # Concatenation + projection
            all_embeddings = [base_embeddings] + graph_embeddings
            concatenated = torch.cat(all_embeddings, dim=-1)
            final_embeddings = self.final_projection(concatenated)
        
        return final_embeddings

class MetaPathEmbedding(nn.Module):
    """Meta-path based embeddings for heterogeneous graphs"""
    
    def __init__(self, node_types: List[str], vocab_sizes: Dict[str, int],
                 embedding_dim: int, meta_paths: List[List[str]],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.node_types = node_types
        self.vocab_sizes = vocab_sizes
        self.embedding_dim = embedding_dim
        self.meta_paths = meta_paths
        
        # Type-specific embeddings
        self.type_embeddings = nn.ModuleDict()
        for node_type, vocab_size in vocab_sizes.items():
            self.type_embeddings[node_type] = nn.Embedding(vocab_size, embedding_dim)
        
        # Meta-path specific networks
        self.metapath_networks = nn.ModuleList()
        for meta_path in meta_paths:
            network = self._build_metapath_network(len(meta_path), dropout_rate)
            self.metapath_networks.append(network)
        
        # Attention mechanism for meta-path fusion
        self.metapath_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=4, dropout=dropout_rate
        )
        
        # Final projection
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Initialize embeddings
        for embedding in self.type_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
    
    def _build_metapath_network(self, path_length: int, dropout_rate: float) -> nn.Module:
        """Build network for processing meta-path"""
        layers = []
        
        for i in range(path_length - 1):
            layers.extend([
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        return nn.Sequential(*layers)
    
    def walk_metapath(self, start_nodes: torch.Tensor, start_type: str,
                     meta_path: List[str], adjacency_matrices: Dict[Tuple[str, str], torch.Tensor]) -> torch.Tensor:
        """Perform meta-path walk"""
        
        current_embeddings = self.type_embeddings[start_type](start_nodes)
        current_type = start_type
        
        for next_type in meta_path[1:]:
            # Get adjacency matrix for current -> next type
            edge_key = (current_type, next_type)
            
            if edge_key in adjacency_matrices:
                adj_matrix = adjacency_matrices[edge_key]
                
                # Aggregate embeddings from neighbors of next type
                # This is a simplified version - in practice, you'd need proper node mapping
                current_embeddings = torch.matmul(adj_matrix, current_embeddings)
            
            current_type = next_type
        
        return current_embeddings
    
    def forward(self, node_ids: torch.Tensor, node_type: str,
                adjacency_matrices: Dict[Tuple[str, str], torch.Tensor]) -> torch.Tensor:
        """Forward pass with meta-path based aggregation"""
        
        # Collect embeddings from all meta-paths
        metapath_embeddings = []
        
        for i, meta_path in enumerate(self.meta_paths):
            if meta_path[0] == node_type:  # Meta-path starts with correct type
                # Perform meta-path walk
                path_emb = self.walk_metapath(node_ids, node_type, meta_path, adjacency_matrices)
                
                # Apply meta-path specific network
                processed_emb = self.metapath_networks[i](path_emb)
                metapath_embeddings.append(processed_emb)
        
        if metapath_embeddings:
            # Stack and attend
            stacked_embeddings = torch.stack(metapath_embeddings, dim=0)
            
            # Use base embedding as query
            base_emb = self.type_embeddings[node_type](node_ids)
            query = base_emb.unsqueeze(0)
            
            # Attention fusion
            fused_emb, _ = self.metapath_attention(query, stacked_embeddings, stacked_embeddings)
            final_embedding = self.final_projection(fused_emb.squeeze(0))
            
            return final_embedding
        else:
            # Fall back to base embedding
            return self.type_embeddings[node_type](node_ids)

class KnowledgeGraphEmbedding(nn.Module):
    """Knowledge graph embeddings with multiple relation types"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 kg_method: str = 'TransE', regularization: float = 0.01):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.kg_method = kg_method
        self.regularization = regularization
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Method-specific parameters
        if kg_method == 'TransR':
            # Relation-specific projection matrices
            self.relation_projections = nn.Embedding(num_relations, embedding_dim * embedding_dim)
        elif kg_method == 'DistMult':
            # Relations as diagonal matrices (represented as vectors)
            pass  # Use relation_embeddings directly
        elif kg_method == 'ComplEx':
            # Complex embeddings
            self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
            self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with proper normalization"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
        
        if hasattr(self, 'entity_embeddings_imag'):
            nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
            nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)
    
    def TransE_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """TransE scoring function: ||h + r - t||"""
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        score = head_emb + relation_emb - tail_emb
        return -torch.norm(score, p=2, dim=-1)  # Negative distance
    
    def TransR_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """TransR scoring function with relation-specific projections"""
        head_emb = self.entity_embeddings(head)
        tail_emb = self.entity_embeddings(tail)
        relation_emb = self.relation_embeddings(relation)
        
        # Get projection matrix
        projection_params = self.relation_projections(relation)
        projection_matrix = projection_params.view(-1, self.embedding_dim, self.embedding_dim)
        
        # Project entities to relation space
        head_proj = torch.bmm(head_emb.unsqueeze(1), projection_matrix).squeeze(1)
        tail_proj = torch.bmm(tail_emb.unsqueeze(1), projection_matrix).squeeze(1)
        
        # Compute score in relation space
        score = head_proj + relation_emb - tail_proj
        return -torch.norm(score, p=2, dim=-1)
    
    def DistMult_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """DistMult scoring function: <h, r, t>"""
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        score = torch.sum(head_emb * relation_emb * tail_emb, dim=-1)
        return score
    
    def ComplEx_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """ComplEx scoring function with complex embeddings"""
        # Real parts
        head_real = self.entity_embeddings(head)
        relation_real = self.relation_embeddings(relation)
        tail_real = self.entity_embeddings(tail)
        
        # Imaginary parts
        head_imag = self.entity_embeddings_imag(head)
        relation_imag = self.relation_embeddings_imag(relation)
        tail_imag = self.entity_embeddings_imag(tail)
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # Score is real part of <h, r, conj(t)>
        score = torch.sum(
            head_real * relation_real * tail_real +
            head_real * relation_imag * tail_imag +
            head_imag * relation_real * tail_imag -
            head_imag * relation_imag * tail_real, dim=-1
        )
        
        return score
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute knowledge graph scores"""
        
        if self.kg_method == 'TransE':
            return self.TransE_score(head, relation, tail)
        elif self.kg_method == 'TransR':
            return self.TransR_score(head, relation, tail)
        elif self.kg_method == 'DistMult':
            return self.DistMult_score(head, relation, tail)
        elif self.kg_method == 'ComplEx':
            return self.ComplEx_score(head, relation, tail)
        else:
            raise ValueError(f"Unknown KG method: {self.kg_method}")
    
    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """Get embedding for a specific entity"""
        if self.kg_method == 'ComplEx':
            real_part = self.entity_embeddings(torch.tensor([entity_id]))
            imag_part = self.entity_embeddings_imag(torch.tensor([entity_id]))
            return torch.cat([real_part, imag_part], dim=-1)
        else:
            return self.entity_embeddings(torch.tensor([entity_id]))
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss"""
        reg_loss = torch.norm(self.entity_embeddings.weight, p=2) + torch.norm(self.relation_embeddings.weight, p=2)
        
        if hasattr(self, 'entity_embeddings_imag'):
            reg_loss += torch.norm(self.entity_embeddings_imag.weight, p=2)
            reg_loss += torch.norm(self.relation_embeddings_imag.weight, p=2)
        
        return self.regularization * reg_loss
```

## 3. Multi-Modal Embedding Fusion

### Cross-Modal Attention Networks

```python
class CrossModalEmbeddingFusion(nn.Module):
    """Cross-modal embedding fusion with attention mechanisms"""
    
    def __init__(self, modality_dims: Dict[str, int], target_dim: int = 128,
                 fusion_method: str = 'attention', num_heads: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.modalities = list(modality_dims.keys())
        self.modality_dims = modality_dims
        self.target_dim = target_dim
        self.fusion_method = fusion_method
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            projection = nn.Sequential(
                nn.Linear(dim, target_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(target_dim)
            )
            self.modality_projections[modality] = projection
        
        # Fusion mechanisms
        if fusion_method == 'attention':
            self.cross_modal_attention = nn.MultiheadAttention(
                target_dim, num_heads=num_heads, dropout=dropout_rate
            )
        elif fusion_method == 'transformer':
            self.fusion_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(target_dim, num_heads, target_dim * 4, dropout_rate),
                num_layers=2
            )
        elif fusion_method == 'bilinear':
            # Bilinear fusion for pairs of modalities
            self.bilinear_layers = nn.ModuleDict()
            modality_pairs = [(m1, m2) for i, m1 in enumerate(self.modalities) 
                             for m2 in self.modalities[i+1:]]
            for m1, m2 in modality_pairs:
                bilinear = nn.Bilinear(target_dim, target_dim, target_dim)
                self.bilinear_layers[f"{m1}_{m2}"] = bilinear
        
        # Final fusion layer
        if fusion_method == 'bilinear':
            num_pairs = len(self.modalities) * (len(self.modalities) - 1) // 2
            final_input_dim = len(self.modalities) * target_dim + num_pairs * target_dim
        else:
            final_input_dim = target_dim
        
        self.final_fusion = nn.Sequential(
            nn.Linear(final_input_dim, target_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(target_dim, target_dim)
        )
        
    def attention_fusion(self, projected_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Attention-based cross-modal fusion"""
        
        # Stack all modality embeddings
        modality_embeddings = list(projected_modalities.values())
        stacked_embeddings = torch.stack(modality_embeddings, dim=0)  # (num_modalities, batch_size, target_dim)
        
        # Use first modality as query, all as keys and values
        query = modality_embeddings[0].unsqueeze(0)  # (1, batch_size, target_dim)
        
        # Cross-modal attention
        fused_embedding, attention_weights = self.cross_modal_attention(
            query, stacked_embeddings, stacked_embeddings
        )
        
        return fused_embedding.squeeze(0), attention_weights
    
    def transformer_fusion(self, projected_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Transformer-based fusion"""
        
        # Stack modalities as sequence
        modality_embeddings = list(projected_modalities.values())
        stacked_embeddings = torch.stack(modality_embeddings, dim=0)  # (seq_len, batch_size, target_dim)
        
        # Apply transformer
        fused_sequence = self.fusion_transformer(stacked_embeddings)
        
        # Aggregate sequence (mean pooling)
        fused_embedding = fused_sequence.mean(dim=0)
        
        return fused_embedding
    
    def bilinear_fusion(self, projected_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Bilinear fusion"""
        
        # Individual modality contributions
        individual_contributions = list(projected_modalities.values())
        
        # Pairwise bilinear interactions
        pairwise_interactions = []
        modalities = list(projected_modalities.keys())
        
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i+1:]:
                bilinear_key = f"{m1}_{m2}"
                if bilinear_key in self.bilinear_layers:
                    interaction = self.bilinear_layers[bilinear_key](
                        projected_modalities[m1], projected_modalities[m2]
                    )
                    pairwise_interactions.append(interaction)
        
        # Concatenate all contributions
        all_contributions = individual_contributions + pairwise_interactions
        fused_embedding = torch.cat(all_contributions, dim=-1)
        
        return fused_embedding
    
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with cross-modal fusion"""
        
        # Project each modality to common space
        projected_modalities = {}
        for modality, data in modality_data.items():
            if modality in self.modality_projections:
                projected = self.modality_projections[modality](data)
                projected_modalities[modality] = projected
        
        # Fusion
        attention_weights = None
        
        if self.fusion_method == 'attention':
            fused_embedding, attention_weights = self.attention_fusion(projected_modalities)
        elif self.fusion_method == 'transformer':
            fused_embedding = self.transformer_fusion(projected_modalities)
        elif self.fusion_method == 'bilinear':
            fused_embedding = self.bilinear_fusion(projected_modalities)
        else:
            # Simple concatenation
            embeddings = list(projected_modalities.values())
            fused_embedding = torch.cat(embeddings, dim=-1)
        
        # Final fusion layer
        final_embedding = self.final_fusion(fused_embedding)
        
        return final_embedding, attention_weights

class HierarchicalMultiModalFusion(nn.Module):
    """Hierarchical fusion of multi-modal embeddings"""
    
    def __init__(self, modality_groups: Dict[str, List[str]], 
                 modality_dims: Dict[str, int], target_dim: int = 128,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.modality_groups = modality_groups
        self.modality_dims = modality_dims
        self.target_dim = target_dim
        
        # Intra-group fusion networks
        self.intra_group_fusion = nn.ModuleDict()
        for group_name, modalities in modality_groups.items():
            group_dims = {mod: modality_dims[mod] for mod in modalities if mod in modality_dims}
            fusion_network = CrossModalEmbeddingFusion(
                group_dims, target_dim, fusion_method='attention', dropout_rate=dropout_rate
            )
            self.intra_group_fusion[group_name] = fusion_network
        
        # Inter-group fusion network
        group_dims = {group: target_dim for group in modality_groups.keys()}
        self.inter_group_fusion = CrossModalEmbeddingFusion(
            group_dims, target_dim, fusion_method='transformer', dropout_rate=dropout_rate
        )
        
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Hierarchical fusion: intra-group then inter-group"""
        
        # Step 1: Intra-group fusion
        group_embeddings = {}
        group_attention_weights = {}
        
        for group_name, modalities in self.modality_groups.items():
            # Get data for this group
            group_data = {mod: modality_data[mod] for mod in modalities if mod in modality_data}
            
            if group_data:
                # Fuse within group
                group_emb, attention_weights = self.intra_group_fusion[group_name](group_data)
                group_embeddings[group_name] = group_emb
                group_attention_weights[group_name] = attention_weights
        
        # Step 2: Inter-group fusion
        if group_embeddings:
            final_embedding, inter_group_attention = self.inter_group_fusion(group_embeddings)
            group_attention_weights['inter_group'] = inter_group_attention
        else:
            final_embedding = torch.zeros(1, self.target_dim)  # Fallback
        
        return final_embedding, group_attention_weights

class ModalitySpecificEmbedding(nn.Module):
    """Modality-specific embedding with cross-modal alignment"""
    
    def __init__(self, modality_vocabs: Dict[str, int], embedding_dim: int = 128,
                 alignment_method: str = 'cca', shared_space_dim: int = 64,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.modalities = list(modality_vocabs.keys())
        self.embedding_dim = embedding_dim
        self.alignment_method = alignment_method
        self.shared_space_dim = shared_space_dim
        
        # Modality-specific embedding tables
        self.modality_embeddings = nn.ModuleDict()
        for modality, vocab_size in modality_vocabs.items():
            embedding = nn.Embedding(vocab_size, embedding_dim)
            nn.init.xavier_uniform_(embedding.weight)
            self.modality_embeddings[modality] = embedding
        
        # Alignment networks
        if alignment_method == 'cca':
            # Canonical Correlation Analysis style alignment
            self.alignment_projections = nn.ModuleDict()
            for modality in self.modalities:
                projection = nn.Sequential(
                    nn.Linear(embedding_dim, shared_space_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                self.alignment_projections[modality] = projection
        
        elif alignment_method == 'adversarial':
            # Adversarial alignment
            self.domain_discriminator = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim // 2, len(self.modalities)),
                nn.Softmax(dim=-1)
            )
        
        elif alignment_method == 'contrastive':
            # Contrastive learning alignment
            self.contrastive_projection = nn.Sequential(
                nn.Linear(embedding_dim, shared_space_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_space_dim, shared_space_dim)
            )
    
    def cca_alignment_loss(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """CCA-style alignment loss"""
        if len(embeddings) < 2:
            return torch.tensor(0.0)
        
        modalities = list(embeddings.keys())
        total_loss = 0.0
        pairs = 0
        
        # Pairwise alignment
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                # Project to shared space
                proj1 = self.alignment_projections[mod1](embeddings[mod1])
                proj2 = self.alignment_projections[mod2](embeddings[mod2])
                
                # Correlation loss (maximize correlation)
                proj1_centered = proj1 - proj1.mean(dim=0, keepdim=True)
                proj2_centered = proj2 - proj2.mean(dim=0, keepdim=True)
                
                correlation = torch.sum(proj1_centered * proj2_centered, dim=0)
                correlation_loss = -correlation.mean()  # Negative to maximize
                
                total_loss += correlation_loss
                pairs += 1
        
        return total_loss / pairs if pairs > 0 else torch.tensor(0.0)
    
    def adversarial_alignment_loss(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Adversarial alignment loss"""
        total_loss = 0.0
        
        for i, (modality, emb) in enumerate(embeddings.items()):
            # Domain prediction
            domain_pred = self.domain_discriminator(emb)
            
            # True domain labels
            true_domain = torch.full((emb.size(0),), i, dtype=torch.long, device=emb.device)
            
            # Adversarial loss (want to confuse discriminator)
            adv_loss = F.cross_entropy(domain_pred, true_domain)
            total_loss += adv_loss
        
        return total_loss
    
    def contrastive_alignment_loss(self, embeddings: Dict[str, torch.Tensor],
                                  positive_pairs: List[Tuple[str, str, torch.Tensor]]) -> torch.Tensor:
        """Contrastive alignment loss"""
        if not positive_pairs:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        
        for mod1, mod2, pair_mask in positive_pairs:
            if mod1 in embeddings and mod2 in embeddings:
                # Project embeddings
                proj1 = self.contrastive_projection(embeddings[mod1])
                proj2 = self.contrastive_projection(embeddings[mod2])
                
                # Normalize
                proj1 = F.normalize(proj1, p=2, dim=-1)
                proj2 = F.normalize(proj2, p=2, dim=-1)
                
                # Contrastive loss
                similarity = torch.sum(proj1 * proj2, dim=-1)
                
                # Positive pairs should have high similarity
                positive_loss = -torch.log(torch.sigmoid(similarity[pair_mask])).mean()
                
                # Negative pairs should have low similarity
                negative_loss = -torch.log(torch.sigmoid(-similarity[~pair_mask])).mean()
                
                total_loss += positive_loss + negative_loss
        
        return total_loss
    
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass to get modality-specific embeddings"""
        
        embeddings = {}
        for modality, data in modality_data.items():
            if modality in self.modality_embeddings:
                emb = self.modality_embeddings[modality](data)
                embeddings[modality] = emb
        
        return embeddings
    
    def compute_alignment_loss(self, embeddings: Dict[str, torch.Tensor],
                              positive_pairs: List[Tuple[str, str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute cross-modal alignment loss"""
        
        if self.alignment_method == 'cca':
            return self.cca_alignment_loss(embeddings)
        elif self.alignment_method == 'adversarial':
            return self.adversarial_alignment_loss(embeddings)
        elif self.alignment_method == 'contrastive':
            return self.contrastive_alignment_loss(embeddings, positive_pairs or [])
        else:
            return torch.tensor(0.0)
```

## Key Takeaways

1. **Dynamic Embeddings**: Time-aware and contextual embeddings adapt to changing patterns and situational factors

2. **Graph-Based Methods**: Leverage graph structures for richer embedding representations through message passing

3. **Multi-Modal Fusion**: Cross-modal attention and hierarchical fusion enable effective integration of diverse data types

4. **Adaptive Systems**: Embedding systems that specialize based on usage patterns and semantic similarity

5. **Knowledge Integration**: Knowledge graph embeddings enhance recommendations with structured external knowledge

6. **Alignment Techniques**: Cross-modal alignment ensures semantic consistency across different modalities

## Study Questions

### Beginner Level
1. What are the advantages of dynamic embeddings over static ones?
2. How do contextual embeddings adapt to different situations?
3. What is the role of message passing in graph-based embeddings?
4. How does cross-modal attention work in multi-modal fusion?

### Intermediate Level
1. Compare different temporal update methods for dynamic embeddings
2. How would you design an embedding system for cold-start items?
3. What are the challenges in aligning embeddings across different modalities?
4. How can knowledge graphs enhance recommendation embeddings?

### Advanced Level
1. Design a real-time embedding update system for streaming data
2. Implement a privacy-preserving embedding system using differential privacy
3. How would you handle embedding drift in production systems?
4. Design a federated embedding learning system for cross-domain recommendations

## Course Summary

Over these 6 days, we've built a comprehensive foundation in AI/ML for Search and Recommendation systems:

- **Days 1-2**: Fundamentals of IR and traditional recommendation systems
- **Days 3-4**: Content-based and hybrid recommendation approaches  
- **Day 5**: Advanced hybrid systems with sophisticated evaluation methods
- **Day 6**: Cutting-edge neural approaches including transformers, GNNs, multi-task learning and advanced embeddings

This comprehensive course provides both theoretical understanding and practical implementation skills for building modern, production-ready recommendation systems!