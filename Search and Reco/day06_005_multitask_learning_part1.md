# Day 6.5 Part 1: Multi-task Learning Foundations for Search and Recommendation

## Learning Objectives
- Master multi-task neural architectures for recommendations
- Implement shared and task-specific layer designs
- Design multi-objective optimization strategies
- Build cross-domain knowledge transfer systems
- Understand parameter sharing strategies in multi-task learning

## 1. Multi-task Learning Fundamentals

### Base Multi-task Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class MultiTaskBase(nn.Module, ABC):
    """Base class for multi-task learning models"""
    
    def __init__(self, input_dim: int, task_configs: Dict[str, Dict[str, Any]]):
        super().__init__()
        
        self.input_dim = input_dim
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Shared layers
        self.shared_layers = self._build_shared_layers()
        
        # Task-specific layers
        self.task_specific_layers = nn.ModuleDict()
        for task_name, config in task_configs.items():
            self.task_specific_layers[task_name] = self._build_task_specific_layers(config)
    
    @abstractmethod
    def _build_shared_layers(self) -> nn.Module:
        """Build shared layers across all tasks"""
        pass
    
    @abstractmethod
    def _build_task_specific_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build task-specific layers"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, task_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass for all tasks"""
        pass

class HardParameterSharing(MultiTaskBase):
    """Hard parameter sharing multi-task architecture"""
    
    def __init__(self, input_dim: int, shared_dims: List[int], 
                 task_configs: Dict[str, Dict[str, Any]], dropout_rate: float = 0.2):
        
        self.shared_dims = shared_dims
        self.dropout_rate = dropout_rate
        
        super().__init__(input_dim, task_configs)
    
    def _build_shared_layers(self) -> nn.Module:
        """Build shared bottom layers"""
        layers = []
        
        input_dim = self.input_dim
        for shared_dim in self.shared_dims:
            layers.extend([
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.BatchNorm1d(shared_dim)
            ])
            input_dim = shared_dim
        
        return nn.Sequential(*layers)
    
    def _build_task_specific_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build task-specific top layers"""
        task_dims = config.get('hidden_dims', [64, 32])
        output_dim = config.get('output_dim', 1)
        task_type = config.get('task_type', 'regression')
        
        layers = []
        input_dim = self.shared_dims[-1] if self.shared_dims else self.input_dim
        
        for task_dim in task_dims:
            layers.extend([
                nn.Linear(input_dim, task_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            input_dim = task_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))
        
        # Add appropriate activation for task type
        if task_type == 'classification' and output_dim == 1:
            layers.append(nn.Sigmoid())
        elif task_type == 'classification' and output_dim > 1:
            layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, task_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through shared and task-specific layers"""
        
        # Shared representation
        shared_features = self.shared_layers(x)
        
        # Task-specific outputs
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_specific_layers[task_name](shared_features)
            outputs[task_name] = task_output
        
        return outputs

class SoftParameterSharing(MultiTaskBase):
    """Soft parameter sharing with cross-stitch networks"""
    
    def __init__(self, input_dim: int, task_configs: Dict[str, Dict[str, Any]],
                 shared_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        
        self.shared_dims = shared_dims
        self.dropout_rate = dropout_rate
        
        super().__init__(input_dim, task_configs)
        
        # Cross-stitch units for soft parameter sharing
        self.cross_stitch_units = nn.ModuleList()
        
        n_tasks = len(self.task_names)
        for i in range(len(shared_dims)):
            # Cross-stitch matrix for each layer
            cross_stitch = nn.Linear(shared_dims[i] * n_tasks, shared_dims[i] * n_tasks, bias=False)
            # Initialize as identity with small noise
            with torch.no_grad():
                cross_stitch.weight.copy_(torch.eye(shared_dims[i] * n_tasks) + 0.1 * torch.randn(shared_dims[i] * n_tasks, shared_dims[i] * n_tasks))
            self.cross_stitch_units.append(cross_stitch)
    
    def _build_shared_layers(self) -> nn.Module:
        """Build task-specific shared layers (one per task)"""
        task_networks = nn.ModuleDict()
        
        for task_name in self.task_names:
            layers = []
            input_dim = self.input_dim
            
            for shared_dim in self.shared_dims:
                layers.extend([
                    nn.Linear(input_dim, shared_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ])
                input_dim = shared_dim
            
            task_networks[task_name] = nn.Sequential(*layers)
        
        return task_networks
    
    def _build_task_specific_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build final task-specific layers"""
        task_dims = config.get('hidden_dims', [32])
        output_dim = config.get('output_dim', 1)
        task_type = config.get('task_type', 'regression')
        
        layers = []
        input_dim = self.shared_dims[-1]
        
        for task_dim in task_dims:
            layers.extend([
                nn.Linear(input_dim, task_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            input_dim = task_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        
        if task_type == 'classification' and output_dim == 1:
            layers.append(nn.Sigmoid())
        elif task_type == 'classification' and output_dim > 1:
            layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, task_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with cross-stitch connections"""
        
        # Initialize task-specific features
        task_features = {}
        for task_name in self.task_names:
            task_features[task_name] = x
        
        # Pass through shared layers with cross-stitch connections
        for layer_idx in range(len(self.shared_dims)):
            # Get outputs from each task's network at this layer
            layer_outputs = []
            for task_name in self.task_names:
                # Get the layer_idx-th block from the task network
                layer_start = layer_idx * 3  # Each block has 3 components (Linear, ReLU, Dropout)
                layer_end = layer_start + 3
                
                if layer_idx == 0:
                    task_input = x
                else:
                    task_input = task_features[task_name]
                
                # Apply the specific layer block
                for i in range(layer_start, min(layer_end, len(self.shared_layers[task_name]))):
                    task_input = self.shared_layers[task_name][i](task_input)
                
                layer_outputs.append(task_input)
            
            # Concatenate outputs from all tasks
            concatenated = torch.cat(layer_outputs, dim=-1)
            
            # Apply cross-stitch unit
            cross_stitched = self.cross_stitch_units[layer_idx](concatenated)
            
            # Split back to task-specific features
            split_size = self.shared_dims[layer_idx]
            split_features = torch.split(cross_stitched, split_size, dim=-1)
            
            for i, task_name in enumerate(self.task_names):
                task_features[task_name] = split_features[i]
        
        # Apply task-specific layers
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_specific_layers[task_name](task_features[task_name])
            outputs[task_name] = task_output
        
        return outputs

class MultiModalMultiTask(nn.Module):
    """Multi-modal multi-task learning for recommendations"""
    
    def __init__(self, user_vocab_size: int, item_vocab_size: int, text_vocab_size: int,
                 embedding_dim: int = 64, hidden_dims: List[int] = [128, 64],
                 task_configs: Dict[str, Dict[str, Any]] = None, dropout_rate: float = 0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.task_configs = task_configs or {
            'rating': {'output_dim': 1, 'task_type': 'regression'},
            'click': {'output_dim': 1, 'task_type': 'classification'},
            'category': {'output_dim': 10, 'task_type': 'classification'}
        }
        
        # Embeddings for different modalities
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)
        self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim)
        
        # Modality-specific encoders
        self.user_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.item_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_dims[0], num_heads=4, dropout=dropout_rate
        )
        
        # Shared fusion layers
        fusion_input_dim = hidden_dims[0] * 3  # Three modalities
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dims[1])
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in self.task_configs.items():
            self.task_heads[task_name] = self._build_task_head(hidden_dims[1], config)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.text_embedding.weight)
    
    def _build_task_head(self, input_dim: int, config: Dict[str, Any]) -> nn.Module:
        """Build task-specific prediction head"""
        
        output_dim = config.get('output_dim', 1)
        task_type = config.get('task_type', 'regression')
        
        layers = [
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim)
        ]
        
        if task_type == 'classification' and output_dim == 1:
            layers.append(nn.Sigmoid())
        elif task_type == 'classification' and output_dim > 1:
            layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                text_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-modal forward pass"""
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_ids)
        text_emb = self.text_embedding(text_ids).mean(dim=1)  # Average over sequence
        
        # Encode each modality
        user_encoded = self.user_encoder(user_emb)  # (batch_size, hidden_dim)
        item_encoded = self.item_encoder(item_emb)
        text_encoded = self.text_encoder(text_emb)
        
        # Stack for cross-modal attention
        modality_features = torch.stack([user_encoded, item_encoded, text_encoded], dim=0)
        # (3, batch_size, hidden_dim)
        
        # Apply cross-modal attention
        attended_features, attention_weights = self.cross_modal_attention(
            modality_features, modality_features, modality_features
        )
        
        # Flatten attended features
        attended_features = attended_features.transpose(0, 1)  # (batch_size, 3, hidden_dim)
        flattened_features = attended_features.reshape(attended_features.size(0), -1)
        
        # Fusion
        fused_features = self.fusion_layers(flattened_features)
        
        # Task-specific predictions
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[task_name] = task_head(fused_features)
        
        return outputs

class ProgressiveMultiTask(nn.Module):
    """Progressive Multi-task Learning with Task Progression"""
    
    def __init__(self, input_dim: int, base_dims: List[int],
                 task_sequence: List[str], task_configs: Dict[str, Dict[str, Any]],
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.base_dims = base_dims
        self.task_sequence = task_sequence
        self.task_configs = task_configs
        
        # Base network (shared across all tasks)
        self.base_network = self._build_base_network()
        
        # Progressive columns (one per task)
        self.progressive_columns = nn.ModuleDict()
        self.lateral_connections = nn.ModuleDict()
        
        for i, task_name in enumerate(task_sequence):
            # Progressive column for this task
            column = self._build_progressive_column(task_configs[task_name])
            self.progressive_columns[task_name] = column
            
            # Lateral connections from previous tasks
            if i > 0:
                lateral_adapters = nn.ModuleDict()
                for prev_task in task_sequence[:i]:
                    # Adapter to connect previous task features
                    adapter = nn.Sequential(
                        nn.Linear(base_dims[-1], base_dims[-1] // 2),
                        nn.ReLU(),
                        nn.Linear(base_dims[-1] // 2, base_dims[-1])
                    )
                    lateral_adapters[prev_task] = adapter
                
                self.lateral_connections[task_name] = lateral_adapters
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def _build_base_network(self) -> nn.Module:
        """Build base shared network"""
        layers = []
        input_dim = self.input_dim
        
        for base_dim in self.base_dims:
            layers.extend([
                nn.Linear(input_dim, base_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = base_dim
        
        return nn.Sequential(*layers)
    
    def _build_progressive_column(self, task_config: Dict[str, Any]) -> nn.Module:
        """Build progressive column for a specific task"""
        
        task_dims = task_config.get('hidden_dims', [64, 32])
        output_dim = task_config.get('output_dim', 1)
        task_type = task_config.get('task_type', 'regression')
        
        layers = []
        input_dim = self.base_dims[-1]
        
        for task_dim in task_dims:
            layers.extend([
                nn.Linear(input_dim, task_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = task_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        
        if task_type == 'classification' and output_dim == 1:
            layers.append(nn.Sigmoid())
        elif task_type == 'classification' and output_dim > 1:
            layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, target_tasks: List[str] = None) -> Dict[str, torch.Tensor]:
        """Progressive forward pass"""
        
        if target_tasks is None:
            target_tasks = self.task_sequence
        
        # Base features
        base_features = self.base_network(x)
        
        # Progressive task execution
        outputs = {}
        task_features = {}
        
        for task_name in self.task_sequence:
            if task_name not in target_tasks:
                continue
            
            # Start with base features
            current_features = base_features
            
            # Add lateral connections from previous tasks
            if task_name in self.lateral_connections:
                lateral_contributions = []
                for prev_task, adapter in self.lateral_connections[task_name].items():
                    if prev_task in task_features:
                        adapted_features = adapter(task_features[prev_task])
                        lateral_contributions.append(adapted_features)
                
                if lateral_contributions:
                    # Combine lateral contributions
                    lateral_sum = torch.stack(lateral_contributions, dim=0).sum(dim=0)
                    current_features = current_features + lateral_sum
            
            # Apply task-specific column
            task_output = self.progressive_columns[task_name](current_features)
            
            # Store task features (before final output layer)
            # Get intermediate features from the column
            intermediate_features = current_features
            for layer in self.progressive_columns[task_name][:-1]:  # Exclude final layer
                intermediate_features = layer(intermediate_features)
            
            task_features[task_name] = intermediate_features
            outputs[task_name] = task_output
        
        return outputs
```

## 2. Multi-objective Optimization Strategies

### Advanced Loss Functions and Optimization

```python
class MultiTaskLoss(nn.Module):
    """Multi-task loss with various balancing strategies"""
    
    def __init__(self, task_names: List[str], loss_types: Dict[str, str],
                 balancing_method: str = 'uncertainty', device: str = 'cpu'):
        super().__init__()
        
        self.task_names = task_names
        self.loss_types = loss_types
        self.balancing_method = balancing_method
        self.device = device
        
        # Task-specific loss functions
        self.loss_functions = {}
        for task_name, loss_type in loss_types.items():
            if loss_type == 'mse':
                self.loss_functions[task_name] = nn.MSELoss()
            elif loss_type == 'bce':
                self.loss_functions[task_name] = nn.BCELoss()
            elif loss_type == 'ce':
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif loss_type == 'mae':
                self.loss_functions[task_name] = nn.L1Loss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Balancing parameters
        if balancing_method == 'uncertainty':
            # Learnable uncertainty parameters
            self.log_vars = nn.Parameter(torch.zeros(len(task_names), device=device))
        elif balancing_method == 'weights':
            # Fixed task weights
            self.task_weights = nn.Parameter(torch.ones(len(task_names), device=device))
        elif balancing_method == 'gradnorm':
            # GradNorm balancing
            self.task_weights = nn.Parameter(torch.ones(len(task_names), device=device))
            self.initial_losses = None
            self.alpha = 0.12  # GradNorm hyperparameter
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                model_parameters: List[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss"""
        
        task_losses = {}
        
        # Compute individual task losses
        for i, task_name in enumerate(self.task_names):
            if task_name in predictions and task_name in targets:
                loss_fn = self.loss_functions[task_name]
                
                pred = predictions[task_name]
                target = targets[task_name]
                
                # Handle different tensor shapes
                if pred.dim() > target.dim():
                    target = target.squeeze()
                elif pred.dim() < target.dim():
                    pred = pred.squeeze()
                
                task_loss = loss_fn(pred, target)
                task_losses[task_name] = task_loss
        
        # Balance task losses
        if self.balancing_method == 'equal':
            total_loss = sum(task_losses.values()) / len(task_losses)
        
        elif self.balancing_method == 'uncertainty':
            # Uncertainty-based weighting
            total_loss = 0
            for i, task_name in enumerate(self.task_names):
                if task_name in task_losses:
                    precision = torch.exp(-self.log_vars[i])
                    total_loss += precision * task_losses[task_name] + self.log_vars[i]
            
        elif self.balancing_method == 'weights':
            # Weighted combination
            total_loss = 0
            weights_sum = 0
            for i, task_name in enumerate(self.task_names):
                if task_name in task_losses:
                    weight = F.softmax(self.task_weights, dim=0)[i]
                    total_loss += weight * task_losses[task_name]
                    weights_sum += weight
            total_loss = total_loss / weights_sum if weights_sum > 0 else total_loss
        
        elif self.balancing_method == 'gradnorm':
            # GradNorm balancing
            if model_parameters is None:
                # Fallback to equal weighting
                total_loss = sum(task_losses.values()) / len(task_losses)
            else:
                total_loss = self._gradnorm_loss(task_losses, model_parameters)
        
        else:
            raise ValueError(f"Unknown balancing method: {self.balancing_method}")
        
        return total_loss, task_losses
    
    def _gradnorm_loss(self, task_losses: Dict[str, torch.Tensor],
                      model_parameters: List[torch.Tensor]) -> torch.Tensor:
        """GradNorm loss balancing"""
        
        if self.initial_losses is None:
            # Store initial losses
            self.initial_losses = {name: loss.detach() for name, loss in task_losses.items()}
        
        # Compute weighted loss
        weighted_losses = []
        for i, task_name in enumerate(self.task_names):
            if task_name in task_losses:
                weight = F.softmax(self.task_weights, dim=0)[i]
                weighted_losses.append(weight * task_losses[task_name])
        
        total_loss = sum(weighted_losses)
        
        # Compute gradients for GradNorm update
        if self.training:
            # Compute gradient norms
            grad_norms = []
            for i, task_name in enumerate(self.task_names):
                if task_name in task_losses:
                    # Get gradients w.r.t. shared parameters
                    grads = torch.autograd.grad(
                        task_losses[task_name], model_parameters,
                        retain_graph=True, create_graph=True
                    )
                    grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
                    grad_norms.append(grad_norm)
            
            if grad_norms:
                grad_norms = torch.stack(grad_norms)
                
                # Compute relative inverse training rates
                loss_ratios = []
                for i, task_name in enumerate(self.task_names):
                    if task_name in task_losses:
                        ratio = task_losses[task_name] / self.initial_losses[task_name]
                        loss_ratios.append(ratio)
                
                loss_ratios = torch.stack(loss_ratios)
                mean_loss_ratio = loss_ratios.mean()
                
                # Compute target gradient norms
                target_grad_norms = grad_norms.mean() * (loss_ratios / mean_loss_ratio) ** self.alpha
                
                # GradNorm loss for updating task weights
                gradnorm_loss = F.l1_loss(grad_norms, target_grad_norms.detach())
                
                # Add GradNorm loss to total loss
                total_loss = total_loss + 0.1 * gradnorm_loss
        
        return total_loss

class AdaptiveTaskScheduler:
    """Adaptive scheduler for multi-task training"""
    
    def __init__(self, task_names: List[str], scheduling_method: str = 'loss_based',
                 temperature: float = 1.0, update_frequency: int = 100):
        
        self.task_names = task_names
        self.scheduling_method = scheduling_method
        self.temperature = temperature
        self.update_frequency = update_frequency
        
        # Task statistics
        self.task_losses = {name: [] for name in task_names}
        self.task_accuracies = {name: [] for name in task_names}
        self.task_probabilities = torch.ones(len(task_names)) / len(task_names)
        
        self.step_count = 0
    
    def update_statistics(self, task_losses: Dict[str, float],
                         task_metrics: Dict[str, float] = None):
        """Update task statistics"""
        
        for task_name in self.task_names:
            if task_name in task_losses:
                self.task_losses[task_name].append(task_losses[task_name])
        
        if task_metrics:
            for task_name in self.task_names:
                if task_name in task_metrics:
                    self.task_accuracies[task_name].append(task_metrics[task_name])
        
        self.step_count += 1
        
        # Update probabilities periodically
        if self.step_count % self.update_frequency == 0:
            self._update_probabilities()
    
    def _update_probabilities(self):
        """Update task sampling probabilities"""
        
        if self.scheduling_method == 'uniform':
            # Uniform sampling
            self.task_probabilities = torch.ones(len(self.task_names)) / len(self.task_names)
        
        elif self.scheduling_method == 'loss_based':
            # Sample based on recent loss values
            recent_losses = []
            for task_name in self.task_names:
                if self.task_losses[task_name]:
                    # Use recent losses (last 10 values)
                    recent_loss = np.mean(self.task_losses[task_name][-10:])
                    recent_losses.append(recent_loss)
                else:
                    recent_losses.append(1.0)  # Default loss
            
            # Higher loss -> higher probability
            recent_losses = torch.tensor(recent_losses, dtype=torch.float32)
            self.task_probabilities = F.softmax(recent_losses / self.temperature, dim=0)
        
        elif self.scheduling_method == 'performance_based':
            # Sample based on recent performance (lower performance -> higher probability)
            recent_accuracies = []
            for task_name in self.task_names:
                if self.task_accuracies[task_name]:
                    recent_acc = np.mean(self.task_accuracies[task_name][-10:])
                    recent_accuracies.append(1.0 - recent_acc)  # Invert for difficulty
                else:
                    recent_accuracies.append(0.5)  # Default
            
            recent_accuracies = torch.tensor(recent_accuracies, dtype=torch.float32)
            self.task_probabilities = F.softmax(recent_accuracies / self.temperature, dim=0)
    
    def sample_task(self) -> str:
        """Sample next task to train on"""
        
        if self.scheduling_method == 'round_robin':
            # Simple round robin
            task_idx = self.step_count % len(self.task_names)
            return self.task_names[task_idx]
        else:
            # Probabilistic sampling
            task_idx = torch.multinomial(self.task_probabilities, 1).item()
            return self.task_names[task_idx]
    
    def get_task_probabilities(self) -> Dict[str, float]:
        """Get current task probabilities"""
        return {name: prob.item() for name, prob in zip(self.task_names, self.task_probabilities)}

class MetaLearningMultiTask(nn.Module):
    """Meta-learning approach for multi-task recommendations"""
    
    def __init__(self, input_dim: int, meta_dim: int, task_configs: Dict[str, Dict[str, Any]],
                 base_network_dims: List[int] = [128, 64], meta_network_dims: List[int] = [64, 32]):
        super().__init__()
        
        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Base feature extractor
        self.base_network = self._build_base_network(base_network_dims)
        
        # Meta network for generating task-specific parameters
        self.meta_network = self._build_meta_network(meta_network_dims)
        
        # Task embedding
        self.task_embedding = nn.Embedding(len(self.task_names), meta_dim)
        
        # Task-specific output dimensions
        self.task_output_dims = {name: config.get('output_dim', 1) for name, config in task_configs.items()}
        
        # Generate parameters for task-specific heads
        max_output_dim = max(self.task_output_dims.values())
        self.meta_output_dim = base_network_dims[-1] * max_output_dim + max_output_dim  # weights + bias
        
        self.parameter_generator = nn.Linear(meta_network_dims[-1], self.meta_output_dim)
        
    def _build_base_network(self, dims: List[int]) -> nn.Module:
        """Build base feature extraction network"""
        layers = []
        input_dim = self.input_dim
        
        for dim in dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = dim
        
        return nn.Sequential(*layers)
    
    def _build_meta_network(self, dims: List[int]) -> nn.Module:
        """Build meta network for parameter generation"""
        layers = []
        input_dim = self.meta_dim
        
        for dim in dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = dim
        
        return nn.Sequential(*layers)
    
    def generate_task_parameters(self, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate parameters for a specific task"""
        
        task_idx = self.task_names.index(task_name)
        task_emb = self.task_embedding(torch.tensor([task_idx], device=next(self.parameters()).device))
        
        # Generate parameters through meta network
        meta_features = self.meta_network(task_emb)
        raw_parameters = self.parameter_generator(meta_features)
        
        # Split into weights and bias
        output_dim = self.task_output_dims[task_name]
        base_dim = list(self.base_network.modules())[-2].out_features
        
        weight_size = base_dim * output_dim
        bias_size = output_dim
        
        weights = raw_parameters[:, :weight_size].view(output_dim, base_dim)
        bias = raw_parameters[:, weight_size:weight_size + bias_size].view(output_dim)
        
        return weights, bias
    
    def forward(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """Forward pass for a specific task"""
        
        # Extract base features
        base_features = self.base_network(x)
        
        # Generate task-specific parameters
        weights, bias = self.generate_task_parameters(task_name)
        
        # Apply task-specific linear layer
        output = F.linear(base_features, weights, bias)
        
        # Apply task-specific activation
        task_type = self.task_configs[task_name].get('task_type', 'regression')
        output_dim = self.task_output_dims[task_name]
        
        if task_type == 'classification' and output_dim == 1:
            output = torch.sigmoid(output)
        elif task_type == 'classification' and output_dim > 1:
            output = F.softmax(output, dim=-1)
        
        return output
    
    def forward_all_tasks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for all tasks"""
        
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.forward(x, task_name)
        
        return outputs
```

## Key Takeaways (Part 1)

1. **Parameter Sharing**: Different strategies (hard, soft, progressive) offer various trade-offs between sharing and specialization

2. **Multi-modal Integration**: Combining different data modalities through attention mechanisms enhances recommendation quality

3. **Loss Balancing**: Sophisticated balancing strategies (uncertainty weighting, GradNorm) improve multi-task optimization

4. **Adaptive Scheduling**: Dynamic task scheduling based on loss or performance improves training efficiency  

5. **Meta-learning**: Generating task-specific parameters enables better adaptation to diverse tasks

6. **Progressive Learning**: Sequential task learning with lateral connections allows knowledge transfer

## Study Questions (Part 1)

### Beginner Level
1. What are the main differences between hard and soft parameter sharing?
2. How does uncertainty weighting help in multi-task loss balancing?
3. What are the advantages of multi-modal multi-task learning?
4. How does progressive multi-task learning work?

### Intermediate Level
1. Compare different multi-task loss balancing strategies and their use cases
2. How would you design a multi-task architecture for both explicit and implicit feedback?
3. What are the challenges in training multi-modal multi-task systems?
4. How does meta-learning help in multi-task scenarios?

### Advanced Level
1. Design a multi-task system that handles temporal dynamics across tasks
2. Implement a federated multi-task learning approach for recommendations
3. How would you adapt multi-task learning for real-time recommendation systems?
4. Design a multi-task architecture that can dynamically add new tasks

## Next Part Preview

In **Part 2**, we'll cover:
- Cross-domain knowledge transfer mechanisms
- Advanced meta-learning approaches (MAML, Reptile)
- Multi-task reinforcement learning for recommendations
- Continual learning in multi-task settings
- Advanced training techniques and optimization strategies
- Real-world deployment considerations for multi-task systems

The second part will focus on more advanced concepts and practical deployment aspects!