# Day 6.5 Part 2: Advanced Multi-task Learning and Cross-domain Transfer

## Learning Objectives
- Master cross-domain knowledge transfer mechanisms
- Implement advanced meta-learning approaches (MAML, Reptile)
- Design multi-task reinforcement learning for recommendations
- Build continual learning systems for evolving tasks
- Develop advanced training techniques and optimization strategies

## 1. Cross-Domain Knowledge Transfer

### Domain Adaptation Networks

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
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class DomainAdversarialNetwork(nn.Module):
    """Domain Adversarial Neural Network for Cross-Domain Recommendations"""
    
    def __init__(self, feature_dim: int, hidden_dims: List[int], 
                 n_domains: int, task_configs: Dict[str, Dict[str, Any]],
                 lambda_domain: float = 1.0, dropout_rate: float = 0.2):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_domains = n_domains
        self.lambda_domain = lambda_domain
        self.task_configs = task_configs
        
        # Feature extractor (shared across domains)
        self.feature_extractor = self._build_feature_extractor(hidden_dims, dropout_rate)
        
        # Task predictors (one per task)
        self.task_predictors = nn.ModuleDict()
        for task_name, config in task_configs.items():
            self.task_predictors[task_name] = self._build_task_predictor(
                hidden_dims[-1], config, dropout_rate
            )
        
        # Domain classifier (adversarial)
        self.domain_classifier = self._build_domain_classifier(
            hidden_dims[-1], n_domains, dropout_rate
        )
        
    def _build_feature_extractor(self, hidden_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build shared feature extractor"""
        layers = []
        input_dim = self.feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_task_predictor(self, input_dim: int, config: Dict[str, Any], 
                             dropout_rate: float) -> nn.Module:
        """Build task-specific predictor"""
        output_dim = config.get('output_dim', 1)
        task_type = config.get('task_type', 'regression')
        
        layers = [
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, output_dim)
        ]
        
        if task_type == 'classification' and output_dim == 1:
            layers.append(nn.Sigmoid())
        elif task_type == 'classification' and output_dim > 1:
            layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers)
    
    def _build_domain_classifier(self, input_dim: int, n_domains: int, 
                                dropout_rate: float) -> nn.Module:
        """Build domain classifier"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, n_domains),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, task_name: str, 
                alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient reversal
        Args:
            x: input features
            task_name: target task
            alpha: gradient reversal strength
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Task prediction
        task_output = self.task_predictors[task_name](features)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return task_output, domain_output

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None

class CrossDomainTransferNetwork(nn.Module):
    """Cross-Domain Transfer Network with Domain-Specific Adaptation"""
    
    def __init__(self, source_vocab_sizes: Dict[str, int], target_vocab_sizes: Dict[str, int],
                 embedding_dim: int = 64, hidden_dims: List[int] = [128, 64],
                 shared_dims: List[int] = [32], dropout_rate: float = 0.2):
        super().__init__()
        
        self.source_domains = list(source_vocab_sizes.keys())
        self.target_domains = list(target_vocab_sizes.keys())
        self.embedding_dim = embedding_dim
        
        # Domain-specific embeddings
        self.source_embeddings = nn.ModuleDict({
            domain: nn.Embedding(vocab_size, embedding_dim)
            for domain, vocab_size in source_vocab_sizes.items()
        })
        
        self.target_embeddings = nn.ModuleDict({
            domain: nn.Embedding(vocab_size, embedding_dim)
            for domain, vocab_size in target_vocab_sizes.items()
        })
        
        # Domain encoders
        self.domain_encoders = nn.ModuleDict()
        for domain in self.source_domains + self.target_domains:
            encoder = self._build_domain_encoder(embedding_dim, hidden_dims, dropout_rate)
            self.domain_encoders[domain] = encoder
        
        # Cross-domain mapping network
        self.cross_domain_mapper = self._build_cross_domain_mapper(
            hidden_dims[-1], shared_dims, dropout_rate
        )
        
        # Domain-specific adaptation layers
        self.adaptation_layers = nn.ModuleDict()
        for target_domain in self.target_domains:
            adaptation = nn.Sequential(
                nn.Linear(shared_dims[-1], shared_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dims[-1], 1)  # Recommendation score
            )
            self.adaptation_layers[target_domain] = adaptation
        
        # Domain alignment loss components
        self.domain_discriminator = nn.Sequential(
            nn.Linear(shared_dims[-1], shared_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_dims[-1] // 2, len(self.source_domains) + len(self.target_domains)),
            nn.Softmax(dim=-1)
        )
        
    def _build_domain_encoder(self, input_dim: int, hidden_dims: List[int], 
                             dropout_rate: float) -> nn.Module:
        """Build domain-specific encoder"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_cross_domain_mapper(self, input_dim: int, shared_dims: List[int],
                                  dropout_rate: float) -> nn.Module:
        """Build cross-domain mapping network"""
        layers = []
        current_dim = input_dim
        
        for shared_dim in shared_dims:
            layers.extend([
                nn.Linear(current_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = shared_dim
        
        return nn.Sequential(*layers)
    
    def encode_domain(self, domain: str, user_ids: torch.Tensor, 
                     item_ids: torch.Tensor) -> torch.Tensor:
        """Encode user-item pairs for a specific domain"""
        
        if domain in self.source_domains:
            embeddings = self.source_embeddings[domain]
        else:
            embeddings = self.target_embeddings[domain]
        
        # Get embeddings
        user_emb = embeddings(user_ids)
        item_emb = embeddings(item_ids)
        
        # Combine user and item embeddings
        combined = user_emb * item_emb  # Element-wise product
        
        # Encode through domain-specific encoder
        encoded = self.domain_encoders[domain](combined)
        
        return encoded
    
    def forward(self, source_domain: str, target_domain: str,
                source_user_ids: torch.Tensor, source_item_ids: torch.Tensor,
                target_user_ids: torch.Tensor, target_item_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cross-domain forward pass"""
        
        # Encode source and target domain data
        source_encoded = self.encode_domain(source_domain, source_user_ids, source_item_ids)
        target_encoded = self.encode_domain(target_domain, target_user_ids, target_item_ids)
        
        # Map to shared representation space
        source_shared = self.cross_domain_mapper(source_encoded)
        target_shared = self.cross_domain_mapper(target_encoded)
        
        # Generate predictions for target domain
        target_predictions = self.adaptation_layers[target_domain](target_shared)
        
        # Domain classification for alignment
        source_domain_pred = self.domain_discriminator(source_shared)
        target_domain_pred = self.domain_discriminator(target_shared)
        
        return {
            'target_predictions': target_predictions,
            'source_shared': source_shared,
            'target_shared': target_shared,
            'source_domain_pred': source_domain_pred,
            'target_domain_pred': target_domain_pred
        }
    
    def compute_domain_alignment_loss(self, source_shared: torch.Tensor,
                                    target_shared: torch.Tensor) -> torch.Tensor:
        """Compute domain alignment loss using Maximum Mean Discrepancy"""
        
        def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
            """Gaussian RBF kernel"""
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            y_norm = (y ** 2).sum(dim=1, keepdim=True)
            
            dist = x_norm + y_norm.T - 2 * torch.mm(x, y.T)
            return torch.exp(-dist / (2 * sigma ** 2))
        
        # MMD between source and target representations
        xx = gaussian_kernel(source_shared, source_shared).mean()
        yy = gaussian_kernel(target_shared, target_shared).mean()
        xy = gaussian_kernel(source_shared, target_shared).mean()
        
        mmd_loss = xx + yy - 2 * xy
        
        return mmd_loss

class MultiDomainAttentionNetwork(nn.Module):
    """Multi-Domain Attention Network for Cross-Domain Recommendations"""
    
    def __init__(self, domain_features: Dict[str, int], embedding_dim: int = 64,
                 attention_dim: int = 32, hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.domains = list(domain_features.keys())
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        
        # Domain-specific feature extractors
        self.domain_extractors = nn.ModuleDict()
        for domain, feature_dim in domain_features.items():
            extractor = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.domain_extractors[domain] = extractor
        
        # Cross-domain attention mechanism
        self.query_projection = nn.Linear(embedding_dim, attention_dim)
        self.key_projection = nn.Linear(embedding_dim, attention_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Shared representation network
        self.shared_network = self._build_shared_network(embedding_dim, hidden_dims, dropout_rate)
        
        # Domain-specific output heads
        self.output_heads = nn.ModuleDict()
        for domain in self.domains:
            head = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[-1] // 2, 1),
                nn.Sigmoid()
            )
            self.output_heads[domain] = head
        
    def _build_shared_network(self, input_dim: int, hidden_dims: List[int],
                             dropout_rate: float) -> nn.Module:
        """Build shared representation network"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def compute_cross_domain_attention(self, query_domain: str, 
                                     domain_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-domain attention weights"""
        
        # Extract features for all domains
        extracted_features = {}
        for domain, features in domain_features.items():
            extracted_features[domain] = self.domain_extractors[domain](features)
        
        # Query from target domain
        query = self.query_projection(extracted_features[query_domain])  # (batch_size, attention_dim)
        
        # Keys and values from all domains
        keys = []
        values = []
        domain_names = []
        
        for domain, features in extracted_features.items():
            key = self.key_projection(features)  # (batch_size, attention_dim)
            value = self.value_projection(features)  # (batch_size, embedding_dim)
            
            keys.append(key)
            values.append(value)
            domain_names.append(domain)
        
        keys = torch.stack(keys, dim=1)  # (batch_size, n_domains, attention_dim)
        values = torch.stack(values, dim=1)  # (batch_size, n_domains, embedding_dim)
        
        # Compute attention scores
        attention_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))  # (batch_size, 1, n_domains)
        attention_weights = F.softmax(attention_scores / np.sqrt(self.attention_dim), dim=-1)
        
        # Apply attention to values
        attended_features = torch.bmm(attention_weights, values).squeeze(1)  # (batch_size, embedding_dim)
        
        return attended_features, attention_weights.squeeze(1)
    
    def forward(self, query_domain: str, domain_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with cross-domain attention"""
        
        # Compute cross-domain attention
        attended_features, attention_weights = self.compute_cross_domain_attention(
            query_domain, domain_features
        )
        
        # Pass through shared network
        shared_repr = self.shared_network(attended_features)
        
        # Generate domain-specific predictions
        predictions = self.output_heads[query_domain](shared_repr)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'shared_representation': shared_repr
        }
```

## 2. Advanced Meta-Learning Approaches

### Model-Agnostic Meta-Learning (MAML)

```python
class MAMLRecommender(nn.Module):
    """Model-Agnostic Meta-Learning for Recommendations"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 1, meta_lr: float = 0.01, 
                 inner_lr: float = 0.1, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        
        # Build network
        self.network = self._build_network(dropout_rate)
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=meta_lr)
        
    def _build_network(self, dropout_rate: float) -> nn.Module:
        """Build the recommendation network"""
        layers = []
        current_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional custom parameters"""
        
        if params is None:
            return self.network(x)
        
        # Use custom parameters
        x_current = x
        param_idx = 0
        
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                weight_key = f'network.{i}.weight'
                bias_key = f'network.{i}.bias'
                
                if weight_key in params and bias_key in params:
                    x_current = F.linear(x_current, params[weight_key], params[bias_key])
                else:
                    x_current = layer(x_current)
            else:
                x_current = layer(x_current)
        
        return x_current
    
    def inner_update(self, support_x: torch.Tensor, support_y: torch.Tensor,
                    num_steps: int = 1) -> Dict[str, torch.Tensor]:
        """Perform inner loop updates"""
        
        # Get current parameters
        params = dict(self.named_parameters())
        
        for step in range(num_steps):
            # Forward pass with current parameters
            predictions = self.forward(support_x, params)
            
            # Compute loss
            loss = F.mse_loss(predictions.squeeze(), support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            
            # Update parameters
            updated_params = {}
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - self.inner_lr * grad
            
            params = updated_params
        
        return params
    
    def meta_update(self, tasks_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """Perform meta-learning update across multiple tasks"""
        
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks_data:
            # Inner loop adaptation
            adapted_params = self.inner_update(support_x, support_y)
            
            # Evaluate on query set with adapted parameters
            query_predictions = self.forward(query_x, adapted_params)
            task_loss = F.mse_loss(query_predictions.squeeze(), query_y)
            
            meta_loss += task_loss
        
        meta_loss = meta_loss / len(tasks_data)
        
        # Meta-gradient step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_new_task(self, support_x: torch.Tensor, support_y: torch.Tensor,
                         num_steps: int = 5) -> Dict[str, torch.Tensor]:
        """Adapt to a new task using few-shot learning"""
        
        adapted_params = self.inner_update(support_x, support_y, num_steps)
        return adapted_params

class ReptileRecommender(nn.Module):
    """Reptile Meta-Learning for Recommendations"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 1, meta_lr: float = 0.01, 
                 inner_lr: float = 0.1, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        
        # Build network
        self.network = self._build_network(dropout_rate)
        
        # Store initial parameters for Reptile update
        self.initial_params = None
        
    def _build_network(self, dropout_rate: float) -> nn.Module:
        """Build the recommendation network"""
        layers = []
        current_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def inner_loop(self, task_data: Tuple[torch.Tensor, torch.Tensor], 
                  num_steps: int = 10) -> Dict[str, torch.Tensor]:
        """Reptile inner loop for a single task"""
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Create task-specific optimizer
        task_optimizer = torch.optim.SGD(self.parameters(), lr=self.inner_lr)
        
        x, y = task_data
        
        # Adapt to task
        for step in range(num_steps):
            task_optimizer.zero_grad()
            
            predictions = self.forward(x)
            loss = F.mse_loss(predictions.squeeze(), y)
            
            loss.backward()
            task_optimizer.step()
        
        # Get adapted parameters
        adapted_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Restore initial parameters
        for name, param in self.named_parameters():
            param.data.copy_(initial_params[name])
        
        return adapted_params
    
    def meta_update(self, tasks_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Reptile meta-update"""
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Collect adapted parameters from all tasks
        adapted_params_list = []
        
        for task_data in tasks_data:
            adapted_params = self.inner_loop(task_data)
            adapted_params_list.append(adapted_params)
        
        # Compute meta-gradient (average of parameter differences)
        meta_gradients = {}
        
        for name in initial_params.keys():
            # Average adapted parameters
            avg_adapted = torch.stack([params[name] for params in adapted_params_list]).mean(dim=0)
            
            # Meta-gradient is difference between adapted and initial
            meta_gradients[name] = avg_adapted - initial_params[name]
        
        # Apply meta-update
        for name, param in self.named_parameters():
            param.data.add_(meta_gradients[name], alpha=self.meta_lr)

class PrototypicalNetworkRecommender(nn.Module):
    """Prototypical Networks for Few-Shot Recommendation"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Embedding network
        self.embedding_network = self._build_embedding_network(hidden_dims, dropout_rate)
        
    def _build_embedding_network(self, hidden_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build embedding network"""
        layers = []
        current_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.embedding_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embeddings"""
        return self.embedding_network(x)
    
    def compute_prototypes(self, support_embeddings: torch.Tensor,
                          support_labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes"""
        
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            # Get embeddings for this class
            class_mask = (support_labels == label)
            class_embeddings = support_embeddings[class_mask]
            
            # Compute prototype (mean embedding)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def classify(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Classify query points using nearest prototype"""
        
        # Compute distances to all prototypes
        distances = torch.cdist(query_embeddings, prototypes)  # (n_queries, n_classes)
        
        # Convert distances to probabilities (negative log-softmax)
        log_probabilities = -F.log_softmax(distances, dim=1)
        
        return log_probabilities
    
    def episode_forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                       query_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for an episode (support + query)"""
        
        # Compute embeddings
        support_embeddings = self.forward(support_x)
        query_embeddings = self.forward(query_x)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_y)
        
        # Classify query points
        log_probabilities = self.classify(query_embeddings, prototypes)
        
        return log_probabilities
```

## 3. Multi-task Reinforcement Learning

### Multi-task Policy Networks

```python
class MultiTaskPolicyNetwork(nn.Module):
    """Multi-task Policy Network for Recommendation RL"""
    
    def __init__(self, state_dim: int, action_dims: Dict[str, int],
                 shared_dims: List[int] = [128, 64], task_dims: List[int] = [32],
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.task_names = list(action_dims.keys())
        
        # Shared feature extractor
        self.shared_network = self._build_shared_network(shared_dims, dropout_rate)
        
        # Task-specific policy heads
        self.policy_heads = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        
        for task_name, action_dim in action_dims.items():
            # Policy head (actor)
            policy_layers = []
            current_dim = shared_dims[-1]
            
            for task_dim in task_dims:
                policy_layers.extend([
                    nn.Linear(current_dim, task_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                current_dim = task_dim
            
            policy_layers.append(nn.Linear(current_dim, action_dim))
            policy_layers.append(nn.Softmax(dim=-1))
            
            self.policy_heads[task_name] = nn.Sequential(*policy_layers)
            
            # Value head (critic)
            value_layers = []
            current_dim = shared_dims[-1]
            
            for task_dim in task_dims:
                value_layers.extend([
                    nn.Linear(current_dim, task_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                current_dim = task_dim
            
            value_layers.append(nn.Linear(current_dim, 1))
            
            self.value_heads[task_name] = nn.Sequential(*value_layers)
    
    def _build_shared_network(self, shared_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build shared feature extractor"""
        layers = []
        current_dim = self.state_dim
        
        for shared_dim in shared_dims:
            layers.extend([
                nn.Linear(current_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = shared_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a specific task"""
        
        # Extract shared features
        shared_features = self.shared_network(state)
        
        # Task-specific policy and value
        policy = self.policy_heads[task_name](shared_features)
        value = self.value_heads[task_name](shared_features)
        
        return policy, value
    
    def get_action(self, state: torch.Tensor, task_name: str, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        
        policy, value = self.forward(state, task_name)
        
        if deterministic:
            action = torch.argmax(policy, dim=-1)
            log_prob = torch.log(policy.gather(1, action.unsqueeze(1))).squeeze(1)
        else:
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob

class MultiTaskDistilledQNetwork(nn.Module):
    """Multi-task Q-Network with Knowledge Distillation"""
    
    def __init__(self, state_dim: int, action_dims: Dict[str, int],
                 shared_dims: List[int] = [256, 128], task_dims: List[int] = [64],
                 distillation_weight: float = 0.5, dropout_rate: float = 0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.task_names = list(action_dims.keys())
        self.distillation_weight = distillation_weight
        
        # Shared feature extractor
        self.shared_network = self._build_shared_network(shared_dims, dropout_rate)
        
        # Task-specific Q-value heads
        self.q_heads = nn.ModuleDict()
        
        for task_name, action_dim in action_dims.items():
            q_layers = []
            current_dim = shared_dims[-1]
            
            for task_dim in task_dims:
                q_layers.extend([
                    nn.Linear(current_dim, task_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                current_dim = task_dim
            
            q_layers.append(nn.Linear(current_dim, action_dim))
            
            self.q_heads[task_name] = nn.Sequential(*q_layers)
        
        # Cross-task attention for knowledge sharing
        self.cross_task_attention = nn.MultiheadAttention(
            shared_dims[-1], num_heads=4, dropout=dropout_rate
        )
        
    def _build_shared_network(self, shared_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build shared feature extractor"""
        layers = []
        current_dim = self.state_dim
        
        for shared_dim in shared_dims:
            layers.extend([
                nn.Linear(current_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = shared_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, task_name: str = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        
        # Extract shared features
        shared_features = self.shared_network(state)
        
        if task_name is not None:
            # Single task forward
            q_values = self.q_heads[task_name](shared_features)
            return q_values
        else:
            # Multi-task forward
            all_q_values = {}
            
            # Apply cross-task attention
            batch_size = shared_features.size(0)
            feature_dim = shared_features.size(1)
            
            # Create task queries (one per task)
            task_queries = shared_features.unsqueeze(1).repeat(1, len(self.task_names), 1)
            task_keys = shared_features.unsqueeze(1).repeat(1, len(self.task_names), 1)
            task_values = shared_features.unsqueeze(1).repeat(1, len(self.task_names), 1)
            
            # Reshape for attention
            task_queries = task_queries.view(-1, feature_dim).unsqueeze(0)
            task_keys = task_keys.view(-1, feature_dim).unsqueeze(0)
            task_values = task_values.view(-1, feature_dim).unsqueeze(0)
            
            # Apply attention
            attended_features, _ = self.cross_task_attention(task_queries, task_keys, task_values)
            attended_features = attended_features.squeeze(0).view(batch_size, len(self.task_names), feature_dim)
            
            # Generate Q-values for each task
            for i, task_name in enumerate(self.task_names):
                task_features = attended_features[:, i, :]
                q_values = self.q_heads[task_name](task_features)
                all_q_values[task_name] = q_values
            
            return all_q_values
    
    def compute_distillation_loss(self, state: torch.Tensor, target_task: str,
                                 teacher_q_values: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        
        # Get student Q-values
        student_q_values = self.forward(state, target_task)
        
        # KL divergence loss between teacher and student
        teacher_probs = F.softmax(teacher_q_values / 3.0, dim=-1)  # Temperature scaling
        student_log_probs = F.log_softmax(student_q_values / 3.0, dim=-1)
        
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        return distillation_loss

class HierarchicalMultiTaskRL(nn.Module):
    """Hierarchical Multi-Task Reinforcement Learning"""
    
    def __init__(self, state_dim: int, meta_action_dim: int, 
                 task_action_dims: Dict[str, int], meta_dims: List[int] = [128, 64],
                 task_dims: List[int] = [64, 32], dropout_rate: float = 0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.meta_action_dim = meta_action_dim
        self.task_action_dims = task_action_dims
        self.task_names = list(task_action_dims.keys())
        
        # Meta-controller (selects which task to execute)
        self.meta_controller = self._build_meta_controller(meta_dims, dropout_rate)
        
        # Task-specific controllers
        self.task_controllers = nn.ModuleDict()
        for task_name, action_dim in task_action_dims.items():
            controller = self._build_task_controller(action_dim, task_dims, dropout_rate)
            self.task_controllers[task_name] = controller
        
        # Goal embedding network
        self.goal_embedding = nn.Sequential(
            nn.Linear(meta_action_dim, meta_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def _build_meta_controller(self, meta_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build meta-controller network"""
        layers = []
        current_dim = self.state_dim
        
        for meta_dim in meta_dims:
            layers.extend([
                nn.Linear(current_dim, meta_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = meta_dim
        
        # Output layer for meta-actions (task selection)
        layers.extend([
            nn.Linear(current_dim, self.meta_action_dim),
            nn.Softmax(dim=-1)
        ])
        
        return nn.Sequential(*layers)
    
    def _build_task_controller(self, action_dim: int, task_dims: List[int],
                              dropout_rate: float) -> nn.Module:
        """Build task-specific controller"""
        layers = []
        current_dim = self.state_dim + task_dims[0]  # State + goal embedding
        
        for task_dim in task_dims:
            layers.extend([
                nn.Linear(current_dim, task_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = task_dim
        
        # Output layer for task actions
        layers.extend([
            nn.Linear(current_dim, action_dim),
            nn.Softmax(dim=-1)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, 
                level: str = 'meta') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Hierarchical forward pass"""
        
        if level == 'meta':
            # Meta-controller: select task
            meta_policy = self.meta_controller(state)
            return meta_policy
        
        elif level == 'task':
            # First get meta-action (task selection)
            with torch.no_grad():
                meta_policy = self.meta_controller(state)
                meta_action = torch.argmax(meta_policy, dim=-1)
            
            # Get goal embedding
            goal_emb = self.goal_embedding(F.one_hot(meta_action, self.meta_action_dim).float())
            
            # Augment state with goal
            augmented_state = torch.cat([state, goal_emb], dim=-1)
            
            # Execute selected task
            task_policies = {}
            for i, task_name in enumerate(self.task_names):
                # Check if this task is selected
                task_mask = (meta_action == i).float().unsqueeze(-1)
                
                if task_mask.sum() > 0:  # Only compute if task is selected
                    task_policy = self.task_controllers[task_name](augmented_state)
                    task_policies[task_name] = task_policy * task_mask
            
            return meta_policy, task_policies
        
        else:
            raise ValueError(f"Unknown level: {level}")
    
    def get_hierarchical_action(self, state: torch.Tensor) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Get hierarchical action (meta + task)"""
        
        meta_policy, task_policies = self.forward(state, level='task')
        
        # Sample meta-action
        meta_dist = torch.distributions.Categorical(meta_policy)
        meta_action = meta_dist.sample()
        
        # Sample task actions
        task_actions = {}
        for task_name, task_policy in task_policies.items():
            if task_policy.sum() > 0:  # Only sample if task is active
                task_dist = torch.distributions.Categorical(task_policy)
                task_action = task_dist.sample()
                task_actions[task_name] = task_action
        
        return meta_action.item(), task_actions
```

## Key Takeaways (Part 2)

1. **Cross-Domain Transfer**: Domain adversarial training and attention mechanisms enable effective knowledge transfer across domains

2. **Advanced Meta-Learning**: MAML, Reptile, and Prototypical Networks provide sophisticated few-shot learning capabilities

3. **Multi-task RL**: Hierarchical and distilled approaches enable efficient multi-task reinforcement learning

4. **Knowledge Distillation**: Cross-task knowledge sharing improves performance on related tasks

5. **Adaptive Systems**: Dynamic task scheduling and meta-learning enable systems that adapt to new scenarios

6. **Hierarchical Learning**: Multi-level learning architectures handle complex decision-making processes

## Study Questions (Part 2)

### Beginner Level
1. How does domain adversarial training work in cross-domain recommendations?
2. What are the key differences between MAML and Reptile meta-learning?
3. How do hierarchical RL systems make decisions at multiple levels?
4. What is knowledge distillation in multi-task learning?

### Intermediate Level
1. Compare different cross-domain transfer mechanisms and their applications
2. How would you design a meta-learning system for cold-start recommendations?
3. What are the challenges in multi-task reinforcement learning for recommendations?
4. How can attention mechanisms improve cross-domain knowledge transfer?

### Advanced Level
1. Design a continual learning system that can add new tasks without forgetting
2. Implement a federated meta-learning approach for privacy-preserving recommendations
3. How would you handle catastrophic forgetting in multi-task neural networks?
4. Design a meta-learning system that can adapt to changing user preferences in real-time

## Next Session Preview

Tomorrow we'll explore **Advanced Embedding Techniques**, covering:
- Dynamic and contextual embeddings
- Graph-based embedding methods
- Multi-modal embedding fusion
- Embedding compression and quantization
- Privacy-preserving embeddings
- Real-time embedding updates and serving

We'll implement sophisticated embedding systems that handle complex, dynamic recommendation scenarios!