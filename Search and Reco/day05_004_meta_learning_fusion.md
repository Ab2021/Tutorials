# Day 5.4: Meta-Learning for Recommendation Fusion

## Learning Objectives
By the end of this session, you will:
- Understand meta-learning architectures for recommendation system fusion
- Implement neural approaches to hybrid recommendation systems
- Apply transfer learning across recommendation domains
- Build automated hyperparameter optimization for hybrid systems
- Master learning-to-learn approaches for recommendation combination

## 1. Meta-Learning Architectures

### Neural Meta-Learning for Recommendation Fusion

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import warnings
warnings.filterwarnings('ignore')

class MetaRecommenderFusion(nn.Module):
    """
    Neural meta-learning architecture for recommendation fusion
    """
    
    def __init__(self, num_algorithms: int, feature_dim: int, hidden_dim: int = 128,
                 meta_hidden_dim: int = 64, output_dim: int = 1):
        super(MetaRecommenderFusion, self).__init__()
        
        self.num_algorithms = num_algorithms
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.meta_hidden_dim = meta_hidden_dim
        
        # Algorithm-specific encoders
        self.algorithm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            ) for _ in range(num_algorithms)
        ])
        
        # Meta-network for learning combination strategy
        self.meta_network = nn.Sequential(
            nn.Linear(num_algorithms * (hidden_dim // 2) + feature_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(meta_hidden_dim, num_algorithms),  # Attention weights
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Algorithm confidence estimators
        self.confidence_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(num_algorithms)
        ])
        
    def forward(self, algorithm_features: torch.Tensor, context_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through meta-learning fusion network
        
        Args:
            algorithm_features: [batch_size, num_algorithms, feature_dim]
            context_features: [batch_size, feature_dim]
            
        Returns:
            Dictionary with predictions, weights, and confidences
        """
        batch_size = algorithm_features.size(0)
        
        # Encode algorithm-specific features
        algorithm_encodings = []
        algorithm_confidences = []
        
        for i in range(self.num_algorithms):
            # Get features for algorithm i
            algo_features = algorithm_features[:, i, :]  # [batch_size, feature_dim]
            
            # Encode features
            encoded = self.algorithm_encoders[i](algo_features)  # [batch_size, hidden_dim//2]
            algorithm_encodings.append(encoded)
            
            # Estimate confidence
            confidence = self.confidence_estimators[i](encoded)  # [batch_size, 1]
            algorithm_confidences.append(confidence)
        
        # Stack encodings
        stacked_encodings = torch.stack(algorithm_encodings, dim=1)  # [batch_size, num_algorithms, hidden_dim//2]
        stacked_confidences = torch.stack(algorithm_confidences, dim=1)  # [batch_size, num_algorithms, 1]
        
        # Prepare input for meta-network
        flattened_encodings = stacked_encodings.view(batch_size, -1)  # [batch_size, num_algorithms * hidden_dim//2]
        meta_input = torch.cat([flattened_encodings, context_features], dim=1)
        
        # Compute attention weights via meta-network
        attention_weights = self.meta_network(meta_input)  # [batch_size, num_algorithms]
        
        # Apply confidence weighting
        confidence_weights = stacked_confidences.squeeze(-1)  # [batch_size, num_algorithms]
        combined_weights = attention_weights * confidence_weights
        
        # Normalize weights
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        # Weighted combination of algorithm encodings
        weighted_encoding = torch.sum(
            stacked_encodings * combined_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, hidden_dim//2]
        
        # Final prediction
        prediction = self.fusion_layer(weighted_encoding)  # [batch_size, output_dim]
        
        return {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence_weights': confidence_weights,
            'combined_weights': combined_weights,
            'algorithm_encodings': stacked_encodings
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, 
                    algorithm_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for meta-learning
        
        Args:
            outputs: Model outputs from forward pass
            targets: Ground truth targets [batch_size, output_dim]
            algorithm_targets: Individual algorithm targets [batch_size, num_algorithms]
            
        Returns:
            Dictionary of loss components
        """
        
        # Primary prediction loss (MSE)
        prediction_loss = F.mse_loss(outputs['prediction'], targets)
        
        # Regularization losses
        attention_entropy = -torch.sum(
            outputs['attention_weights'] * torch.log(outputs['attention_weights'] + 1e-8),
            dim=-1
        ).mean()
        
        # Encourage diversity in attention weights
        diversity_loss = -attention_entropy
        
        # Confidence calibration loss (if algorithm targets available)
        confidence_loss = torch.tensor(0.0)
        if algorithm_targets is not None:
            # Simple confidence calibration: confidence should correlate with accuracy
            algorithm_errors = torch.abs(algorithm_targets - targets.expand_as(algorithm_targets))
            confidence_targets = 1.0 - (algorithm_errors / torch.max(algorithm_errors, dim=1, keepdim=True)[0])
            confidence_loss = F.mse_loss(
                outputs['confidence_weights'], 
                confidence_targets
            )
        
        # Total loss
        total_loss = (prediction_loss + 
                     0.01 * diversity_loss + 
                     0.1 * confidence_loss)
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'diversity_loss': diversity_loss,
            'confidence_loss': confidence_loss,
            'attention_entropy': attention_entropy
        }

class MAMLRecommenderFusion(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for recommendation fusion
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super(MAMLRecommenderFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Meta-learner network
        self.meta_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Store initial parameters
        self.initial_params = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through meta-network"""
        return self.meta_net(x)
    
    def clone_parameters(self) -> OrderedDict:
        """Clone current parameters"""
        return OrderedDict([(name, param.clone()) for name, param in self.named_parameters()])
    
    def load_parameters(self, params: OrderedDict):
        """Load parameters into model"""
        for name, param in self.named_parameters():
            if name in params:
                param.data.copy_(params[name])
    
    def inner_update(self, support_x: torch.Tensor, support_y: torch.Tensor, 
                    learning_rate: float = 0.01, num_steps: int = 5) -> OrderedDict:
        """
        Perform inner loop update for MAML
        
        Args:
            support_x: Support set features [support_size, feature_dim]
            support_y: Support set targets [support_size, 1]
            learning_rate: Inner loop learning rate
            num_steps: Number of inner loop steps
            
        Returns:
            Updated parameters
        """
        # Clone current parameters
        fast_params = self.clone_parameters()
        
        for step in range(num_steps):
            # Set fast parameters
            self.load_parameters(fast_params)
            
            # Forward pass
            predictions = self.forward(support_x)
            loss = F.mse_loss(predictions, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, self.parameters(), 
                create_graph=True, retain_graph=True
            )
            
            # Update fast parameters
            updated_params = OrderedDict()
            for (name, param), grad in zip(self.named_parameters(), grads):
                updated_params[name] = param - learning_rate * grad
            
            fast_params = updated_params
        
        return fast_params
    
    def meta_update(self, task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], 
                   meta_learning_rate: float = 0.001) -> float:
        """
        Perform meta-update across multiple tasks
        
        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) tuples
            meta_learning_rate: Meta-learning rate
            
        Returns:
            Average meta-loss
        """
        meta_losses = []
        meta_gradients = []
        
        for support_x, support_y, query_x, query_y in task_batch:
            # Store original parameters
            original_params = self.clone_parameters()
            
            # Inner loop update
            fast_params = self.inner_update(support_x, support_y)
            
            # Load fast parameters
            self.load_parameters(fast_params)
            
            # Compute meta-loss on query set
            query_predictions = self.forward(query_x)
            meta_loss = F.mse_loss(query_predictions, query_y)
            meta_losses.append(meta_loss.item())
            
            # Compute meta-gradients
            meta_grads = torch.autograd.grad(
                meta_loss, self.parameters(),
                retain_graph=False
            )
            meta_gradients.append(meta_grads)
            
            # Restore original parameters
            self.load_parameters(original_params)
        
        # Average meta-gradients and update parameters
        avg_meta_gradients = []
        for i in range(len(list(self.parameters()))):
            avg_grad = torch.stack([grads[i] for grads in meta_gradients]).mean(dim=0)
            avg_meta_gradients.append(avg_grad)
        
        # Apply meta-update
        with torch.no_grad():
            for param, meta_grad in zip(self.parameters(), avg_meta_gradients):
                param -= meta_learning_rate * meta_grad
        
        return np.mean(meta_losses)

class HyperNetworkFusion(nn.Module):
    """
    HyperNetwork approach for dynamic recommendation fusion
    """
    
    def __init__(self, num_algorithms: int, context_dim: int, 
                 target_net_dim: int = 64, hyper_hidden_dim: int = 128):
        super(HyperNetworkFusion, self).__init__()
        
        self.num_algorithms = num_algorithms
        self.context_dim = context_dim
        self.target_net_dim = target_net_dim
        
        # HyperNetwork that generates fusion network parameters
        self.hyper_network = nn.Sequential(
            nn.Linear(context_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, hyper_hidden_dim),
            nn.ReLU()
        )
        
        # Generate weights for target fusion network
        self.weight_generators = nn.ModuleDict({
            'layer1_weight': nn.Linear(hyper_hidden_dim, num_algorithms * target_net_dim),
            'layer1_bias': nn.Linear(hyper_hidden_dim, target_net_dim),
            'layer2_weight': nn.Linear(hyper_hidden_dim, target_net_dim * 1),
            'layer2_bias': nn.Linear(hyper_hidden_dim, 1)
        })
        
        self.target_net_dim = target_net_dim
        
    def forward(self, algorithm_predictions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HyperNetwork fusion
        
        Args:
            algorithm_predictions: [batch_size, num_algorithms]
            context: [batch_size, context_dim]
            
        Returns:
            Fused predictions [batch_size, 1]
        """
        batch_size = algorithm_predictions.size(0)
        
        # Generate target network parameters
        hyper_features = self.hyper_network(context)  # [batch_size, hyper_hidden_dim]
        
        # Generate weights and biases for target network
        layer1_weight = self.weight_generators['layer1_weight'](hyper_features)
        layer1_bias = self.weight_generators['layer1_bias'](hyper_features)
        layer2_weight = self.weight_generators['layer2_weight'](hyper_features)
        layer2_bias = self.weight_generators['layer2_bias'](hyper_features)
        
        # Reshape weights
        layer1_weight = layer1_weight.view(batch_size, self.target_net_dim, self.num_algorithms)
        layer2_weight = layer2_weight.view(batch_size, 1, self.target_net_dim)
        
        # Apply generated target network
        # Layer 1: [batch_size, target_net_dim] = [batch_size, target_net_dim, num_algorithms] @ [batch_size, num_algorithms, 1]
        hidden = torch.bmm(layer1_weight, algorithm_predictions.unsqueeze(-1)).squeeze(-1)
        hidden = hidden + layer1_bias
        hidden = F.relu(hidden)
        
        # Layer 2: [batch_size, 1] = [batch_size, 1, target_net_dim] @ [batch_size, target_net_dim, 1]
        output = torch.bmm(layer2_weight, hidden.unsqueeze(-1)).squeeze(-1)
        output = output + layer2_bias
        
        return output

class MetaLearningTrainer:
    """
    Trainer for meta-learning recommendation fusion models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'meta_losses': []
        }
        
    def train_meta_fusion(self, train_data: List[Dict], val_data: List[Dict],
                         num_epochs: int = 100, learning_rate: float = 0.001,
                         batch_size: int = 32):
        """
        Train meta-learning fusion model
        
        Args:
            train_data: List of training examples
            val_data: List of validation examples
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            # Create batches
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                # Prepare batch data
                batch_data = self._prepare_batch(batch)
                
                # Forward pass
                optimizer.zero_grad()
                
                if isinstance(self.model, MetaRecommenderFusion):
                    outputs = self.model(
                        batch_data['algorithm_features'],
                        batch_data['context_features']
                    )
                    loss_dict = self.model.compute_loss(
                        outputs, 
                        batch_data['targets'],
                        batch_data.get('algorithm_targets')
                    )
                    loss = loss_dict['total_loss']
                    
                elif isinstance(self.model, HyperNetworkFusion):
                    predictions = self.model(
                        batch_data['algorithm_predictions'],
                        batch_data['context_features']
                    )
                    loss = F.mse_loss(predictions, batch_data['targets'])
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i + batch_size]
                    batch_data = self._prepare_batch(batch)
                    
                    if isinstance(self.model, MetaRecommenderFusion):
                        outputs = self.model(
                            batch_data['algorithm_features'],
                            batch_data['context_features']
                        )
                        loss_dict = self.model.compute_loss(
                            outputs, 
                            batch_data['targets']
                        )
                        loss = loss_dict['total_loss']
                        
                    elif isinstance(self.model, HyperNetworkFusion):
                        predictions = self.model(
                            batch_data['algorithm_predictions'],
                            batch_data['context_features']
                        )
                        loss = F.mse_loss(predictions, batch_data['targets'])
                    
                    val_losses.append(loss.item())
            
            # Update learning rate
            avg_val_loss = np.mean(val_losses)
            scheduler.step(avg_val_loss)
            
            # Record history
            self.training_history['train_losses'].append(np.mean(train_losses))
            self.training_history['val_losses'].append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"Train Loss: {np.mean(train_losses):.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
    
    def train_maml(self, task_batches: List[List], num_meta_epochs: int = 100,
                  meta_learning_rate: float = 0.001):
        """
        Train MAML model
        
        Args:
            task_batches: List of task batches for meta-learning
            num_meta_epochs: Number of meta-training epochs
            meta_learning_rate: Meta-learning rate
        """
        
        if not isinstance(self.model, MAMLRecommenderFusion):
            raise ValueError("Model must be MAMLRecommenderFusion for MAML training")
        
        for epoch in range(num_meta_epochs):
            meta_losses = []
            
            for task_batch in task_batches:
                meta_loss = self.model.meta_update(task_batch, meta_learning_rate)
                meta_losses.append(meta_loss)
            
            avg_meta_loss = np.mean(meta_losses)
            self.training_history['meta_losses'].append(avg_meta_loss)
            
            if epoch % 10 == 0:
                print(f"Meta-Epoch {epoch}/{num_meta_epochs}: Meta Loss: {avg_meta_loss:.4f}")
    
    def _prepare_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare batch data for training"""
        
        batch_data = {}
        
        # Stack all data
        if 'algorithm_features' in batch[0]:
            algorithm_features = torch.stack([
                torch.tensor(item['algorithm_features'], dtype=torch.float32)
                for item in batch
            ]).to(self.device)
            batch_data['algorithm_features'] = algorithm_features
        
        if 'context_features' in batch[0]:
            context_features = torch.stack([
                torch.tensor(item['context_features'], dtype=torch.float32)
                for item in batch
            ]).to(self.device)
            batch_data['context_features'] = context_features
        
        if 'algorithm_predictions' in batch[0]:
            algorithm_predictions = torch.stack([
                torch.tensor(item['algorithm_predictions'], dtype=torch.float32)
                for item in batch
            ]).to(self.device)
            batch_data['algorithm_predictions'] = algorithm_predictions
        
        if 'targets' in batch[0]:
            targets = torch.stack([
                torch.tensor([item['targets']], dtype=torch.float32)
                for item in batch
            ]).to(self.device)
            batch_data['targets'] = targets
        
        if 'algorithm_targets' in batch[0]:
            algorithm_targets = torch.stack([
                torch.tensor(item['algorithm_targets'], dtype=torch.float32)
                for item in batch
            ]).to(self.device)
            batch_data['algorithm_targets'] = algorithm_targets
        
        return batch_data
```

## 2. Transfer Learning for Recommendation Domains

### Cross-Domain Meta-Learning

```python
class CrossDomainMetaLearner:
    """
    Meta-learning system for transfer across recommendation domains
    """
    
    def __init__(self, source_domains: List[str], target_domain: str):
        self.source_domains = source_domains
        self.target_domain = target_domain
        
        # Domain-specific models
        self.domain_models = {}
        self.domain_features = {}
        
        # Meta-model for domain adaptation
        self.meta_adapter = None
        
        # Domain similarity matrix
        self.domain_similarities = {}
        
        # Transfer learning strategies
        self.transfer_strategies = {
            'feature_alignment': self._feature_alignment_transfer,
            'model_agnostic': self._model_agnostic_transfer,
            'gradient_based': self._gradient_based_transfer,
            'prototype_based': self._prototype_based_transfer
        }
    
    def add_source_domain(self, domain_name: str, model: nn.Module, 
                         training_data: List[Dict]):
        """Add source domain with trained model and data"""
        
        self.domain_models[domain_name] = model
        
        # Extract domain-specific features
        domain_features = self._extract_domain_features(training_data)
        self.domain_features[domain_name] = domain_features
        
        # Update domain similarities
        self._update_domain_similarities(domain_name)
    
    def transfer_to_target(self, target_data: List[Dict], 
                          strategy: str = 'feature_alignment',
                          few_shot_size: int = 10) -> nn.Module:
        """
        Transfer knowledge from source domains to target domain
        
        Args:
            target_data: Limited target domain data
            strategy: Transfer learning strategy
            few_shot_size: Size of few-shot learning set
            
        Returns:
            Adapted model for target domain
        """
        
        if strategy not in self.transfer_strategies:
            raise ValueError(f"Unknown transfer strategy: {strategy}")
        
        return self.transfer_strategies[strategy](target_data, few_shot_size)
    
    def _feature_alignment_transfer(self, target_data: List[Dict], 
                                   few_shot_size: int) -> nn.Module:
        """Feature alignment-based transfer learning"""
        
        # Extract target domain features
        target_features = self._extract_domain_features(target_data[:few_shot_size])
        
        # Find most similar source domain
        most_similar_domain = self._find_most_similar_domain(target_features)
        
        if most_similar_domain is None:
            raise ValueError("No similar source domain found")
        
        # Clone source model
        source_model = self.domain_models[most_similar_domain]
        adapted_model = copy.deepcopy(source_model)
        
        # Feature alignment layer
        feature_aligner = FeatureAligner(
            source_features=self.domain_features[most_similar_domain],
            target_features=target_features
        )
        
        # Create adapted model with alignment
        adapted_model = AlignedModel(adapted_model, feature_aligner)
        
        # Fine-tune on target data
        self._fine_tune_model(adapted_model, target_data[:few_shot_size])
        
        return adapted_model
    
    def _model_agnostic_transfer(self, target_data: List[Dict], 
                                few_shot_size: int) -> nn.Module:
        """Model-agnostic meta-learning transfer"""
        
        # Use MAML approach
        maml_model = MAMLRecommenderFusion(
            feature_dim=len(target_data[0]['features']),
            hidden_dim=64
        )
        
        # Create meta-learning tasks from source domains
        meta_tasks = []
        for domain_name, domain_model in self.domain_models.items():
            # Create tasks from source domain
            domain_tasks = self._create_meta_tasks_from_domain(domain_name)
            meta_tasks.extend(domain_tasks)
        
        # Meta-train on source domains
        trainer = MetaLearningTrainer(maml_model)
        trainer.train_maml(meta_tasks, num_meta_epochs=50)
        
        # Adapt to target domain
        target_support = target_data[:few_shot_size//2]
        target_query = target_data[few_shot_size//2:few_shot_size]
        
        # Prepare data for MAML
        support_x = torch.tensor([item['features'] for item in target_support], dtype=torch.float32)
        support_y = torch.tensor([[item['rating']] for item in target_support], dtype=torch.float32)
        
        # Inner loop adaptation
        adapted_params = maml_model.inner_update(support_x, support_y, num_steps=10)
        maml_model.load_parameters(adapted_params)
        
        return maml_model
    
    def _gradient_based_transfer(self, target_data: List[Dict], 
                                few_shot_size: int) -> nn.Module:
        """Gradient-based transfer learning"""
        
        # Select best source models based on domain similarity
        target_features = self._extract_domain_features(target_data[:few_shot_size])
        
        # Compute gradients for each source model on target data
        gradient_similarities = {}
        
        for domain_name, source_model in self.domain_models.items():
            # Compute gradient similarity
            grad_sim = self._compute_gradient_similarity(
                source_model, target_data[:few_shot_size]
            )
            gradient_similarities[domain_name] = grad_sim
        
        # Select best source model
        best_domain = max(gradient_similarities.items(), key=lambda x: x[1])[0]
        adapted_model = copy.deepcopy(self.domain_models[best_domain])
        
        # Fine-tune with target-specific gradient updates
        self._gradient_based_fine_tuning(adapted_model, target_data[:few_shot_size])
        
        return adapted_model
    
    def _prototype_based_transfer(self, target_data: List[Dict], 
                                 few_shot_size: int) -> nn.Module:
        """Prototype-based transfer learning"""
        
        # Extract prototypes from source domains
        source_prototypes = {}
        for domain_name in self.source_domains:
            if domain_name in self.domain_features:
                prototypes = self._extract_prototypes(self.domain_features[domain_name])
                source_prototypes[domain_name] = prototypes
        
        # Extract target prototypes
        target_features = self._extract_domain_features(target_data[:few_shot_size])
        target_prototypes = self._extract_prototypes(target_features)
        
        # Create prototype-based model
        prototype_model = PrototypeBasedModel(
            source_prototypes=source_prototypes,
            target_prototypes=target_prototypes
        )
        
        # Train prototype matching
        self._train_prototype_matching(prototype_model, target_data[:few_shot_size])
        
        return prototype_model
    
    def _extract_domain_features(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract domain-specific statistical features"""
        
        features = {
            'rating_distribution': [],
            'feature_means': [],
            'feature_stds': [],
            'interaction_patterns': []
        }
        
        ratings = [item['rating'] for item in data if 'rating' in item]
        if ratings:
            features['rating_distribution'] = np.histogram(ratings, bins=5)[0]
            features['rating_mean'] = np.mean(ratings)
            features['rating_std'] = np.std(ratings)
        
        # Extract feature statistics
        if 'features' in data[0]:
            all_features = np.array([item['features'] for item in data])
            features['feature_means'] = np.mean(all_features, axis=0)
            features['feature_stds'] = np.std(all_features, axis=0)
        
        return features
    
    def _update_domain_similarities(self, new_domain: str):
        """Update domain similarity matrix"""
        
        new_domain_features = self.domain_features[new_domain]
        
        for existing_domain in self.domain_features:
            if existing_domain != new_domain:
                similarity = self._compute_domain_similarity(
                    new_domain_features,
                    self.domain_features[existing_domain]
                )
                self.domain_similarities[(new_domain, existing_domain)] = similarity
                self.domain_similarities[(existing_domain, new_domain)] = similarity
    
    def _compute_domain_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute similarity between two domains"""
        
        similarity_scores = []
        
        # Rating distribution similarity
        if 'rating_distribution' in features1 and 'rating_distribution' in features2:
            # Jensen-Shannon divergence
            p = features1['rating_distribution'] / np.sum(features1['rating_distribution'])
            q = features2['rating_distribution'] / np.sum(features2['rating_distribution'])
            
            m = (p + q) / 2
            js_div = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)
            js_similarity = 1.0 - np.sqrt(js_div)
            similarity_scores.append(js_similarity)
        
        # Feature distribution similarity
        if 'feature_means' in features1 and 'feature_means' in features2:
            mean_similarity = np.corrcoef(features1['feature_means'], features2['feature_means'])[0, 1]
            if not np.isnan(mean_similarity):
                similarity_scores.append(abs(mean_similarity))
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between distributions"""
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))
    
    def _find_most_similar_domain(self, target_features: Dict) -> Optional[str]:
        """Find most similar source domain to target"""
        
        best_domain = None
        best_similarity = -1.0
        
        for domain_name in self.source_domains:
            if domain_name in self.domain_features:
                similarity = self._compute_domain_similarity(
                    target_features,
                    self.domain_features[domain_name]
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_domain = domain_name
        
        return best_domain if best_similarity > 0.3 else None
    
    def _fine_tune_model(self, model: nn.Module, target_data: List[Dict]):
        """Fine-tune model on target data"""
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(20):  # Quick fine-tuning
            total_loss = 0.0
            
            for item in target_data:
                features = torch.tensor(item['features'], dtype=torch.float32).unsqueeze(0)
                target = torch.tensor([item['rating']], dtype=torch.float32).unsqueeze(0)
                
                optimizer.zero_grad()
                prediction = model(features)
                loss = F.mse_loss(prediction, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"Fine-tuning epoch {epoch}: Loss = {total_loss/len(target_data):.4f}")

class FeatureAligner(nn.Module):
    """Neural network for aligning features across domains"""
    
    def __init__(self, source_features: Dict, target_features: Dict):
        super(FeatureAligner, self).__init__()
        
        # Determine feature dimensions
        source_dim = len(source_features.get('feature_means', []))
        target_dim = len(target_features.get('feature_means', []))
        
        if source_dim > 0 and target_dim > 0:
            self.aligner = nn.Sequential(
                nn.Linear(source_dim, max(source_dim, target_dim)),
                nn.ReLU(),
                nn.Linear(max(source_dim, target_dim), target_dim)
            )
        else:
            self.aligner = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aligner(x)

class AlignedModel(nn.Module):
    """Model with feature alignment layer"""
    
    def __init__(self, base_model: nn.Module, feature_aligner: FeatureAligner):
        super(AlignedModel, self).__init__()
        self.feature_aligner = feature_aligner
        self.base_model = base_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned_features = self.feature_aligner(x)
        return self.base_model(aligned_features)

class PrototypeBasedModel(nn.Module):
    """Prototype-based model for few-shot transfer learning"""
    
    def __init__(self, source_prototypes: Dict, target_prototypes: Dict):
        super(PrototypeBasedModel, self).__init__()
        
        self.source_prototypes = source_prototypes
        self.target_prototypes = target_prototypes
        
        # Prototype matching network
        prototype_dim = 64  # Dimension of prototype embeddings
        self.prototype_encoder = nn.Sequential(
            nn.Linear(20, prototype_dim),  # Assuming 20 input features
            nn.ReLU(),
            nn.Linear(prototype_dim, prototype_dim)
        )
        
        self.prototype_matcher = nn.Sequential(
            nn.Linear(prototype_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode input to prototype space
        prototype_embedding = self.prototype_encoder(x)
        
        # Match against prototypes and predict
        prediction = self.prototype_matcher(prototype_embedding)
        
        return prediction
```

## 3. Study Questions

### Beginner Level

1. What is meta-learning and how does it apply to recommendation system fusion?
2. How does MAML (Model-Agnostic Meta-Learning) work for recommendation systems?
3. What are the advantages of using neural approaches for hybrid recommendation?
4. How can transfer learning help with cold start problems in new recommendation domains?
5. What is the role of HyperNetworks in dynamic recommendation fusion?

### Intermediate Level

6. Implement a meta-learning system that can quickly adapt to new user preferences with minimal data.
7. How would you design a transfer learning approach for recommendations across different content types (movies, books, music)?
8. Compare gradient-based meta-learning with prototype-based approaches for recommendation fusion.
9. What are the computational challenges of implementing meta-learning in production recommendation systems?
10. How would you measure domain similarity for effective transfer learning in recommendations?

### Advanced Level

11. Design a meta-learning architecture that can automatically discover optimal fusion strategies for different user segments.
12. Implement a few-shot learning system for recommendation that can adapt to new domains with less than 10 examples per user.
13. How would you create a meta-learning system that can handle both explicit and implicit feedback across domains?
14. Design a neural architecture that can learn to combine both collaborative and content-based signals meta-learning style.
15. Implement a continual learning system that can adapt to new recommendation domains without forgetting previous ones.

### Tricky Questions

16. How would you design a meta-learning system that can handle the heterogeneity of features across different recommendation domains?
17. What are the privacy implications of transfer learning in recommendation systems, and how would you address them?
18. How would you implement a meta-learning system that can work with both neural and non-neural base recommenders?
19. Design a system that can automatically determine when transfer learning will be beneficial versus training from scratch.
20. How would you create a meta-learning approach that can handle the temporal dynamics and concept drift in recommendation systems?

## Key Takeaways

1. **Meta-learning** enables automatic discovery of optimal fusion strategies across different contexts
2. **Neural approaches** provide flexible architectures for complex recommendation fusion scenarios
3. **Transfer learning** significantly reduces data requirements for new recommendation domains
4. **MAML and HyperNetworks** offer powerful frameworks for adaptive recommendation systems
5. **Few-shot learning** enables quick adaptation to new users and domains with minimal data
6. **Cross-domain knowledge transfer** can improve recommendation quality across related domains
7. **Automated fusion learning** reduces the need for manual hyperparameter tuning and architecture design

## Next Session Preview

In Day 5.5, we'll explore **Dynamic Hybridization Strategies**, covering:
- Real-time adaptation of hybrid strategies
- Context-driven dynamic fusion
- Online learning for hybrid optimization
- Temporal dynamics in recommendation fusion
- Adaptive ensemble methods