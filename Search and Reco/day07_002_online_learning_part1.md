# Day 7.2 Part 1: Online Learning Foundations for Recommendations

## Learning Objectives
- Master incremental learning algorithms for recommendation systems
- Implement online model updates with streaming data
- Design adaptive learning systems that handle concept drift
- Build real-time model training and inference pipelines
- Understand online optimization techniques for large-scale systems

## 1. Online Learning Fundamentals

### Incremental Learning Base Classes

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json
import pickle
from dataclasses import dataclass
from enum import Enum
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning modes for online algorithms"""
    BATCH = "batch"
    MINI_BATCH = "mini_batch"
    SINGLE_SAMPLE = "single_sample"
    ADAPTIVE = "adaptive"

@dataclass
class OnlineSample:
    """Sample for online learning"""
    user_id: str
    item_id: str
    features: np.ndarray
    target: float
    timestamp: datetime
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class OnlineLearner(ABC):
    """Abstract base class for online learning algorithms"""
    
    def __init__(self, learning_rate: float = 0.01, regularization: float = 0.001):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.model_version = 0
        self.samples_processed = 0
        self.last_update = datetime.now()
        self.performance_history = deque(maxlen=1000)
        
    @abstractmethod
    def partial_fit(self, sample: OnlineSample) -> float:
        """Fit model on a single sample"""
        pass
    
    @abstractmethod
    def predict(self, user_id: str, item_id: str, features: np.ndarray) -> float:
        """Make prediction for user-item pair"""
        pass
    
    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state"""
        pass
    
    @abstractmethod
    def set_model_state(self, state: Dict[str, Any]):
        """Set model state"""
        pass
    
    def batch_update(self, samples: List[OnlineSample]) -> List[float]:
        """Update model with batch of samples"""
        losses = []
        for sample in samples:
            loss = self.partial_fit(sample)
            losses.append(loss)
        return losses
    
    def evaluate_sample(self, sample: OnlineSample) -> float:
        """Evaluate model on a single sample"""
        prediction = self.predict(sample.user_id, sample.item_id, sample.features)
        error = (prediction - sample.target) ** 2
        return error
    
    def update_performance_history(self, error: float):
        """Update performance tracking"""
        self.performance_history.append({
            'error': error,
            'timestamp': datetime.now(),
            'samples_processed': self.samples_processed
        })

class OnlineMatrixFactorization(OnlineLearner):
    """Online Matrix Factorization using SGD"""
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 regularization: float = 0.001, user_bias_reg: float = 0.001,
                 item_bias_reg: float = 0.001):
        super().__init__(learning_rate, regularization)
        
        self.n_factors = n_factors
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg
        
        # Model parameters
        self.user_factors = {}  # Dict[user_id, np.ndarray]
        self.item_factors = {}  # Dict[item_id, np.ndarray]
        self.user_biases = {}   # Dict[user_id, float]
        self.item_biases = {}   # Dict[item_id, float]
        self.global_bias = 0.0
        
        # Statistics for initialization
        self.rating_sum = 0.0
        self.rating_count = 0
        
    def _init_user_factors(self, user_id: str):
        """Initialize user factors if not exists"""
        if user_id not in self.user_factors:
            self.user_factors[user_id] = np.random.normal(0, 0.1, self.n_factors)
            self.user_biases[user_id] = 0.0
    
    def _init_item_factors(self, item_id: str):
        """Initialize item factors if not exists"""
        if item_id not in self.item_factors:
            self.item_factors[item_id] = np.random.normal(0, 0.1, self.n_factors)
            self.item_biases[item_id] = 0.0
    
    def partial_fit(self, sample: OnlineSample) -> float:
        """Update model with single sample using SGD"""
        user_id = sample.user_id
        item_id = sample.item_id
        rating = sample.target
        
        # Initialize factors if needed
        self._init_user_factors(user_id)
        self._init_item_factors(item_id)
        
        # Update global bias
        self.rating_sum += rating
        self.rating_count += 1
        self.global_bias = self.rating_sum / self.rating_count
        
        # Get current factors
        user_factors = self.user_factors[user_id]
        item_factors = self.item_factors[item_id]
        user_bias = self.user_biases[user_id]
        item_bias = self.item_biases[item_id]
        
        # Compute prediction and error
        prediction = (self.global_bias + user_bias + item_bias + 
                     np.dot(user_factors, item_factors))
        error = rating - prediction
        
        # SGD updates
        # Update biases
        self.user_biases[user_id] += (self.learning_rate * 
                                     (error - self.user_bias_reg * user_bias))
        self.item_biases[item_id] += (self.learning_rate * 
                                     (error - self.item_bias_reg * item_bias))
        
        # Update factors
        user_factors_old = user_factors.copy()
        self.user_factors[user_id] += (self.learning_rate * 
                                      (error * item_factors - self.regularization * user_factors))
        self.item_factors[item_id] += (self.learning_rate * 
                                      (error * user_factors_old - self.regularization * item_factors))
        
        self.samples_processed += 1
        self.last_update = datetime.now()
        
        # Return squared error
        loss = error ** 2
        self.update_performance_history(loss)
        
        return loss
    
    def predict(self, user_id: str, item_id: str, features: np.ndarray = None) -> float:
        """Predict rating for user-item pair"""
        # Initialize if needed
        self._init_user_factors(user_id)
        self._init_item_factors(item_id)
        
        user_factors = self.user_factors[user_id]
        item_factors = self.item_factors[item_id]
        user_bias = self.user_biases[user_id]
        item_bias = self.item_biases[item_id]
        
        prediction = (self.global_bias + user_bias + item_bias + 
                     np.dot(user_factors, item_factors))
        
        return prediction
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for serialization"""
        return {
            'user_factors': {uid: factors.tolist() for uid, factors in self.user_factors.items()},
            'item_factors': {iid: factors.tolist() for iid, factors in self.item_factors.items()},
            'user_biases': self.user_biases.copy(),
            'item_biases': self.item_biases.copy(),
            'global_bias': self.global_bias,
            'model_version': self.model_version,
            'samples_processed': self.samples_processed,
            'rating_sum': self.rating_sum,
            'rating_count': self.rating_count
        }
    
    def set_model_state(self, state: Dict[str, Any]):
        """Set model state from serialized data"""
        self.user_factors = {uid: np.array(factors) for uid, factors in state['user_factors'].items()}
        self.item_factors = {iid: np.array(factors) for iid, factors in state['item_factors'].items()}
        self.user_biases = state['user_biases']
        self.item_biases = state['item_biases']
        self.global_bias = state['global_bias']
        self.model_version = state['model_version']
        self.samples_processed = state['samples_processed']
        self.rating_sum = state['rating_sum']
        self.rating_count = state['rating_count']

class OnlineFactorizationMachine(OnlineLearner):
    """Online Factorization Machine for recommendations"""
    
    def __init__(self, n_factors: int = 10, learning_rate: float = 0.01,
                 regularization: float = 0.001, feature_dim: int = None):
        super().__init__(learning_rate, regularization)
        
        self.n_factors = n_factors
        self.feature_dim = feature_dim
        
        # Model parameters
        self.w0 = 0.0  # Global bias
        self.w = None  # Linear weights
        self.V = None  # Factorization matrix
        
        # For adaptive feature dimension
        self.max_feature_seen = 0
        
    def _ensure_capacity(self, feature_dim: int):
        """Ensure model capacity for feature dimension"""
        if self.w is None or feature_dim > len(self.w):
            old_dim = len(self.w) if self.w is not None else 0
            
            # Expand linear weights
            new_w = np.zeros(feature_dim)
            if self.w is not None:
                new_w[:old_dim] = self.w
            self.w = new_w
            
            # Expand factorization matrix
            new_V = np.random.normal(0, 0.01, (feature_dim, self.n_factors))
            if self.V is not None:
                new_V[:old_dim] = self.V
            self.V = new_V
    
    def _predict_with_gradients(self, features: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """Predict with gradients for SGD update"""
        n_features = len(features)
        self._ensure_capacity(n_features)
        
        # Linear part
        linear_pred = self.w0 + np.dot(self.w[:n_features], features)
        
        # Interaction part
        interaction_pred = 0.0
        interaction_grads = np.zeros((n_features, self.n_factors))
        
        for f in range(self.n_factors):
            sum_vx = np.sum(self.V[:n_features, f] * features)
            sum_vx_squared = np.sum((self.V[:n_features, f] * features) ** 2)
            
            interaction_pred += 0.5 * (sum_vx ** 2 - sum_vx_squared)
            
            # Gradients for V
            for i in range(n_features):
                interaction_grads[i, f] = features[i] * sum_vx - self.V[i, f] * (features[i] ** 2)
        
        prediction = linear_pred + interaction_pred
        
        return prediction, interaction_grads, features, 1.0
    
    def partial_fit(self, sample: OnlineSample) -> float:
        """Update FM with single sample"""
        features = sample.features
        target = sample.target
        
        # Get prediction and gradients
        prediction, v_grads, x, w0_grad = self._predict_with_gradients(features)
        
        # Compute error
        error = target - prediction
        
        # SGD updates
        # Update global bias
        self.w0 += self.learning_rate * error * w0_grad
        
        # Update linear weights
        n_features = len(features)
        self.w[:n_features] += self.learning_rate * (error * features - self.regularization * self.w[:n_features])
        
        # Update factorization matrix
        self.V[:n_features] += self.learning_rate * (error * v_grads - self.regularization * self.V[:n_features])
        
        self.samples_processed += 1
        self.last_update = datetime.now()
        
        loss = error ** 2
        self.update_performance_history(loss)
        
        return loss
    
    def predict(self, user_id: str, item_id: str, features: np.ndarray) -> float:
        """Predict using FM"""
        if self.w is None:
            return 0.0
        
        prediction, _, _, _ = self._predict_with_gradients(features)
        return prediction
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get FM model state"""
        return {
            'w0': self.w0,
            'w': self.w.tolist() if self.w is not None else None,
            'V': self.V.tolist() if self.V is not None else None,
            'model_version': self.model_version,
            'samples_processed': self.samples_processed,
            'max_feature_seen': self.max_feature_seen
        }
    
    def set_model_state(self, state: Dict[str, Any]):
        """Set FM model state"""
        self.w0 = state['w0']
        self.w = np.array(state['w']) if state['w'] is not None else None
        self.V = np.array(state['V']) if state['V'] is not None else None
        self.model_version = state['model_version']
        self.samples_processed = state['samples_processed']
        self.max_feature_seen = state['max_feature_seen']

class OnlineNeuralCollaborativeFiltering(OnlineLearner):
    """Online Neural Collaborative Filtering"""
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001, regularization: float = 0.001,
                 user_vocab_size: int = 10000, item_vocab_size: int = 10000):
        super().__init__(learning_rate, regularization)
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        
        # User and item mappings
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.next_user_idx = 0
        self.next_item_idx = 0
        
        # Neural network model
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=regularization)
        self.criterion = nn.MSELoss()
        
    def _build_model(self) -> nn.Module:
        """Build neural collaborative filtering model"""
        
        class NCFModel(nn.Module):
            def __init__(self, user_vocab_size, item_vocab_size, embedding_dim, hidden_dims):
                super().__init__()
                
                self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
                self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)
                
                # MLP layers
                layers = []
                input_dim = embedding_dim * 2
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    input_dim = hidden_dim
                
                layers.append(nn.Linear(input_dim, 1))
                
                self.mlp = nn.Sequential(*layers)
                
                # Initialize embeddings
                nn.init.normal_(self.user_embedding.weight, std=0.01)
                nn.init.normal_(self.item_embedding.weight, std=0.01)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                
                # Concatenate embeddings
                combined = torch.cat([user_emb, item_emb], dim=1)
                
                # MLP prediction
                output = self.mlp(combined)
                
                return output.squeeze()
        
        return NCFModel(self.user_vocab_size, self.item_vocab_size, self.embedding_dim, self.hidden_dims)
    
    def _get_user_idx(self, user_id: str) -> int:
        """Get or create user index"""
        if user_id not in self.user_to_idx:
            if self.next_user_idx >= self.user_vocab_size:
                # Simple strategy: reuse oldest user slot
                # In practice, you'd want a more sophisticated strategy
                self.next_user_idx = 0
            
            self.user_to_idx[user_id] = self.next_user_idx
            self.next_user_idx += 1
        
        return self.user_to_idx[user_id]
    
    def _get_item_idx(self, item_id: str) -> int:
        """Get or create item index"""
        if item_id not in self.item_to_idx:
            if self.next_item_idx >= self.item_vocab_size:
                self.next_item_idx = 0
            
            self.item_to_idx[item_id] = self.next_item_idx
            self.next_item_idx += 1
        
        return self.item_to_idx[item_id]
    
    def partial_fit(self, sample: OnlineSample) -> float:
        """Update neural model with single sample"""
        user_idx = self._get_user_idx(sample.user_id)
        item_idx = self._get_item_idx(sample.item_id)
        
        # Convert to tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_tensor = torch.tensor([item_idx], dtype=torch.long)
        target_tensor = torch.tensor([sample.target], dtype=torch.float32)
        
        # Forward pass
        self.model.train()
        prediction = self.model(user_tensor, item_tensor)
        
        # Compute loss
        loss = self.criterion(prediction, target_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.samples_processed += 1
        self.last_update = datetime.now()
        
        loss_value = loss.item()
        self.update_performance_history(loss_value)
        
        return loss_value
    
    def predict(self, user_id: str, item_id: str, features: np.ndarray = None) -> float:
        """Predict using neural model"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0  # Cold start
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_tensor = torch.tensor([item_idx], dtype=torch.long)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(user_tensor, item_tensor)
        
        return prediction.item()
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get neural model state"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'next_user_idx': self.next_user_idx,
            'next_item_idx': self.next_item_idx,
            'model_version': self.model_version,
            'samples_processed': self.samples_processed
        }
    
    def set_model_state(self, state: Dict[str, Any]):
        """Set neural model state"""
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.user_to_idx = state['user_to_idx']
        self.item_to_idx = state['item_to_idx']
        self.next_user_idx = state['next_user_idx']
        self.next_item_idx = state['next_item_idx']
        self.model_version = state['model_version']
        self.samples_processed = state['samples_processed']
```

## 2. Adaptive Learning Systems

### Concept Drift Detection and Adaptation

```python
class ConceptDriftDetector:
    """Detects concept drift in online learning systems"""
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.05,
                 min_samples: int = 100):
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        
        self.error_window = deque(maxlen=window_size)
        self.baseline_error = None
        self.drift_detected = False
        self.last_drift_time = None
        
    def add_error(self, error: float) -> bool:
        """Add error and check for drift"""
        self.error_window.append(error)
        
        if len(self.error_window) < self.min_samples:
            return False
        
        # Initialize baseline if not set
        if self.baseline_error is None:
            self.baseline_error = np.mean(list(self.error_window)[:self.min_samples])
            return False
        
        # Check for drift using statistical test
        recent_errors = list(self.error_window)[-self.min_samples:]
        recent_mean = np.mean(recent_errors)
        
        # Simple drift detection: significant increase in error
        if recent_mean > self.baseline_error * (1 + self.threshold):
            if not self.drift_detected:
                self.drift_detected = True
                self.last_drift_time = datetime.now()
                logger.info(f"Concept drift detected! Baseline: {self.baseline_error:.4f}, Recent: {recent_mean:.4f}")
                return True
        else:
            self.drift_detected = False
        
        return False
    
    def reset_baseline(self):
        """Reset baseline after adaptation"""
        if len(self.error_window) >= self.min_samples:
            self.baseline_error = np.mean(list(self.error_window)[-self.min_samples:])
            self.drift_detected = False
            logger.info(f"Baseline reset to: {self.baseline_error:.4f}")

class AdaptiveLearningRate:
    """Adaptive learning rate scheduler for online learning"""
    
    def __init__(self, initial_lr: float = 0.01, decay_factor: float = 0.95,
                 min_lr: float = 0.001, patience: int = 100):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.patience = patience
        
        self.best_error = float('inf')
        self.no_improvement_count = 0
        self.error_history = deque(maxlen=patience)
        
    def update(self, error: float) -> float:
        """Update learning rate based on error"""
        self.error_history.append(error)
        
        if error < self.best_error:
            self.best_error = error
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Decay learning rate if no improvement
        if self.no_improvement_count >= self.patience:
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            self.no_improvement_count = 0
            logger.info(f"Learning rate decayed to: {self.current_lr:.6f}")
        
        return self.current_lr
    
    def reset(self):
        """Reset learning rate scheduler"""
        self.current_lr = self.initial_lr
        self.best_error = float('inf')
        self.no_improvement_count = 0
        self.error_history.clear()

class OnlineLearningManager:
    """Manages online learning with drift detection and adaptation"""
    
    def __init__(self, learner: OnlineLearner, learning_mode: LearningMode = LearningMode.SINGLE_SAMPLE,
                 batch_size: int = 32, enable_drift_detection: bool = True):
        self.learner = learner
        self.learning_mode = learning_mode
        self.batch_size = batch_size
        self.enable_drift_detection = enable_drift_detection
        
        # Adaptive components
        self.drift_detector = ConceptDriftDetector() if enable_drift_detection else None
        self.lr_scheduler = AdaptiveLearningRate()
        
        # Buffer for batch processing
        self.sample_buffer = []
        
        # Statistics
        self.total_samples = 0
        self.drift_count = 0
        self.last_adaptation_time = datetime.now()
        
        # Model checkpointing
        self.model_checkpoints = deque(maxlen=5)
        self.checkpoint_interval = 1000
        
    def process_sample(self, sample: OnlineSample) -> Dict[str, Any]:
        """Process a single sample"""
        
        if self.learning_mode == LearningMode.SINGLE_SAMPLE:
            return self._process_single_sample(sample)
        elif self.learning_mode == LearningMode.MINI_BATCH:
            return self._process_mini_batch_sample(sample)
        elif self.learning_mode == LearningMode.ADAPTIVE:
            return self._process_adaptive_sample(sample)
        else:
            return self._process_batch_sample(sample)
    
    def _process_single_sample(self, sample: OnlineSample) -> Dict[str, Any]:
        """Process sample immediately"""
        
        # Update model
        loss = self.learner.partial_fit(sample)
        
        # Update learning rate
        new_lr = self.lr_scheduler.update(loss)
        self.learner.learning_rate = new_lr
        
        # Check for drift
        drift_detected = False
        if self.drift_detector:
            drift_detected = self.drift_detector.add_error(loss)
            
            if drift_detected:
                self._handle_drift()
        
        self.total_samples += 1
        
        # Checkpoint if needed
        if self.total_samples % self.checkpoint_interval == 0:
            self._create_checkpoint()
        
        return {
            'loss': loss,
            'learning_rate': new_lr,
            'drift_detected': drift_detected,
            'samples_processed': self.total_samples
        }
    
    def _process_mini_batch_sample(self, sample: OnlineSample) -> Dict[str, Any]:
        """Process sample in mini-batches"""
        self.sample_buffer.append(sample)
        
        if len(self.sample_buffer) >= self.batch_size:
            # Process batch
            losses = self.learner.batch_update(self.sample_buffer)
            avg_loss = np.mean(losses)
            
            # Update learning rate
            new_lr = self.lr_scheduler.update(avg_loss)
            self.learner.learning_rate = new_lr
            
            # Check for drift
            drift_detected = False
            if self.drift_detector:
                drift_detected = self.drift_detector.add_error(avg_loss)
                
                if drift_detected:
                    self._handle_drift()
            
            self.total_samples += len(self.sample_buffer)
            batch_size = len(self.sample_buffer)
            self.sample_buffer.clear()
            
            return {
                'loss': avg_loss,
                'learning_rate': new_lr,
                'drift_detected': drift_detected,
                'batch_size': batch_size,
                'samples_processed': self.total_samples
            }
        
        return {'status': 'buffered', 'buffer_size': len(self.sample_buffer)}
    
    def _process_adaptive_sample(self, sample: OnlineSample) -> Dict[str, Any]:
        """Adaptively choose processing mode based on drift"""
        
        if self.drift_detector and self.drift_detector.drift_detected:
            # Use single sample updates during drift
            return self._process_single_sample(sample)
        else:
            # Use mini-batch updates during stable periods
            return self._process_mini_batch_sample(sample)
    
    def _handle_drift(self):
        """Handle detected concept drift"""
        logger.info("Handling concept drift...")
        
        self.drift_count += 1
        
        # Reset adaptive components
        self.lr_scheduler.reset()
        self.drift_detector.reset_baseline()
        
        # Increase learning rate temporarily for faster adaptation
        self.learner.learning_rate = min(self.learner.learning_rate * 2, 0.1)
        
        # Update model version
        self.learner.model_version += 1
        
        self.last_adaptation_time = datetime.now()
        
        logger.info(f"Drift handling completed. New model version: {self.learner.model_version}")
    
    def _create_checkpoint(self):
        """Create model checkpoint"""
        checkpoint = {
            'timestamp': datetime.now(),
            'model_state': self.learner.get_model_state(),
            'samples_processed': self.total_samples,
            'drift_count': self.drift_count
        }
        
        self.model_checkpoints.append(checkpoint)
        logger.info(f"Model checkpoint created at sample {self.total_samples}")
    
    def rollback_to_checkpoint(self, checkpoint_index: int = -1):
        """Rollback to a previous checkpoint"""
        if not self.model_checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False
        
        checkpoint = self.model_checkpoints[checkpoint_index]
        self.learner.set_model_state(checkpoint['model_state'])
        
        logger.info(f"Rolled back to checkpoint from {checkpoint['timestamp']}")
        return True
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        
        performance_history = list(self.learner.performance_history)
        recent_errors = [p['error'] for p in performance_history[-100:]]
        
        return {
            'total_samples': self.total_samples,
            'drift_count': self.drift_count,
            'current_learning_rate': self.learner.learning_rate,
            'model_version': self.learner.model_version,
            'recent_avg_error': np.mean(recent_errors) if recent_errors else 0,
            'drift_detected': self.drift_detector.drift_detected if self.drift_detector else False,
            'checkpoints_available': len(self.model_checkpoints),
            'last_adaptation_time': self.last_adaptation_time.isoformat()
        }

# Example demonstration
def demonstrate_online_learning():
    """Demonstrate online learning with different algorithms"""
    
    print("🚀 Online Learning Demonstration")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_users = 100
    n_items = 200
    n_samples = 5000
    
    # Create samples
    samples = []
    for i in range(n_samples):
        user_id = f"user_{np.random.randint(0, n_users)}"
        item_id = f"item_{np.random.randint(0, n_items)}"
        
        # Synthetic features
        features = np.random.randn(10)
        
        # Synthetic rating with some pattern
        user_factor = hash(user_id) % 1000 / 1000
        item_factor = hash(item_id) % 1000 / 1000
        noise = np.random.normal(0, 0.1)
        target = 3.0 + user_factor + item_factor + noise
        target = np.clip(target, 1, 5)
        
        # Add concept drift halfway through
        if i > n_samples // 2:
            target += 0.5  # Shift ratings up
        
        sample = OnlineSample(
            user_id=user_id,
            item_id=item_id,
            features=features,
            target=target,
            timestamp=datetime.now(),
            weight=1.0
        )
        samples.append(sample)
    
    # Test different algorithms
    algorithms = {
        'Matrix Factorization': OnlineMatrixFactorization(n_factors=10, learning_rate=0.01),
        'Factorization Machine': OnlineFactorizationMachine(n_factors=5, learning_rate=0.01),
        'Neural CF': OnlineNeuralCollaborativeFiltering(embedding_dim=32, hidden_dims=[64, 32], 
                                                       learning_rate=0.001)
    }
    
    results = {}
    
    for alg_name, learner in algorithms.items():
        print(f"\n📊 Testing {alg_name}...")
        
        # Create learning manager
        manager = OnlineLearningManager(
            learner=learner,
            learning_mode=LearningMode.ADAPTIVE,
            batch_size=16,
            enable_drift_detection=True
        )
        
        # Process samples
        losses = []
        drift_points = []
        
        for i, sample in enumerate(samples):
            result = manager.process_sample(sample)
            
            if 'loss' in result:
                losses.append(result['loss'])
            
            if result.get('drift_detected', False):
                drift_points.append(i)
            
            if i % 1000 == 0:
                stats = manager.get_learning_stats()
                print(f"  Sample {i}: Avg Loss = {np.mean(losses[-100:]):.4f}, "
                      f"LR = {stats['current_learning_rate']:.6f}, "
                      f"Drifts = {stats['drift_count']}")
        
        # Final statistics
        final_stats = manager.get_learning_stats()
        
        results[alg_name] = {
            'final_loss': np.mean(losses[-100:]) if losses else 0,
            'total_drifts': final_stats['drift_count'],
            'drift_points': drift_points,
            'samples_processed': final_stats['total_samples']
        }
        
        print(f"  ✅ Final Loss: {results[alg_name]['final_loss']:.4f}")
        print(f"  ✅ Concept Drifts Detected: {results[alg_name]['total_drifts']}")
    
    # Summary
    print(f"\n🏆 Online Learning Results Summary:")
    print("-" * 50)
    
    for alg_name, result in results.items():
        print(f"{alg_name}:")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Drifts Detected: {result['total_drifts']}")
        print(f"  Samples Processed: {result['samples_processed']}")
    
    print("\n✅ Online learning demonstration completed!")

if __name__ == "__main__":
    demonstrate_online_learning()
```

## Key Takeaways (Part 1)

1. **Incremental Learning**: Online algorithms update models with streaming data without retraining from scratch

2. **Concept Drift Detection**: Statistical methods detect when data distribution changes, triggering model adaptation

3. **Adaptive Learning Rates**: Dynamic learning rate adjustment improves convergence and handles concept drift

4. **Multiple Algorithms**: Different online algorithms (MF, FM, Neural) suit different recommendation scenarios

5. **Checkpointing**: Model versioning and rollback capabilities ensure system reliability

6. **Learning Modes**: Adaptive processing modes balance between latency and accuracy based on system state

## Study Questions (Part 1)

### Beginner Level
1. What are the advantages of online learning over batch learning?
2. How does concept drift affect recommendation systems?
3. What is the role of learning rate scheduling in online learning?
4. How do you handle cold start problems in online learning?

### Intermediate Level
1. Compare different online learning algorithms for recommendations
2. How would you implement ensemble methods for online learning?
3. What are the challenges in maintaining model performance during concept drift?
4. How can you balance exploration vs exploitation in online learning?

### Advanced Level
1. Design an online learning system that handles multiple types of concept drift
2. Implement a federated online learning approach for recommendations
3. How would you handle adversarial attacks in online learning systems?
4. Design an online learning system with formal convergence guarantees

## Next Part Preview

In **Part 2**, we'll cover:
- Model versioning and A/B testing in production
- Continuous integration/deployment for ML models
- Real-time model serving and inference optimization
- Multi-armed bandit approaches for online experimentation
- Advanced online optimization techniques
- Production deployment strategies for online learning systems

The second part will focus on productionizing online learning systems!