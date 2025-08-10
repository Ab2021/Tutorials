# Day 3.9: SVD and Advanced Factorization Techniques

## Learning Objectives
By the end of this session, you will:
- Master Singular Value Decomposition (SVD) and its applications in recommendations
- Understand advanced factorization techniques like SVD++, timeSVD++, and asymmetric SVD
- Implement non-negative matrix factorization (NMF) and probabilistic matrix factorization
- Learn about tensor factorization for multi-dimensional recommendation problems
- Handle temporal dynamics and side information in factorization models

## 1. Singular Value Decomposition (SVD) Fundamentals

### Mathematical Foundation
SVD decomposes any matrix M into three matrices:
```
M = U Σ V^T
```

Where:
- U: Left singular vectors (m×k orthogonal matrix)
- Σ: Diagonal matrix of singular values (k×k)
- V^T: Right singular vectors (k×n orthogonal matrix)
- k: Rank of the matrix (or truncated rank)

### SVD for Collaborative Filtering
For recommendation systems, we approximate the rating matrix R:
```
R ≈ U_k Σ_k V_k^T
```

Where k is the number of factors we keep (dimensionality reduction).

### Challenges with Traditional SVD
1. **Sparsity**: SVD requires complete matrices, but rating matrices are sparse
2. **Computational Cost**: O(mn²) complexity for m×n matrices
3. **Missing Values**: Traditional SVD cannot handle missing entries directly

## 2. Implementation: Advanced SVD Techniques

```python
import numpy as np
import pandas as pd
from scipy import sparse, linalg
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import time
from collections import defaultdict
import pickle

class SVDRecommender:
    """
    SVD-based recommendation system with various SVD implementations.
    """
    
    def __init__(self, n_factors: int = 50, algorithm: str = 'truncated_svd',
                 fill_method: str = 'mean', random_state: int = 42):
        """
        Initialize SVD recommender.
        
        Args:
            n_factors: Number of latent factors
            algorithm: 'truncated_svd', 'randomized_svd', or 'sparse_svd'
            fill_method: How to fill missing values ('mean', 'zero', 'median')
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.algorithm = algorithm
        self.fill_method = fill_method
        self.random_state = random_state
        
        # SVD components
        self.U = None
        self.sigma = None
        self.Vt = None
        
        # Data structures
        self.rating_matrix = None
        self.filled_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        self.global_mean = 0.0
        
        # Model
        self.svd_model = None
        
    def fit(self, interactions: List[Tuple], verbose: bool = True):
        """
        Fit SVD model to interaction data.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Fitting SVD model with {self.algorithm}...")
        
        start_time = time.time()
        
        # Create mappings and rating matrix
        self._create_mappings(interactions)
        self._create_rating_matrix(interactions)
        
        # Fill missing values
        self._fill_missing_values()
        
        # Apply SVD
        self._apply_svd()
        
        fit_time = time.time() - start_time
        
        if verbose:
            print(f"SVD fitting completed in {fit_time:.2f} seconds")
            print(f"Matrix shape: {self.rating_matrix.shape}")
            print(f"Density: {np.count_nonzero(self.rating_matrix) / self.rating_matrix.size:.4f}")
            print(f"Factors retained: {self.n_factors}")
    
    def _create_mappings(self, interactions: List[Tuple]):
        """Create user and item ID mappings."""
        users = set()
        items = set()
        
        for user_id, item_id, _ in interactions:
            users.add(user_id)
            items.add(item_id)
        
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(users)
        self.n_items = len(items)
    
    def _create_rating_matrix(self, interactions: List[Tuple]):
        """Create dense rating matrix from interactions."""
        self.rating_matrix = np.zeros((self.n_users, self.n_items))
        
        ratings = []
        for user_id, item_id, rating in interactions:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            self.rating_matrix[user_idx, item_idx] = rating
            ratings.append(rating)
        
        self.global_mean = np.mean(ratings)
    
    def _fill_missing_values(self):
        """Fill missing values in rating matrix."""
        self.filled_matrix = self.rating_matrix.copy()
        
        if self.fill_method == 'mean':
            # Fill with global mean
            self.filled_matrix[self.filled_matrix == 0] = self.global_mean
        elif self.fill_method == 'zero':
            # Keep zeros as is
            pass
        elif self.fill_method == 'median':
            # Fill with global median
            non_zero_ratings = self.rating_matrix[self.rating_matrix > 0]
            global_median = np.median(non_zero_ratings) if len(non_zero_ratings) > 0 else 0
            self.filled_matrix[self.filled_matrix == 0] = global_median
        elif self.fill_method == 'user_mean':
            # Fill with user means
            for i in range(self.n_users):
                user_ratings = self.rating_matrix[i, :]
                user_mean = np.mean(user_ratings[user_ratings > 0])
                if np.isnan(user_mean):
                    user_mean = self.global_mean
                self.filled_matrix[i, self.filled_matrix[i, :] == 0] = user_mean
        elif self.fill_method == 'item_mean':
            # Fill with item means
            for j in range(self.n_items):
                item_ratings = self.rating_matrix[:, j]
                item_mean = np.mean(item_ratings[item_ratings > 0])
                if np.isnan(item_mean):
                    item_mean = self.global_mean
                self.filled_matrix[self.filled_matrix[:, j] == 0, j] = item_mean
    
    def _apply_svd(self):
        """Apply SVD decomposition."""
        if self.algorithm == 'truncated_svd':
            # Use scikit-learn's TruncatedSVD
            self.svd_model = TruncatedSVD(
                n_components=self.n_factors,
                random_state=self.random_state
            )
            
            # Fit and transform
            user_factors = self.svd_model.fit_transform(self.filled_matrix)
            
            # Get components
            self.U = user_factors
            self.sigma = np.diag(self.svd_model.singular_values_)
            self.Vt = self.svd_model.components_
            
        elif self.algorithm == 'randomized_svd':
            # Use randomized SVD for efficiency
            self.U, singular_values, self.Vt = randomized_svd(
                self.filled_matrix,
                n_components=self.n_factors,
                random_state=self.random_state
            )
            self.sigma = np.diag(singular_values)
            
        elif self.algorithm == 'sparse_svd':
            # Convert to sparse matrix and use sparse SVD
            sparse_matrix = sparse.csr_matrix(self.filled_matrix)
            self.U, singular_values, self.Vt = svds(
                sparse_matrix,
                k=min(self.n_factors, min(self.n_users, self.n_items) - 1)
            )
            
            # svds returns in ascending order, reverse for descending
            self.U = self.U[:, ::-1]
            singular_values = singular_values[::-1]
            self.Vt = self.Vt[::-1, :]
            self.sigma = np.diag(singular_values)
            
        else:
            raise ValueError(f"Unknown SVD algorithm: {self.algorithm}")
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        if (user_id not in self.user_to_idx or 
            item_id not in self.item_to_idx):
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Reconstruct rating from SVD components
        prediction = np.dot(self.U[user_idx, :], 
                          np.dot(self.sigma, self.Vt[:, item_idx]))
        
        return float(prediction)
    
    def predict_all_ratings(self) -> np.ndarray:
        """Reconstruct the full rating matrix."""
        return np.dot(self.U, np.dot(self.sigma, self.Vt))
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       exclude_rated: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend items for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's predictions for all items
        user_vector = self.U[user_idx, :]
        predictions = np.dot(user_vector, np.dot(self.sigma, self.Vt))
        
        # Create recommendations
        recommendations = []
        for item_idx, prediction in enumerate(predictions):
            item_id = self.idx_to_item[item_idx]
            
            # Skip rated items if requested
            if exclude_rated and self.rating_matrix[user_idx, item_idx] > 0:
                continue
            
            recommendations.append((item_id, float(prediction)))
        
        # Sort by prediction (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_explained_variance_ratio(self) -> float:
        """Get the explained variance ratio of the SVD."""
        if hasattr(self.svd_model, 'explained_variance_ratio_'):
            return np.sum(self.svd_model.explained_variance_ratio_)
        else:
            # Calculate manually from singular values
            total_variance = np.sum(self.sigma.diagonal() ** 2)
            original_variance = np.sum((self.filled_matrix - self.global_mean) ** 2)
            return total_variance / original_variance if original_variance > 0 else 0
    
    def analyze_factors(self) -> Dict:
        """Analyze the learned factors."""
        analysis = {
            'singular_values': self.sigma.diagonal(),
            'user_factor_norms': np.linalg.norm(self.U, axis=1),
            'item_factor_norms': np.linalg.norm(self.Vt, axis=0),
            'explained_variance_ratio': self.get_explained_variance_ratio()
        }
        
        return analysis
    
    def visualize_factors(self, max_display: int = 20):
        """Visualize SVD factors."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Singular values
        axes[0, 0].plot(self.sigma.diagonal(), 'b-o')
        axes[0, 0].set_title('Singular Values')
        axes[0, 0].set_xlabel('Factor Index')
        axes[0, 0].set_ylabel('Singular Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # User factors (sample)
        user_sample = min(max_display, self.n_users)
        sns.heatmap(self.U[:user_sample, :], cmap='RdBu_r', center=0, ax=axes[0, 1])
        axes[0, 1].set_title(f'User Factors (first {user_sample})')
        axes[0, 1].set_xlabel('Factors')
        axes[0, 1].set_ylabel('Users')
        
        # Item factors (sample)
        item_sample = min(max_display, self.n_items)
        sns.heatmap(self.Vt[:, :item_sample], cmap='RdBu_r', center=0, ax=axes[1, 0])
        axes[1, 0].set_title(f'Item Factors (first {item_sample})')
        axes[1, 0].set_xlabel('Items')
        axes[1, 0].set_ylabel('Factors')
        
        # Reconstruction error by factor
        errors = []
        for k in range(1, min(self.n_factors + 1, 21)):
            # Partial reconstruction
            partial_reconstruction = np.dot(self.U[:, :k], 
                                          np.dot(self.sigma[:k, :k], self.Vt[:k, :]))
            
            # Calculate RMSE on observed ratings
            mask = self.rating_matrix > 0
            error = np.sqrt(np.mean((self.rating_matrix[mask] - partial_reconstruction[mask]) ** 2))
            errors.append(error)
        
        axes[1, 1].plot(range(1, len(errors) + 1), errors, 'r-o')
        axes[1, 1].set_title('Reconstruction Error vs Number of Factors')
        axes[1, 1].set_xlabel('Number of Factors')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class SVDPlusPlus:
    """
    SVD++ algorithm that incorporates implicit feedback.
    """
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.005,
                 regularization: float = 0.02, n_epochs: int = 100,
                 init_std: float = 0.1, random_state: int = 42):
        """
        Initialize SVD++ model.
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: Regularization parameter
            n_epochs: Number of training epochs
            init_std: Standard deviation for initialization
            random_state: Random seed
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.init_std = init_std
        self.random_state = random_state
        
        # Model parameters
        self.user_factors = None  # p_u
        self.item_factors = None  # q_i
        self.implicit_factors = None  # y_j
        self.user_biases = None  # b_u
        self.item_biases = None  # b_i
        self.global_mean = 0.0
        
        # Data structures
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        
        # User implicit feedback sets
        self.user_implicit_items = {}
        
        # Training history
        self.training_history = []
    
    def fit(self, interactions: List[Tuple], 
            implicit_feedback: Dict[str, List[str]] = None,
            verbose: bool = True):
        """
        Fit SVD++ model.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
            implicit_feedback: Dict mapping user_id to list of item_ids (implicit feedback)
            verbose: Whether to print progress
        """
        np.random.seed(self.random_state)
        
        if verbose:
            print("Fitting SVD++ model...")
        
        start_time = time.time()
        
        # Create mappings
        self._create_mappings(interactions)
        
        # Process implicit feedback
        self._process_implicit_feedback(interactions, implicit_feedback)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Convert interactions to training format
        train_data = []
        ratings = []
        for user_id, item_id, rating in interactions:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            train_data.append((user_idx, item_idx, rating))
            ratings.append(rating)
        
        self.global_mean = np.mean(ratings)
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            epoch_loss = 0.0
            for user_idx, item_idx, rating in train_data:
                # Make prediction
                prediction = self._predict_rating_idx(user_idx, item_idx)
                error = rating - prediction
                
                # Store old values for update
                user_factors_old = self.user_factors[user_idx, :].copy()
                item_factors_old = self.item_factors[item_idx, :].copy()
                
                # Compute sum of implicit factors for this user
                implicit_sum = self._get_user_implicit_sum(user_idx)
                
                # Update biases
                self.user_biases[user_idx] += self.learning_rate * (
                    error - self.regularization * self.user_biases[user_idx]
                )
                
                self.item_biases[item_idx] += self.learning_rate * (
                    error - self.regularization * self.item_biases[item_idx]
                )
                
                # Update factors
                self.user_factors[user_idx, :] += self.learning_rate * (
                    error * item_factors_old - 
                    self.regularization * user_factors_old
                )
                
                self.item_factors[item_idx, :] += self.learning_rate * (
                    error * (user_factors_old + implicit_sum) - 
                    self.regularization * item_factors_old
                )
                
                # Update implicit factors
                if user_idx in self.user_implicit_items:
                    n_implicit = len(self.user_implicit_items[user_idx])
                    if n_implicit > 0:
                        for implicit_item_idx in self.user_implicit_items[user_idx]:
                            self.implicit_factors[implicit_item_idx, :] += self.learning_rate * (
                                error * item_factors_old / np.sqrt(n_implicit) - 
                                self.regularization * self.implicit_factors[implicit_item_idx, :]
                            )
                
                epoch_loss += error ** 2
            
            # Calculate metrics
            epoch_time = time.time() - epoch_start
            train_rmse = np.sqrt(epoch_loss / len(train_data))
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'epoch_time': epoch_time
            })
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1:3d}: Train RMSE={train_rmse:.4f}, Time={epoch_time:.2f}s")
        
        fit_time = time.time() - start_time
        if verbose:
            print(f"SVD++ fitting completed in {fit_time:.2f} seconds")
    
    def _create_mappings(self, interactions: List[Tuple]):
        """Create user and item mappings."""
        users = set()
        items = set()
        
        for user_id, item_id, _ in interactions:
            users.add(user_id)
            items.add(item_id)
        
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(users)
        self.n_items = len(items)
    
    def _process_implicit_feedback(self, interactions: List[Tuple],
                                 implicit_feedback: Dict[str, List[str]] = None):
        """Process implicit feedback data."""
        # If no explicit implicit feedback provided, use all rated items
        if implicit_feedback is None:
            implicit_feedback = defaultdict(list)
            for user_id, item_id, _ in interactions:
                implicit_feedback[user_id].append(item_id)
        
        # Convert to index format
        self.user_implicit_items = {}
        for user_id, item_list in implicit_feedback.items():
            if user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
                item_indices = []
                for item_id in item_list:
                    if item_id in self.item_to_idx:
                        item_indices.append(self.item_to_idx[item_id])
                self.user_implicit_items[user_idx] = item_indices
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Initialize factors
        self.user_factors = np.random.normal(0, self.init_std, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, self.init_std, (self.n_items, self.n_factors))
        self.implicit_factors = np.random.normal(0, self.init_std, (self.n_items, self.n_factors))
        
        # Initialize biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
    
    def _get_user_implicit_sum(self, user_idx: int) -> np.ndarray:
        """Get normalized sum of implicit factors for a user."""
        if user_idx not in self.user_implicit_items:
            return np.zeros(self.n_factors)
        
        implicit_items = self.user_implicit_items[user_idx]
        if len(implicit_items) == 0:
            return np.zeros(self.n_factors)
        
        # Sum implicit factors and normalize by sqrt(|N(u)|)
        implicit_sum = np.sum(self.implicit_factors[implicit_items, :], axis=0)
        return implicit_sum / np.sqrt(len(implicit_items))
    
    def _predict_rating_idx(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using internal indices."""
        # Base prediction with biases
        prediction = (self.global_mean + 
                     self.user_biases[user_idx] + 
                     self.item_biases[item_idx])
        
        # Add factor interaction
        user_vector = self.user_factors[user_idx, :] + self._get_user_implicit_sum(user_idx)
        item_vector = self.item_factors[item_idx, :]
        
        prediction += np.dot(user_vector, item_vector)
        
        return prediction
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating for user-item pair."""
        if (user_id not in self.user_to_idx or 
            item_id not in self.item_to_idx):
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        return self._predict_rating_idx(user_idx, item_idx)
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       exclude_rated: bool = True) -> List[Tuple[str, float]]:
        """Recommend items for a user."""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get rated items to exclude
        exclude_set = set()
        if exclude_rated and user_idx in self.user_implicit_items:
            exclude_set = {self.idx_to_item[item_idx] 
                          for item_idx in self.user_implicit_items[user_idx]}
        
        # Generate predictions
        recommendations = []
        for item_id in self.item_to_idx.keys():
            if item_id not in exclude_set:
                prediction = self.predict_rating(user_id, item_id)
                recommendations.append((item_id, prediction))
        
        # Sort by prediction (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]

class NonNegativeMatrixFactorization:
    """
    Non-Negative Matrix Factorization for recommendation systems.
    """
    
    def __init__(self, n_factors: int = 50, max_iter: int = 200,
                 random_state: int = 42, alpha: float = 0.0, l1_ratio: float = 0.0):
        """
        Initialize NMF model.
        
        Args:
            n_factors: Number of latent factors
            max_iter: Maximum number of iterations
            random_state: Random seed
            alpha: Regularization strength
            l1_ratio: L1 vs L2 regularization ratio
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # Model
        self.nmf_model = None
        self.user_factors = None
        self.item_factors = None
        
        # Data structures
        self.rating_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        self.global_mean = 0.0
    
    def fit(self, interactions: List[Tuple], verbose: bool = True):
        """Fit NMF model."""
        if verbose:
            print("Fitting NMF model...")
        
        start_time = time.time()
        
        # Create rating matrix
        self._create_rating_matrix(interactions)
        
        # Ensure non-negative values (NMF requirement)
        # Shift ratings to be non-negative
        min_rating = np.min(self.rating_matrix[self.rating_matrix > 0])
        if min_rating < 0:
            self.rating_matrix[self.rating_matrix > 0] -= (min_rating - 0.1)
        
        # Replace zeros with small positive values for NMF
        filled_matrix = self.rating_matrix.copy()
        filled_matrix[filled_matrix == 0] = 0.1
        
        # Apply NMF
        self.nmf_model = NMF(
            n_components=self.n_factors,
            init='random',
            random_state=self.random_state,
            max_iter=self.max_iter,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio
        )
        
        # Fit and transform
        self.user_factors = self.nmf_model.fit_transform(filled_matrix)
        self.item_factors = self.nmf_model.components_.T
        
        fit_time = time.time() - start_time
        
        if verbose:
            print(f"NMF fitting completed in {fit_time:.2f} seconds")
            print(f"Reconstruction error: {self.nmf_model.reconstruction_err_:.4f}")
    
    def _create_rating_matrix(self, interactions: List[Tuple]):
        """Create rating matrix from interactions."""
        # Create mappings
        users = set()
        items = set()
        
        for user_id, item_id, _ in interactions:
            users.add(user_id)
            items.add(item_id)
        
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(users)
        self.n_items = len(items)
        
        # Create matrix
        self.rating_matrix = np.zeros((self.n_users, self.n_items))
        
        ratings = []
        for user_id, item_id, rating in interactions:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            self.rating_matrix[user_idx, item_idx] = rating
            ratings.append(rating)
        
        self.global_mean = np.mean(ratings)
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating for user-item pair."""
        if (user_id not in self.user_to_idx or 
            item_id not in self.item_to_idx):
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        prediction = np.dot(self.user_factors[user_idx, :], 
                          self.item_factors[item_idx, :])
        
        return float(prediction)
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       exclude_rated: bool = True) -> List[Tuple[str, float]]:
        """Recommend items for a user."""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get predictions for all items
        user_vector = self.user_factors[user_idx, :]
        predictions = np.dot(self.item_factors, user_vector)
        
        # Create recommendations
        recommendations = []
        for item_idx, prediction in enumerate(predictions):
            item_id = self.idx_to_item[item_idx]
            
            # Skip rated items if requested
            if exclude_rated and self.rating_matrix[user_idx, item_idx] > 0:
                continue
            
            recommendations.append((item_id, float(prediction)))
        
        # Sort by prediction (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_factor_interpretation(self, top_items_per_factor: int = 5) -> Dict:
        """
        Interpret NMF factors by finding top items for each factor.
        
        Args:
            top_items_per_factor: Number of top items to show per factor
            
        Returns:
            Dictionary with factor interpretations
        """
        interpretation = {}
        
        for factor_idx in range(self.n_factors):
            # Get item loadings for this factor
            factor_loadings = self.item_factors[:, factor_idx]
            
            # Find top items for this factor
            top_item_indices = np.argsort(factor_loadings)[::-1][:top_items_per_factor]
            
            top_items = []
            for item_idx in top_item_indices:
                item_id = self.idx_to_item[item_idx]
                loading = factor_loadings[item_idx]
                top_items.append((item_id, float(loading)))
            
            interpretation[f'Factor_{factor_idx}'] = {
                'top_items': top_items,
                'total_weight': float(np.sum(factor_loadings))
            }
        
        return interpretation

# Example usage and comprehensive testing
def create_advanced_test_data():
    """Create comprehensive test data for advanced factorization techniques."""
    np.random.seed(42)
    
    # Create users with different behavior patterns
    n_users = 500
    n_items = 300
    
    # Define user segments
    segments = {
        'casual': {'size': 150, 'avg_ratings': 8, 'rating_style': 'positive'},
        'critical': {'size': 100, 'avg_ratings': 15, 'rating_style': 'diverse'},
        'enthusiast': {'size': 150, 'avg_ratings': 25, 'rating_style': 'comprehensive'},
        'inactive': {'size': 100, 'avg_ratings': 3, 'rating_style': 'minimal'}
    }
    
    # Define item categories
    categories = {
        'popular': {'items': list(range(0, 50)), 'base_rating': 4.2},
        'niche': {'items': list(range(50, 100)), 'base_rating': 3.8},
        'classic': {'items': list(range(100, 150)), 'base_rating': 4.5},
        'new': {'items': list(range(150, 200)), 'base_rating': 3.5},
        'polarizing': {'items': list(range(200, 250)), 'base_rating': 3.0},
        'specialty': {'items': list(range(250, 300)), 'base_rating': 4.0}
    }
    
    interactions = []
    implicit_feedback = defaultdict(list)
    
    user_id = 0
    for segment_name, segment_info in segments.items():
        for _ in range(segment_info['size']):
            user_name = f'{segment_name}_user_{user_id}'
            
            # Determine number of ratings
            n_ratings = max(1, int(np.random.normal(
                segment_info['avg_ratings'], 
                segment_info['avg_ratings'] * 0.3
            )))
            
            # Select items to rate based on segment behavior
            if segment_name == 'casual':
                # Casual users mostly rate popular items
                candidate_items = (categories['popular']['items'] + 
                                 categories['new']['items'][:20])
            elif segment_name == 'critical':
                # Critical users rate diverse items
                candidate_items = []
                for cat_items in categories.values():
                    candidate_items.extend(cat_items['items'][:15])
            elif segment_name == 'enthusiast':
                # Enthusiasts rate everything
                candidate_items = list(range(n_items))
            else:  # inactive
                # Inactive users rate very few, mostly popular
                candidate_items = categories['popular']['items'][:20]
            
            # Select items to rate
            n_candidates = min(n_ratings, len(candidate_items))
            if n_candidates > 0:
                rated_items = np.random.choice(candidate_items, n_candidates, replace=False)
                
                for item_idx in rated_items:
                    # Determine base rating from item category
                    base_rating = 3.5  # default
                    for cat_name, cat_info in categories.items():
                        if item_idx in cat_info['items']:
                            base_rating = cat_info['base_rating']
                            break
                    
                    # Add user and segment bias
                    if segment_info['rating_style'] == 'positive':
                        rating_bias = np.random.normal(0.3, 0.5)
                    elif segment_info['rating_style'] == 'diverse':
                        rating_bias = np.random.normal(0, 0.8)
                    elif segment_info['rating_style'] == 'comprehensive':
                        rating_bias = np.random.normal(0.1, 0.6)
                    else:  # minimal
                        rating_bias = np.random.normal(0, 0.4)
                    
                    final_rating = np.clip(base_rating + rating_bias, 1, 5)
                    
                    interactions.append((user_name, f'item_{item_idx}', final_rating))
                    implicit_feedback[user_name].append(f'item_{item_idx}')
                
                # Add some implicit-only feedback (items viewed but not rated)
                if segment_name in ['enthusiast', 'critical']:
                    n_implicit = np.random.randint(5, 15)
                    implicit_items = np.random.choice(
                        [f'item_{i}' for i in range(n_items)], 
                        n_implicit, replace=False
                    )
                    for item in implicit_items:
                        if item not in implicit_feedback[user_name]:
                            implicit_feedback[user_name].append(item)
            
            user_id += 1
    
    return interactions, dict(implicit_feedback), segments, categories

if __name__ == "__main__":
    # Create comprehensive test data
    print("Creating advanced factorization test data...")
    interactions, implicit_feedback, segments, categories = create_advanced_test_data()
    
    print(f"Created {len(interactions)} explicit interactions")
    print(f"Created implicit feedback for {len(implicit_feedback)} users")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Test SVD Recommender
    print("\n=== Testing SVD Recommender ===")
    
    algorithms = ['truncated_svd', 'randomized_svd']
    fill_methods = ['mean', 'user_mean']
    
    svd_results = {}
    
    for algorithm in algorithms:
        for fill_method in fill_methods:
            print(f"\nTesting {algorithm} with {fill_method} filling...")
            
            svd_model = SVDRecommender(
                n_factors=30,
                algorithm=algorithm,
                fill_method=fill_method,
                random_state=42
            )
            
            start_time = time.time()
            svd_model.fit(train_data, verbose=False)
            fit_time = time.time() - start_time
            
            # Evaluate
            predictions = []
            actuals = []
            
            for user_id, item_id, rating in test_data[:1000]:  # Sample for speed
                pred = svd_model.predict_rating(user_id, item_id)
                predictions.append(pred)
                actuals.append(rating)
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            result_key = f"{algorithm}_{fill_method}"
            svd_results[result_key] = {
                'rmse': rmse,
                'mae': mae,
                'fit_time': fit_time,
                'explained_variance': svd_model.get_explained_variance_ratio()
            }
            
            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, Time: {fit_time:.2f}s")
            print(f"  Explained variance: {svd_model.get_explained_variance_ratio():.4f}")
    
    # Find and visualize best SVD model
    best_svd_key = min(svd_results.keys(), key=lambda x: svd_results[x]['rmse'])
    print(f"\nBest SVD configuration: {best_svd_key}")
    
    algorithm, fill_method = best_svd_key.split('_', 1)
    best_svd_model = SVDRecommender(
        n_factors=30,
        algorithm=algorithm,
        fill_method=fill_method,
        random_state=42
    )
    best_svd_model.fit(train_data, verbose=False)
    best_svd_model.visualize_factors()
    
    # Test SVD++
    print("\n=== Testing SVD++ ===")
    
    svdpp_model = SVDPlusPlus(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=50,
        random_state=42
    )
    
    start_time = time.time()
    svdpp_model.fit(train_data, implicit_feedback=implicit_feedback, verbose=True)
    svdpp_fit_time = time.time() - start_time
    
    # Evaluate SVD++
    svdpp_predictions = []
    svdpp_actuals = []
    
    for user_id, item_id, rating in test_data[:1000]:
        pred = svdpp_model.predict_rating(user_id, item_id)
        svdpp_predictions.append(pred)
        svdpp_actuals.append(rating)
    
    svdpp_rmse = np.sqrt(mean_squared_error(svdpp_actuals, svdpp_predictions))
    svdpp_mae = mean_absolute_error(svdpp_actuals, svdpp_predictions)
    
    print(f"SVD++ Results:")
    print(f"RMSE: {svdpp_rmse:.4f}, MAE: {svdpp_mae:.4f}, Time: {svdpp_fit_time:.2f}s")
    
    # Test NMF
    print("\n=== Testing Non-Negative Matrix Factorization ===")
    
    nmf_model = NonNegativeMatrixFactorization(
        n_factors=25,
        max_iter=100,
        alpha=0.01,
        random_state=42
    )
    
    start_time = time.time()
    nmf_model.fit(train_data, verbose=True)
    nmf_fit_time = time.time() - start_time
    
    # Evaluate NMF
    nmf_predictions = []
    nmf_actuals = []
    
    for user_id, item_id, rating in test_data[:1000]:
        pred = nmf_model.predict_rating(user_id, item_id)
        nmf_predictions.append(pred)
        nmf_actuals.append(rating)
    
    nmf_rmse = np.sqrt(mean_squared_error(nmf_actuals, nmf_predictions))
    nmf_mae = mean_absolute_error(nmf_actuals, nmf_predictions)
    
    print(f"NMF Results:")
    print(f"RMSE: {nmf_rmse:.4f}, MAE: {nmf_mae:.4f}, Time: {nmf_fit_time:.2f}s")
    
    # Show NMF factor interpretation
    factor_interpretation = nmf_model.get_factor_interpretation(top_items_per_factor=3)
    print("\nNMF Factor Interpretation (top 3 items per factor):")
    for factor_name, factor_info in list(factor_interpretation.items())[:5]:
        print(f"{factor_name}:")
        for item_id, loading in factor_info['top_items']:
            print(f"  {item_id}: {loading:.3f}")
    
    # Compare all methods
    print("\n=== Method Comparison ===")
    comparison_data = [
        ('Best SVD', svd_results[best_svd_key]['rmse'], svd_results[best_svd_key]['fit_time']),
        ('SVD++', svdpp_rmse, svdpp_fit_time),
        ('NMF', nmf_rmse, nmf_fit_time)
    ]
    
    print("Method        RMSE     Training Time")
    print("-" * 35)
    for method, rmse, time_taken in comparison_data:
        print(f"{method:12} {rmse:6.4f}   {time_taken:8.2f}s")
    
    # Sample recommendations from best model
    print(f"\n=== Sample Recommendations from {best_svd_key} ===")
    sample_user = 'casual_user_0'
    recommendations = best_svd_model.recommend_items(sample_user, n_recommendations=5)
    
    print(f"Top 5 recommendations for {sample_user}:")
    for item_id, pred_rating in recommendations:
        print(f"  {item_id}: {pred_rating:.2f}")
```

## 3. Advanced Factorization Techniques

### 3.1 Asymmetric SVD
Handles asymmetric relationships between users and items.

### 3.2 SVD with Temporal Dynamics (timeSVD++)
Incorporates time-evolving user preferences and item characteristics.

### 3.3 Weighted Matrix Factorization
Different weights for different observations based on confidence.

### 3.4 Constrained Matrix Factorization
Incorporates domain knowledge through constraints on factors.

## 4. Tensor Factorization

### 4.1 PARAFAC/CANDECOMP
Decomposes 3D tensors for multi-dimensional recommendations.

### 4.2 Tucker Decomposition
More flexible tensor decomposition for complex relationships.

### 4.3 Applications
- User × Item × Context recommendations
- Temporal recommendation systems
- Multi-criteria rating systems

## 5. Study Questions

### Basic Level
1. What are the main differences between classical SVD and matrix factorization for recommendations?
2. How does SVD++ improve upon basic SVD?
3. What makes NMF particularly suitable for certain types of recommendation problems?
4. When would you choose randomized SVD over truncated SVD?

### Intermediate Level
5. Implement a weighted SVD that gives different importance to different ratings.
6. Design a matrix factorization approach that handles both positive and negative feedback.
7. How would you incorporate item features into SVD-based recommendations?
8. Compare the computational complexity of different SVD algorithms.

### Advanced Level
9. Implement tensor factorization for context-aware recommendations.
10. Design a temporal matrix factorization model that handles concept drift.
11. How would you create an online SVD system that updates incrementally?
12. Implement a multi-task matrix factorization for multiple related recommendation problems.

### Tricky Questions
13. A user's preferences change dramatically over time. How would different SVD variants handle this?
14. You have both explicit ratings and implicit feedback. Design a unified SVD model for both.
15. How would you detect and handle adversarial attacks on SVD-based systems?
16. Design an SVD system that provides uncertainty estimates for its predictions.

## 6. Key Takeaways

1. **SVD provides powerful dimensionality reduction** for sparse rating matrices
2. **SVD++ incorporates implicit feedback** for better recommendations
3. **NMF offers interpretable factors** with non-negativity constraints
4. **Different SVD algorithms** suit different computational and accuracy requirements
5. **Advanced techniques handle temporal dynamics** and multiple data types
6. **Tensor factorization extends** to multi-dimensional recommendation scenarios

## Next Session Preview
In the next session, we'll explore the cold start problem - one of the biggest challenges in recommendation systems - and learn various strategies to handle new users and items effectively.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create day2_09_ranking_evaluation.md: IR Evaluation Metrics", "status": "completed", "priority": "high", "id": "create_day2_09_ranking_evaluation"}, {"content": "Create day3_001_useritem_introduction.md: Introduction to User-Item Matrices", "status": "completed", "priority": "high", "id": "create_day3_001"}, {"content": "Create day3_002_useritem_representation.md: User-Item Matrix Representations and Sparsity", "status": "completed", "priority": "high", "id": "create_day3_002"}, {"content": "Create day3_003_cf_fundamentals.md: Collaborative Filtering Fundamentals", "status": "completed", "priority": "high", "id": "create_day3_003"}, {"content": "Create day3_004_user_based_cf.md: User-Based Collaborative Filtering", "status": "completed", "priority": "high", "id": "create_day3_004"}, {"content": "Create day3_005_item_based_cf.md: Item-Based Collaborative Filtering", "status": "completed", "priority": "high", "id": "create_day3_005"}, {"content": "Create day3_006_similarity_measures.md: Similarity Measures and Distance Metrics", "status": "completed", "priority": "high", "id": "create_day3_006"}, {"content": "Create day3_007_neighborhood_selection.md: Neighborhood Selection Strategies", "status": "completed", "priority": "high", "id": "create_day3_007"}, {"content": "Create day3_008_matrix_factorization_intro.md: Introduction to Matrix Factorization", "status": "completed", "priority": "high", "id": "create_day3_008"}, {"content": "Create day3_009_svd_techniques.md: SVD and Advanced Factorization Techniques", "status": "completed", "priority": "high", "id": "create_day3_009"}, {"content": "Create day3_010_cold_start_problem.md: Cold Start Problem Analysis", "status": "in_progress", "priority": "high", "id": "create_day3_010"}, {"content": "Create day3_011_cold_start_solutions.md: Cold Start Solutions and Strategies", "status": "pending", "priority": "high", "id": "create_day3_011"}]