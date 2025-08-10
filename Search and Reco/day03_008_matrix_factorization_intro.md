# Day 3.8: Introduction to Matrix Factorization

## Learning Objectives
By the end of this session, you will:
- Understand the mathematical foundations of matrix factorization for recommendations
- Master the core concepts and advantages of model-based collaborative filtering
- Implement basic matrix factorization algorithms from scratch
- Learn about different factorization techniques and their applications
- Handle regularization, optimization, and evaluation of factorization models

## 1. Matrix Factorization Fundamentals

### Definition
Matrix factorization decomposes a user-item rating matrix R into the product of two lower-dimensional matrices: user factors U and item factors V.

```
R ≈ U × V^T
```

Where:
- R: m×n rating matrix (m users, n items)
- U: m×k user factor matrix (k latent factors)
- V: n×k item factor matrix (k latent factors)

### Key Advantages over Neighborhood Methods

1. **Dimensionality Reduction**: Captures underlying patterns in lower dimensions
2. **Handles Sparsity Better**: Can predict for completely new user-item pairs
3. **Latent Factor Discovery**: Identifies hidden themes/genres/preferences
4. **Scalability**: More efficient for large datasets
5. **Flexibility**: Easy to incorporate biases, regularization, and constraints

### Intuition Behind Latent Factors

Latent factors represent hidden characteristics:
- **Movie example**: Factors might represent genres (action, romance, comedy)
- **Music example**: Factors might represent styles (pop, rock, classical)
- **E-commerce example**: Factors might represent categories (electronics, books, clothing)

## 2. Mathematical Foundation

### Basic Matrix Factorization Model

For user u and item i, the predicted rating is:
```
r̂ᵤᵢ = Σₖ₌₁ᴷ uᵤₖ × vᵢₖ = uᵤ · vᵢ
```

Where:
- uᵤ: user u's factor vector (1×k)
- vᵢ: item i's factor vector (1×k)
- K: number of latent factors

### Matrix Factorization with Biases

More sophisticated model includes bias terms:
```
r̂ᵤᵢ = μ + bᵤ + bᵢ + uᵤ · vᵢ
```

Where:
- μ: global mean rating
- bᵤ: user u's bias (tendency to rate higher/lower)
- bᵢ: item i's bias (tendency to be rated higher/lower)

### Objective Function

We minimize the squared error with regularization:
```
min Σ(u,i)∈R (rᵤᵢ - r̂ᵤᵢ)² + λ(||U||²F + ||V||²F + ||b||²)
```

Where:
- λ: regularization parameter
- ||·||²F: Frobenius norm (sum of squared elements)

## 3. Implementation: Basic Matrix Factorization

```python
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import time
from collections import defaultdict

class BasicMatrixFactorization:
    """
    Basic Matrix Factorization using Stochastic Gradient Descent.
    """
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 regularization: float = 0.01, n_epochs: int = 100,
                 init_mean: float = 0.0, init_std: float = 0.1,
                 use_bias: bool = True, random_state: int = 42):
        """
        Initialize Matrix Factorization model.
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: Regularization parameter (lambda)
            n_epochs: Number of training epochs
            init_mean: Mean for factor initialization
            init_std: Standard deviation for factor initialization
            use_bias: Whether to use bias terms
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.use_bias = use_bias
        self.random_state = random_state
        
        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0.0
        
        # Data mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        
        # Training history
        self.training_history = []
        
    def fit(self, interactions: List[Tuple], validation_data: List[Tuple] = None,
            verbose: bool = True):
        """
        Fit the matrix factorization model.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
            validation_data: Optional validation data for monitoring
            verbose: Whether to print training progress
        """
        np.random.seed(self.random_state)
        
        # Create mappings
        self._create_mappings(interactions)
        
        # Convert to matrix format for easier processing
        train_matrix, train_indices = self._create_training_data(interactions)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            # Shuffle training data
            np.random.shuffle(train_indices)
            
            # SGD updates
            epoch_loss = 0.0
            for user_idx, item_idx, rating in train_indices:
                # Compute prediction and error
                prediction = self._predict_rating_idx(user_idx, item_idx)
                error = rating - prediction
                
                # Store current parameters for regularization
                user_factors_old = self.user_factors[user_idx, :].copy()
                item_factors_old = self.item_factors[item_idx, :].copy()
                
                # Update factors
                self.user_factors[user_idx, :] += self.learning_rate * (
                    error * self.item_factors[item_idx, :] - 
                    self.regularization * user_factors_old
                )
                
                self.item_factors[item_idx, :] += self.learning_rate * (
                    error * user_factors_old - 
                    self.regularization * item_factors_old
                )
                
                # Update biases if used
                if self.use_bias:
                    self.user_biases[user_idx] += self.learning_rate * (
                        error - self.regularization * self.user_biases[user_idx]
                    )
                    
                    self.item_biases[item_idx] += self.learning_rate * (
                        error - self.regularization * self.item_biases[item_idx]
                    )
                
                epoch_loss += error ** 2
            
            # Calculate metrics
            epoch_time = time.time() - epoch_start
            train_rmse = np.sqrt(epoch_loss / len(train_indices))
            
            metrics = {
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'epoch_time': epoch_time
            }
            
            # Validation metrics if provided
            if validation_data:
                val_predictions = []
                val_actuals = []
                
                for user_id, item_id, rating in validation_data:
                    pred = self.predict_rating(user_id, item_id)
                    val_predictions.append(pred)
                    val_actuals.append(rating)
                
                val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
                val_mae = mean_absolute_error(val_actuals, val_predictions)
                
                metrics['val_rmse'] = val_rmse
                metrics['val_mae'] = val_mae
            
            self.training_history.append(metrics)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if validation_data:
                    print(f"Epoch {epoch + 1:3d}: Train RMSE={train_rmse:.4f}, "
                          f"Val RMSE={val_rmse:.4f}, Time={epoch_time:.2f}s")
                else:
                    print(f"Epoch {epoch + 1:3d}: Train RMSE={train_rmse:.4f}, "
                          f"Time={epoch_time:.2f}s")
    
    def _create_mappings(self, interactions: List[Tuple]):
        """Create user and item ID to index mappings."""
        users = set()
        items = set()
        
        for user_id, item_id, _ in interactions:
            users.add(user_id)
            items.add(item_id)
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(users)
        self.n_items = len(items)
        
        print(f"Created mappings: {self.n_users} users, {self.n_items} items")
    
    def _create_training_data(self, interactions: List[Tuple]):
        """Convert interactions to training format."""
        # Calculate global mean
        ratings = [rating for _, _, rating in interactions]
        self.global_mean = np.mean(ratings)
        
        # Create training indices
        train_indices = []
        for user_id, item_id, rating in interactions:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            train_indices.append((user_idx, item_idx, rating))
        
        # Create sparse matrix for reference
        rows = [user_idx for user_idx, _, _ in train_indices]
        cols = [item_idx for _, item_idx, _ in train_indices]
        data = [rating for _, _, rating in train_indices]
        
        train_matrix = sparse.csr_matrix((data, (rows, cols)), 
                                       shape=(self.n_users, self.n_items))
        
        return train_matrix, train_indices
    
    def _initialize_parameters(self):
        """Initialize user and item factor matrices and biases."""
        # Initialize factor matrices
        self.user_factors = np.random.normal(
            self.init_mean, self.init_std, 
            (self.n_users, self.n_factors)
        )
        
        self.item_factors = np.random.normal(
            self.init_mean, self.init_std, 
            (self.n_items, self.n_factors)
        )
        
        # Initialize biases
        if self.use_bias:
            self.user_biases = np.zeros(self.n_users)
            self.item_biases = np.zeros(self.n_items)
        
        print(f"Initialized parameters with {self.n_factors} factors")
    
    def _predict_rating_idx(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using internal indices."""
        # Base prediction from factors
        prediction = np.dot(self.user_factors[user_idx, :], 
                          self.item_factors[item_idx, :])
        
        # Add biases if used
        if self.use_bias:
            prediction += (self.global_mean + 
                         self.user_biases[user_idx] + 
                         self.item_biases[item_idx])
        
        return prediction
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        # Handle unknown users/items
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        return self._predict_rating_idx(user_idx, item_idx)
    
    def predict_ratings_batch(self, user_item_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-item pairs.
        
        Args:
            user_item_pairs: List of (user_id, item_id) tuples
            
        Returns:
            List of predicted ratings
        """
        predictions = []
        for user_id, item_id in user_item_pairs:
            prediction = self.predict_rating(user_id, item_id)
            predictions.append(prediction)
        
        return predictions
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       exclude_rated: List[str] = None) -> List[Tuple[str, float]]:
        """
        Recommend top N items for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            exclude_rated: List of items to exclude
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        if exclude_rated is None:
            exclude_rated = []
        
        exclude_set = set(exclude_rated)
        
        # Get predictions for all items
        recommendations = []
        for item_id in self.item_to_idx.keys():
            if item_id not in exclude_set:
                predicted_rating = self.predict_rating(user_id, item_id)
                recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_user_factors(self, user_id: str) -> Optional[np.ndarray]:
        """Get user factor vector."""
        if user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        return self.user_factors[user_idx, :].copy()
    
    def get_item_factors(self, item_id: str) -> Optional[np.ndarray]:
        """Get item factor vector."""
        if item_id not in self.item_to_idx:
            return None
        
        item_idx = self.item_to_idx[item_id]
        return self.item_factors[item_idx, :].copy()
    
    def find_similar_users(self, user_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar users based on factor vectors.
        
        Args:
            user_id: Target user
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_id, similarity) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        target_idx = self.user_to_idx[user_id]
        target_factors = self.user_factors[target_idx, :]
        
        # Compute cosine similarities
        similarities = []
        for idx, other_factors in enumerate(self.user_factors):
            if idx != target_idx:
                # Cosine similarity
                dot_product = np.dot(target_factors, other_factors)
                norm_product = (np.linalg.norm(target_factors) * 
                              np.linalg.norm(other_factors))
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    other_user_id = self.idx_to_user[idx]
                    similarities.append((other_user_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def find_similar_items(self, item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar items based on factor vectors.
        
        Args:
            item_id: Target item
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity) tuples
        """
        if item_id not in self.item_to_idx:
            return []
        
        target_idx = self.item_to_idx[item_id]
        target_factors = self.item_factors[target_idx, :]
        
        # Compute cosine similarities
        similarities = []
        for idx, other_factors in enumerate(self.item_factors):
            if idx != target_idx:
                # Cosine similarity
                dot_product = np.dot(target_factors, other_factors)
                norm_product = (np.linalg.norm(target_factors) * 
                              np.linalg.norm(other_factors))
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    other_item_id = self.idx_to_item[idx]
                    similarities.append((other_item_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def explain_recommendation(self, user_id: str, item_id: str, 
                             top_factors: int = 5) -> Dict:
        """
        Explain a recommendation using factor contributions.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            top_factors: Number of top contributing factors to show
            
        Returns:
            Dictionary with explanation details
        """
        if (user_id not in self.user_to_idx or 
            item_id not in self.item_to_idx):
            return {'error': 'User or item not found'}
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        user_factors = self.user_factors[user_idx, :]
        item_factors = self.item_factors[item_idx, :]
        
        # Calculate factor contributions
        contributions = user_factors * item_factors
        
        # Get top contributing factors
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_factors]
        
        factor_explanations = []
        for factor_idx in top_indices:
            factor_explanations.append({
                'factor_id': int(factor_idx),
                'user_value': float(user_factors[factor_idx]),
                'item_value': float(item_factors[factor_idx]),
                'contribution': float(contributions[factor_idx])
            })
        
        explanation = {
            'predicted_rating': self.predict_rating(user_id, item_id),
            'global_mean': self.global_mean,
            'top_factor_contributions': factor_explanations,
            'total_factor_contribution': float(np.sum(contributions))
        }
        
        if self.use_bias:
            explanation['user_bias'] = float(self.user_biases[user_idx])
            explanation['item_bias'] = float(self.item_biases[item_idx])
        
        return explanation
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.training_history:
            print("No training history available")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_rmse = [h['train_rmse'] for h in self.training_history]
        
        plt.figure(figsize=(12, 4))
        
        # Training RMSE
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_rmse, 'b-', label='Train RMSE')
        
        if 'val_rmse' in self.training_history[0]:
            val_rmse = [h['val_rmse'] for h in self.training_history]
            plt.plot(epochs, val_rmse, 'r-', label='Validation RMSE')
        
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training time per epoch
        plt.subplot(1, 2, 2)
        epoch_times = [h['epoch_time'] for h in self.training_history]
        plt.plot(epochs, epoch_times, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_factors(self, factor_names: List[str] = None) -> Dict:
        """
        Analyze learned factors.
        
        Args:
            factor_names: Optional names for factors
            
        Returns:
            Factor analysis results
        """
        if factor_names is None:
            factor_names = [f'Factor_{i}' for i in range(self.n_factors)]
        
        analysis = {}
        
        # User factor analysis
        user_factor_means = np.mean(self.user_factors, axis=0)
        user_factor_stds = np.std(self.user_factors, axis=0)
        
        # Item factor analysis
        item_factor_means = np.mean(self.item_factors, axis=0)
        item_factor_stds = np.std(self.item_factors, axis=0)
        
        analysis['user_factors'] = {
            'means': user_factor_means,
            'stds': user_factor_stds,
            'factor_names': factor_names
        }
        
        analysis['item_factors'] = {
            'means': item_factor_means,
            'stds': item_factor_stds,
            'factor_names': factor_names
        }
        
        return analysis
    
    def visualize_factors(self, max_entities: int = 50):
        """
        Visualize factor matrices.
        
        Args:
            max_entities: Maximum number of entities to show in heatmap
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # User factors heatmap
        n_users_show = min(max_entities, self.n_users)
        user_sample = self.user_factors[:n_users_show, :]
        
        sns.heatmap(user_sample, cmap='RdBu_r', center=0, ax=axes[0, 0])
        axes[0, 0].set_title(f'User Factors (first {n_users_show} users)')
        axes[0, 0].set_xlabel('Factors')
        axes[0, 0].set_ylabel('Users')
        
        # Item factors heatmap
        n_items_show = min(max_entities, self.n_items)
        item_sample = self.item_factors[:n_items_show, :]
        
        sns.heatmap(item_sample, cmap='RdBu_r', center=0, ax=axes[0, 1])
        axes[0, 1].set_title(f'Item Factors (first {n_items_show} items)')
        axes[0, 1].set_xlabel('Factors')
        axes[0, 1].set_ylabel('Items')
        
        # Factor distribution
        axes[1, 0].hist(self.user_factors.flatten(), bins=50, alpha=0.7, 
                       label='User factors', edgecolor='black')
        axes[1, 0].hist(self.item_factors.flatten(), bins=50, alpha=0.7, 
                       label='Item factors', edgecolor='black')
        axes[1, 0].set_xlabel('Factor Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Factor Value Distribution')
        axes[1, 0].legend()
        
        # Bias distribution (if available)
        if self.use_bias:
            axes[1, 1].hist(self.user_biases, bins=30, alpha=0.7, 
                           label='User biases', edgecolor='black')
            axes[1, 1].hist(self.item_biases, bins=30, alpha=0.7, 
                           label='Item biases', edgecolor='black')
            axes[1, 1].set_xlabel('Bias Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Bias Distribution')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No bias terms used', 
                           transform=axes[1, 1].transAxes, 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Bias Analysis')
        
        plt.tight_layout()
        plt.show()

class AlternatingLeastSquares:
    """
    Matrix Factorization using Alternating Least Squares (ALS).
    More efficient for implicit feedback data.
    """
    
    def __init__(self, n_factors: int = 50, regularization: float = 0.01,
                 n_iterations: int = 50, random_state: int = 42,
                 confidence_alpha: float = 40.0):
        """
        Initialize ALS model.
        
        Args:
            n_factors: Number of latent factors
            regularization: Regularization parameter
            n_iterations: Number of ALS iterations
            random_state: Random seed
            confidence_alpha: Confidence parameter for implicit feedback
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.confidence_alpha = confidence_alpha
        
        # Model parameters
        self.user_factors = None
        self.item_factors = None
        
        # Data mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        
        # Training data
        self.confidence_matrix = None
        self.preference_matrix = None
    
    def fit(self, interactions: List[Tuple], verbose: bool = True):
        """
        Fit ALS model.
        
        Args:
            interactions: List of (user_id, item_id, rating/count) tuples
            verbose: Whether to print progress
        """
        np.random.seed(self.random_state)
        
        # Create mappings and matrices
        self._create_mappings(interactions)
        self._create_confidence_matrix(interactions)
        
        # Initialize factors
        self._initialize_factors()
        
        # ALS iterations
        for iteration in range(self.n_iterations):
            iter_start = time.time()
            
            # Update user factors
            self._update_user_factors()
            
            # Update item factors
            self._update_item_factors()
            
            iter_time = time.time() - iter_start
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:3d}: Time={iter_time:.2f}s")
    
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
    
    def _create_confidence_matrix(self, interactions: List[Tuple]):
        """Create confidence and preference matrices for implicit feedback."""
        # Create sparse matrices
        rows = []
        cols = []
        data = []
        
        for user_id, item_id, rating in interactions:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(rating)
        
        # Rating matrix
        rating_matrix = sparse.csr_matrix((data, (rows, cols)), 
                                        shape=(self.n_users, self.n_items))
        
        # Confidence matrix: c_ui = 1 + alpha * r_ui
        self.confidence_matrix = rating_matrix.copy()
        self.confidence_matrix.data = 1 + self.confidence_alpha * rating_matrix.data
        
        # Preference matrix: p_ui = 1 if r_ui > 0, else 0
        self.preference_matrix = rating_matrix.copy()
        self.preference_matrix.data = np.ones_like(rating_matrix.data)
    
    def _initialize_factors(self):
        """Initialize factor matrices."""
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
    
    def _update_user_factors(self):
        """Update user factors using least squares."""
        # For each user, solve: (Y^T C^u Y + λI) x_u = Y^T C^u p_u
        for u in range(self.n_users):
            # Get user's confidence and preference vectors
            c_u = self.confidence_matrix[u, :].toarray().flatten()
            p_u = self.preference_matrix[u, :].toarray().flatten()
            
            # Create diagonal confidence matrix for this user
            C_u_diag = c_u
            
            # Compute Y^T C^u Y + λI
            YT_Cu_Y = np.zeros((self.n_factors, self.n_factors))
            YT_Cu_p = np.zeros(self.n_factors)
            
            for i in range(self.n_items):
                if C_u_diag[i] > 0:
                    y_i = self.item_factors[i, :]
                    YT_Cu_Y += C_u_diag[i] * np.outer(y_i, y_i)
                    YT_Cu_p += C_u_diag[i] * p_u[i] * y_i
            
            # Add regularization
            YT_Cu_Y += self.regularization * np.eye(self.n_factors)
            
            # Solve for user factors
            try:
                self.user_factors[u, :] = np.linalg.solve(YT_Cu_Y, YT_Cu_p)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                self.user_factors[u, :] = np.linalg.lstsq(YT_Cu_Y, YT_Cu_p, rcond=None)[0]
    
    def _update_item_factors(self):
        """Update item factors using least squares."""
        # For each item, solve: (X^T C^i X + λI) y_i = X^T C^i p_i
        for i in range(self.n_items):
            # Get item's confidence and preference vectors
            c_i = self.confidence_matrix[:, i].toarray().flatten()
            p_i = self.preference_matrix[:, i].toarray().flatten()
            
            # Compute X^T C^i X + λI
            XT_Ci_X = np.zeros((self.n_factors, self.n_factors))
            XT_Ci_p = np.zeros(self.n_factors)
            
            for u in range(self.n_users):
                if c_i[u] > 0:
                    x_u = self.user_factors[u, :]
                    XT_Ci_X += c_i[u] * np.outer(x_u, x_u)
                    XT_Ci_p += c_i[u] * p_i[u] * x_u
            
            # Add regularization
            XT_Ci_X += self.regularization * np.eye(self.n_factors)
            
            # Solve for item factors
            try:
                self.item_factors[i, :] = np.linalg.solve(XT_Ci_X, XT_Ci_p)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                self.item_factors[i, :] = np.linalg.lstsq(XT_Ci_X, XT_Ci_p, rcond=None)[0]
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating for user-item pair."""
        if (user_id not in self.user_to_idx or 
            item_id not in self.item_to_idx):
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        return np.dot(self.user_factors[user_idx, :], 
                     self.item_factors[item_idx, :])
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       exclude_interacted: bool = True) -> List[Tuple[str, float]]:
        """Recommend items for a user."""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's interacted items if excluding
        interacted_items = set()
        if exclude_interacted:
            user_interactions = self.preference_matrix[user_idx, :].nonzero()[1]
            interacted_items = {self.idx_to_item[i] for i in user_interactions}
        
        # Calculate scores for all items
        user_vector = self.user_factors[user_idx, :]
        scores = np.dot(self.item_factors, user_vector)
        
        # Create recommendations
        recommendations = []
        for item_idx, score in enumerate(scores):
            item_id = self.idx_to_item[item_idx]
            if item_id not in interacted_items:
                recommendations.append((item_id, float(score)))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]

# Example usage and testing
def create_matrix_factorization_test_data():
    """Create test data for matrix factorization."""
    np.random.seed(42)
    
    # Create synthetic data with latent factors
    n_users = 300
    n_items = 200
    n_factors_true = 5
    
    # True latent factors
    true_user_factors = np.random.normal(0, 1, (n_users, n_factors_true))
    true_item_factors = np.random.normal(0, 1, (n_items, n_factors_true))
    
    # Generate ratings from latent factors
    true_ratings = np.dot(true_user_factors, true_item_factors.T)
    
    # Add biases
    user_biases = np.random.normal(0, 0.5, n_users)
    item_biases = np.random.normal(0, 0.3, n_items)
    global_mean = 3.5
    
    for i in range(n_users):
        for j in range(n_items):
            true_ratings[i, j] += global_mean + user_biases[i] + item_biases[j]
    
    # Add noise and make sparse
    noise = np.random.normal(0, 0.3, (n_users, n_items))
    true_ratings += noise
    
    # Create sparsity pattern (only keep 10% of ratings)
    observed_mask = np.random.random((n_users, n_items)) < 0.1
    
    # Convert to interaction format
    interactions = []
    for i in range(n_users):
        for j in range(n_items):
            if observed_mask[i, j]:
                rating = np.clip(true_ratings[i, j], 1, 5)
                interactions.append((f'user_{i}', f'item_{j}', rating))
    
    return interactions, true_user_factors, true_item_factors

if __name__ == "__main__":
    # Create test data
    print("Creating matrix factorization test data...")
    interactions, true_user_factors, true_item_factors = create_matrix_factorization_test_data()
    
    print(f"Created {len(interactions)} interactions")
    
    # Split data
    train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Test Basic Matrix Factorization
    print("\n=== Testing Basic Matrix Factorization ===")
    
    mf_model = BasicMatrixFactorization(
        n_factors=10,
        learning_rate=0.01,
        regularization=0.01,
        n_epochs=100,
        use_bias=True,
        random_state=42
    )
    
    # Train model
    start_time = time.time()
    mf_model.fit(train_data, validation_data=val_data, verbose=True)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate on test set
    test_predictions = []
    test_actuals = []
    
    for user_id, item_id, rating in test_data:
        pred = mf_model.predict_rating(user_id, item_id)
        test_predictions.append(pred)
        test_actuals.append(rating)
    
    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
    test_mae = mean_absolute_error(test_actuals, test_predictions)
    
    print(f"\nTest Results:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    
    # Plot training history
    mf_model.plot_training_history()
    
    # Visualize factors
    mf_model.visualize_factors(max_entities=30)
    
    # Test recommendations
    print("\n=== Sample Recommendations ===")
    sample_user = 'user_0'
    recommendations = mf_model.recommend_items(sample_user, n_recommendations=5)
    
    print(f"Top 5 recommendations for {sample_user}:")
    for item_id, pred_rating in recommendations:
        print(f"  {item_id}: {pred_rating:.2f}")
    
    # Test similarity
    similar_users = mf_model.find_similar_users(sample_user, n_similar=5)
    print(f"\nUsers similar to {sample_user}:")
    for user_id, similarity in similar_users:
        print(f"  {user_id}: {similarity:.4f}")
    
    # Test explanation
    explanation = mf_model.explain_recommendation(sample_user, recommendations[0][0])
    print(f"\nExplanation for recommending {recommendations[0][0]} to {sample_user}:")
    print(f"Predicted rating: {explanation['predicted_rating']:.2f}")
    print("Top factor contributions:")
    for factor in explanation['top_factor_contributions']:
        print(f"  Factor {factor['factor_id']}: {factor['contribution']:.4f}")
    
    # Test ALS model
    print("\n=== Testing ALS Model ===")
    
    als_model = AlternatingLeastSquares(
        n_factors=10,
        regularization=0.01,
        n_iterations=50,
        confidence_alpha=40.0,
        random_state=42
    )
    
    als_model.fit(train_data, verbose=True)
    
    # Test ALS recommendations
    als_recommendations = als_model.recommend_items(sample_user, n_recommendations=5)
    print(f"\nALS recommendations for {sample_user}:")
    for item_id, score in als_recommendations:
        print(f"  {item_id}: {score:.2f}")
```

## 4. Types of Matrix Factorization

### 4.1 Basic Matrix Factorization
- Simple dot product model
- Good for explicit feedback
- Fast to train with SGD

### 4.2 Non-Negative Matrix Factorization (NMF)
- Constraints factors to be non-negative
- More interpretable factors
- Good for count/frequency data

### 4.3 Probabilistic Matrix Factorization (PMF)
- Bayesian approach with priors
- Better uncertainty quantification
- Handles missing data naturally

### 4.4 Weighted Matrix Factorization
- Different weights for different observations
- Good for implicit feedback
- Handles confidence in observations

## 5. Optimization Techniques

### 5.1 Stochastic Gradient Descent (SGD)
- Updates after each observation
- Fast convergence
- Easy to implement

### 5.2 Alternating Least Squares (ALS)
- Alternates between optimizing U and V
- Efficient for implicit feedback
- Parallelizable

### 5.3 Coordinate Descent
- Optimizes one parameter at a time
- Good convergence properties
- Memory efficient

## 6. Study Questions

### Basic Level
1. What are the main advantages of matrix factorization over neighborhood methods?
2. How do bias terms improve matrix factorization models?
3. What role does regularization play in matrix factorization?
4. How do you determine the optimal number of latent factors?

### Intermediate Level
5. Implement matrix factorization with different regularization schemes (L1, L2, elastic net).
6. Design a matrix factorization model that handles both explicit and implicit feedback.
7. How would you adapt matrix factorization for temporal recommendation scenarios?
8. Compare SGD vs ALS optimization for different types of data.

### Advanced Level
9. Implement a distributed matrix factorization algorithm using MapReduce.
10. Design a matrix factorization model that incorporates side information (user/item features).
11. How would you handle the cold start problem in matrix factorization?
12. Implement online matrix factorization that updates with streaming data.

### Tricky Questions
13. A new item has no ratings. How can matrix factorization make recommendations for it?
14. If users' preferences change over time, how would you modify matrix factorization to handle this?
15. Design a matrix factorization system that can handle both positive and negative feedback.
16. How would you detect and handle adversarial ratings designed to manipulate the factorization?

## 7. Key Takeaways

1. **Matrix factorization learns latent representations** that capture underlying patterns
2. **Bias terms are crucial** for handling systematic differences in users and items
3. **Regularization prevents overfitting** and improves generalization
4. **Different optimization methods** (SGD, ALS) suit different scenarios
5. **Factor interpretation** provides insights into user preferences and item characteristics
6. **Scalability and efficiency** make MF suitable for large-scale systems

## Next Session Preview
In the next session, we'll explore advanced matrix factorization techniques including SVD++, temporal matrix factorization, and deep learning approaches that extend the basic matrix factorization framework.