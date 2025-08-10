# Day 3.5: Item-Based Collaborative Filtering

## Learning Objectives
By the end of this session, you will:
- Master item-based collaborative filtering algorithms and their advantages
- Understand why item-based CF often outperforms user-based CF
- Implement advanced item similarity measures and optimization techniques
- Learn preprocessing and model selection strategies for item-based systems
- Handle scalability challenges in item-based recommendation systems

## 1. Item-Based Collaborative Filtering Fundamentals

Item-based collaborative filtering (IBCF) predicts a user's preferences by analyzing relationships between items rather than users. The core principle: **"Items liked by similar users are similar to each other."**

### Key Advantages over User-Based CF

1. **Item relationships are more stable** than user preferences
2. **Fewer items than users** in most systems (better scalability)
3. **Items can be pre-processed offline** for faster online recommendations
4. **Better handling of sparse data** due to item-item relationships
5. **More interpretable recommendations** ("because you liked X, you might like Y")

### Mathematical Formulation

Given:
- User u wants rating prediction for item i
- N(i) = set of k most similar items to i that user u has rated
- sim(i,j) = similarity between items i and j
- rᵤⱼ = rating of user u for item j

**Basic Prediction Formula:**
```
r̂ᵤᵢ = (Σⱼ∈N(i) sim(i,j) × rᵤⱼ) / (Σⱼ∈N(i) |sim(i,j)|)
```

**With Item Mean Adjustment:**
```
r̂ᵤᵢ = r̄ᵢ + (Σⱼ∈N(i) sim(i,j) × (rᵤⱼ - r̄ⱼ)) / (Σⱼ∈N(i) |sim(i,j)|)
```

Where r̄ᵢ is the mean rating for item i.

## 2. Implementation: Advanced Item-Based Collaborative Filtering

```python
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Optional, Set
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import time
import pickle
from numba import jit
import heapq

class AdvancedItemBasedCF:
    """
    Advanced Item-Based Collaborative Filtering with multiple similarity measures,
    optimization techniques, and sophisticated prediction strategies.
    """
    
    def __init__(self,
                 similarity_metric: str = 'adjusted_cosine',
                 n_neighbors: int = 50,
                 min_common_users: int = 5,
                 shrinkage_factor: float = 100.0,
                 use_inverse_user_frequency: bool = True,
                 normalization_method: str = 'item_mean',
                 prediction_method: str = 'weighted_average'):
        """
        Initialize Advanced Item-Based CF.
        
        Args:
            similarity_metric: 'cosine', 'adjusted_cosine', 'pearson', 'conditional_probability'
            n_neighbors: Number of neighbors for prediction
            min_common_users: Minimum common users for similarity computation
            shrinkage_factor: Regularization factor for similarity computation
            use_inverse_user_frequency: Apply IUF weighting to similarities
            normalization_method: 'item_mean', 'user_mean', 'z_score'
            prediction_method: 'weighted_average', 'regression', 'deviation_from_mean'
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_users = min_common_users
        self.shrinkage_factor = shrinkage_factor
        self.use_inverse_user_frequency = use_inverse_user_frequency
        self.normalization_method = normalization_method
        self.prediction_method = prediction_method
        
        # Data structures
        self.rating_matrix = None
        self.normalized_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        
        # Statistics
        self.global_mean = 0.0
        self.user_means = None
        self.item_means = None
        self.item_stds = None
        self.user_item_counts = None
        self.item_user_counts = None
        
        # Precomputed structures
        self.item_similarity_matrix = None
        self.item_neighborhoods = {}
        
        # IUF weights
        self.iuf_weights = None
        
    def fit(self, interactions: List[Tuple]):
        """
        Fit the item-based collaborative filtering model.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
        """
        print("Fitting Advanced Item-Based CF model...")
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
        
        # Create mappings
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        # Create rating matrix
        self.rating_matrix = np.zeros((self.n_users, self.n_items))
        
        for _, row in df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            self.rating_matrix[user_idx, item_idx] = row['rating']
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Calculate IUF weights if needed
        if self.use_inverse_user_frequency:
            self._calculate_iuf_weights()
        
        # Normalize ratings
        self._normalize_ratings()
        
        # Precompute item similarities
        self._compute_item_similarities()
        
        fit_time = time.time() - start_time
        print(f"Model fitted in {fit_time:.2f} seconds")
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        print(f"Matrix density: {np.count_nonzero(self.rating_matrix) / self.rating_matrix.size:.4f}")
    
    def _calculate_statistics(self):
        """Calculate comprehensive user and item statistics."""
        # Global statistics
        non_zero_ratings = self.rating_matrix[self.rating_matrix > 0]
        self.global_mean = np.mean(non_zero_ratings) if len(non_zero_ratings) > 0 else 0.0
        
        # User statistics
        self.user_means = np.zeros(self.n_users)
        self.user_item_counts = np.zeros(self.n_users)
        
        for i in range(self.n_users):
            user_ratings = self.rating_matrix[i, :]
            non_zero_user_ratings = user_ratings[user_ratings > 0]
            
            if len(non_zero_user_ratings) > 0:
                self.user_means[i] = np.mean(non_zero_user_ratings)
                self.user_item_counts[i] = len(non_zero_user_ratings)
            else:
                self.user_means[i] = self.global_mean
                self.user_item_counts[i] = 0
        
        # Item statistics
        self.item_means = np.zeros(self.n_items)
        self.item_stds = np.zeros(self.n_items)
        self.item_user_counts = np.zeros(self.n_items)
        
        for j in range(self.n_items):
            item_ratings = self.rating_matrix[:, j]
            non_zero_item_ratings = item_ratings[item_ratings > 0]
            
            if len(non_zero_item_ratings) > 0:
                self.item_means[j] = np.mean(non_zero_item_ratings)
                self.item_stds[j] = np.std(non_zero_item_ratings) if len(non_zero_item_ratings) > 1 else 1.0
                self.item_user_counts[j] = len(non_zero_item_ratings)
            else:
                self.item_means[j] = self.global_mean
                self.item_stds[j] = 1.0
                self.item_user_counts[j] = 0
    
    def _calculate_iuf_weights(self):
        """Calculate Inverse User Frequency weights."""
        print("Calculating IUF weights...")
        
        # IUF weight for item i = log(n_users / n_users_who_rated_i)
        self.iuf_weights = np.zeros(self.n_items)
        
        for j in range(self.n_items):
            n_users_rated = self.item_user_counts[j]
            if n_users_rated > 0:
                self.iuf_weights[j] = np.log(self.n_users / n_users_rated)
            else:
                self.iuf_weights[j] = 0.0
    
    def _normalize_ratings(self):
        """Normalize rating matrix based on specified method."""
        self.normalized_matrix = self.rating_matrix.copy()
        
        if self.normalization_method == 'item_mean':
            # Subtract item mean from each item's ratings
            for j in range(self.n_items):
                item_mask = self.normalized_matrix[:, j] > 0
                if np.any(item_mask):
                    self.normalized_matrix[item_mask, j] -= self.item_means[j]
        
        elif self.normalization_method == 'user_mean':
            # Subtract user mean from each user's ratings
            for i in range(self.n_users):
                user_mask = self.normalized_matrix[i, :] > 0
                if np.any(user_mask):
                    self.normalized_matrix[i, user_mask] -= self.user_means[i]
        
        elif self.normalization_method == 'z_score':
            # Z-score normalization per item
            for j in range(self.n_items):
                item_mask = self.normalized_matrix[:, j] > 0
                if np.any(item_mask) and self.item_stds[j] > 0:
                    self.normalized_matrix[item_mask, j] = \
                        (self.normalized_matrix[item_mask, j] - self.item_means[j]) / self.item_stds[j]
    
    def _compute_item_similarities(self):
        """Compute item similarity matrix with optimizations."""
        print("Computing item similarity matrix...")
        
        similarity_start = time.time()
        
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = self._compute_cosine_similarities()
        elif self.similarity_metric == 'adjusted_cosine':
            self.item_similarity_matrix = self._compute_adjusted_cosine_similarities()
        elif self.similarity_metric == 'pearson':
            self.item_similarity_matrix = self._compute_pearson_similarities()
        elif self.similarity_metric == 'conditional_probability':
            self.item_similarity_matrix = self._compute_conditional_probability_similarities()
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Apply shrinkage regularization
        if self.shrinkage_factor > 0:
            self._apply_shrinkage_regularization()
        
        # Apply IUF weighting
        if self.use_inverse_user_frequency:
            self._apply_iuf_weighting()
        
        similarity_time = time.time() - similarity_start
        print(f"Similarity computation completed in {similarity_time:.2f} seconds")
        
        # Precompute neighborhoods
        self._precompute_neighborhoods()
    
    def _compute_cosine_similarities(self) -> np.ndarray:
        """Compute cosine similarities between items."""
        # Transpose matrix for item-item computation
        item_matrix = self.rating_matrix.T
        
        # Compute norms
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        
        # Normalize
        normalized = item_matrix / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Apply minimum common users constraint
        self._apply_min_common_constraint(similarity_matrix)
        
        # Remove self-similarities
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _compute_adjusted_cosine_similarities(self) -> np.ndarray:
        """Compute adjusted cosine similarities (user-mean centered)."""
        # Create user-mean centered matrix
        adjusted_matrix = self.rating_matrix.copy()
        
        for i in range(self.n_users):
            user_mask = adjusted_matrix[i, :] > 0
            if np.any(user_mask):
                adjusted_matrix[i, user_mask] -= self.user_means[i]
        
        # Transpose for item-item computation
        item_matrix = adjusted_matrix.T
        
        # Compute cosine similarity
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        
        normalized = item_matrix / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        self._apply_min_common_constraint(similarity_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _compute_pearson_similarities(self) -> np.ndarray:
        """Compute Pearson correlation similarities between items."""
        similarity_matrix = np.zeros((self.n_items, self.n_items))
        
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                item_i_ratings = self.rating_matrix[:, i]
                item_j_ratings = self.rating_matrix[:, j]
                
                # Find common users
                common_mask = (item_i_ratings > 0) & (item_j_ratings > 0)
                
                if np.sum(common_mask) >= self.min_common_users:
                    common_i = item_i_ratings[common_mask]
                    common_j = item_j_ratings[common_mask]
                    
                    if len(common_i) > 1:
                        correlation, _ = pearsonr(common_i, common_j)
                        if not np.isnan(correlation):
                            similarity_matrix[i, j] = correlation
                            similarity_matrix[j, i] = correlation
        
        return similarity_matrix
    
    def _compute_conditional_probability_similarities(self) -> np.ndarray:
        """Compute conditional probability based similarities."""
        # Convert to binary matrix (liked/not liked)
        binary_matrix = (self.rating_matrix > self.global_mean).astype(int)
        item_matrix = binary_matrix.T
        
        similarity_matrix = np.zeros((self.n_items, self.n_items))
        
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                # P(like i | like j) and P(like j | like i)
                users_like_i = np.sum(item_matrix[i, :])
                users_like_j = np.sum(item_matrix[j, :])
                users_like_both = np.sum(item_matrix[i, :] & item_matrix[j, :])
                
                if users_like_i > 0 and users_like_j > 0 and users_like_both >= self.min_common_users:
                    prob_i_given_j = users_like_both / users_like_j
                    prob_j_given_i = users_like_both / users_like_i
                    
                    # Use harmonic mean of conditional probabilities
                    if prob_i_given_j + prob_j_given_i > 0:
                        similarity = 2 * prob_i_given_j * prob_j_given_i / (prob_i_given_j + prob_j_given_i)
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _apply_min_common_constraint(self, similarity_matrix: np.ndarray):
        """Apply minimum common users constraint."""
        if self.min_common_users <= 1:
            return
        
        # Compute common users matrix
        binary_matrix = (self.rating_matrix > 0).astype(int)
        item_matrix = binary_matrix.T
        common_users_matrix = np.dot(item_matrix, item_matrix.T)
        
        # Zero out similarities with insufficient common users
        mask = common_users_matrix < self.min_common_users
        similarity_matrix[mask] = 0
    
    def _apply_shrinkage_regularization(self):
        """Apply shrinkage regularization to similarity matrix."""
        if self.shrinkage_factor <= 0:
            return
        
        # Compute common users matrix
        binary_matrix = (self.rating_matrix > 0).astype(int)
        item_matrix = binary_matrix.T
        common_users_matrix = np.dot(item_matrix, item_matrix.T)
        
        # Apply shrinkage: similarity * (common_users / (common_users + shrinkage))
        shrinkage_weights = common_users_matrix / (common_users_matrix + self.shrinkage_factor)
        self.item_similarity_matrix *= shrinkage_weights
    
    def _apply_iuf_weighting(self):
        """Apply Inverse User Frequency weighting."""
        if self.iuf_weights is None:
            return
        
        # Weight similarities by IUF
        for i in range(self.n_items):
            for j in range(self.n_items):
                if i != j:
                    # Geometric mean of IUF weights
                    weight = np.sqrt(self.iuf_weights[i] * self.iuf_weights[j])
                    self.item_similarity_matrix[i, j] *= weight
    
    def _precompute_neighborhoods(self):
        """Precompute neighborhoods for each item."""
        print("Precomputing item neighborhoods...")
        
        self.item_neighborhoods = {}
        
        for item_idx in range(self.n_items):
            similarities = self.item_similarity_matrix[item_idx, :]
            
            # Get top k neighbors (excluding self)
            top_indices = np.argsort(similarities)[::-1]
            top_indices = top_indices[top_indices != item_idx]
            
            # Keep only positive similarities
            neighbors = []
            for idx in top_indices[:self.n_neighbors]:
                if similarities[idx] > 0:
                    neighbors.append((idx, similarities[idx]))
                else:
                    break
            
            item_id = self.idx_to_item[item_idx]
            self.item_neighborhoods[item_id] = neighbors
    
    def get_item_neighbors(self, item_id: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Get k most similar items.
        
        Args:
            item_id: Target item ID
            k: Number of neighbors (default uses self.n_neighbors)
            
        Returns:
            List of (neighbor_item_id, similarity) tuples
        """
        if item_id not in self.item_neighborhoods:
            return []
        
        neighbors = self.item_neighborhoods[item_id]
        
        if k is not None:
            neighbors = neighbors[:k]
        
        # Convert indices to item IDs
        result = []
        for neighbor_idx, similarity in neighbors:
            neighbor_id = self.idx_to_item[neighbor_idx]
            result.append((neighbor_id, similarity))
        
        return result
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Return existing rating if available
        if self.rating_matrix[user_idx, item_idx] > 0:
            return self.rating_matrix[user_idx, item_idx]
        
        if self.prediction_method == 'weighted_average':
            return self._predict_weighted_average(user_id, item_id)
        elif self.prediction_method == 'deviation_from_mean':
            return self._predict_deviation_from_mean(user_id, item_id)
        elif self.prediction_method == 'regression':
            return self._predict_regression(user_id, item_id)
        else:
            raise ValueError(f"Unknown prediction method: {self.prediction_method}")
    
    def _predict_weighted_average(self, user_id: str, item_id: str) -> float:
        """Weighted average prediction method."""
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Get similar items that this user has rated
        neighbors = self.get_item_neighbors(item_id)
        
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.item_to_idx[neighbor_id]
            user_rating = self.rating_matrix[user_idx, neighbor_idx]
            
            if user_rating > 0:  # User has rated this similar item
                numerator += similarity * user_rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.item_means[item_idx]
        
        prediction = numerator / denominator
        return np.clip(prediction, 1.0, 5.0)
    
    def _predict_deviation_from_mean(self, user_id: str, item_id: str) -> float:
        """Deviation from mean prediction method."""
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        item_mean = self.item_means[item_idx]
        
        # Get similar items that this user has rated
        neighbors = self.get_item_neighbors(item_id)
        
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.item_to_idx[neighbor_id]
            user_rating = self.rating_matrix[user_idx, neighbor_idx]
            
            if user_rating > 0:
                neighbor_mean = self.item_means[neighbor_idx]
                deviation = user_rating - neighbor_mean
                numerator += similarity * deviation
                denominator += abs(similarity)
        
        if denominator == 0:
            return item_mean
        
        prediction = item_mean + (numerator / denominator)
        return np.clip(prediction, 1.0, 5.0)
    
    def _predict_regression(self, user_id: str, item_id: str) -> float:
        """Regression-based prediction method."""
        user_idx = self.user_to_idx[user_id]
        
        # Get similar items that this user has rated
        neighbors = self.get_item_neighbors(item_id, k=min(20, self.n_neighbors))
        
        if len(neighbors) < 2:
            return self.item_means[self.item_to_idx[item_id]]
        
        # Prepare data for regression
        X = []  # Similarities
        y = []  # User ratings
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.item_to_idx[neighbor_id]
            user_rating = self.rating_matrix[user_idx, neighbor_idx]
            
            if user_rating > 0:
                X.append(similarity)
                y.append(user_rating)
        
        if len(X) < 2:
            return self.item_means[self.item_to_idx[item_id]]
        
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Simple linear regression: y = a*x + b
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X.flatten() - X_mean) * (y - y_mean))
        denominator = np.sum((X.flatten() - X_mean) ** 2)
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * X_mean
        
        # Predict using average similarity to target item
        avg_similarity = np.mean([sim for _, sim in neighbors[:len(X)]])
        prediction = slope * avg_similarity + intercept
        
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       exclude_rated: bool = True,
                       candidate_items: List[str] = None) -> List[Tuple[str, float]]:
        """
        Recommend top N items for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            candidate_items: Specific items to consider
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        recommendations = []
        
        # Determine candidate items
        if candidate_items is None:
            candidate_items = list(self.item_to_idx.keys())
        
        for item_id in candidate_items:
            if item_id not in self.item_to_idx:
                continue
            
            item_idx = self.item_to_idx[item_id]
            
            # Skip if user already rated this item
            if exclude_rated and self.rating_matrix[user_idx, item_idx] > 0:
                continue
            
            predicted_rating = self.predict_rating(user_id, item_id)
            recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict:
        """
        Provide explanation for a recommendation.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Dictionary with explanation details
        """
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return {'error': 'User or item not found'}
        
        user_idx = self.user_to_idx[user_id]
        
        # Get similar items that user has rated
        neighbors = self.get_item_neighbors(item_id)
        contributing_items = []
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.item_to_idx[neighbor_id]
            user_rating = self.rating_matrix[user_idx, neighbor_idx]
            
            if user_rating > 0:
                contributing_items.append({
                    'item_id': neighbor_id,
                    'similarity': similarity,
                    'user_rating': user_rating,
                    'contribution': similarity * user_rating
                })
        
        # Sort by contribution
        contributing_items.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        predicted_rating = self.predict_rating(user_id, item_id)
        item_mean = self.item_means[self.item_to_idx[item_id]]
        
        explanation = {
            'predicted_rating': predicted_rating,
            'item_mean_rating': item_mean,
            'global_mean_rating': self.global_mean,
            'n_contributing_items': len(contributing_items),
            'top_contributing_items': contributing_items[:5],
            'similarity_metric': self.similarity_metric,
            'prediction_method': self.prediction_method,
            'recommendation_reason': self._generate_recommendation_reason(contributing_items[:3])
        }
        
        return explanation
    
    def _generate_recommendation_reason(self, top_items: List[Dict]) -> str:
        """Generate human-readable recommendation reason."""
        if not top_items:
            return "Based on global preferences"
        
        if len(top_items) == 1:
            item = top_items[0]
            return f"Because you rated {item['item_id']} highly ({item['user_rating']:.1f})"
        
        item_names = [item['item_id'] for item in top_items[:2]]
        return f"Because you liked {', '.join(item_names)} and similar items"
    
    def analyze_item_profile(self, item_id: str) -> Dict:
        """
        Analyze item's rating profile and characteristics.
        
        Args:
            item_id: Item ID
            
        Returns:
            Dictionary with item profile analysis
        """
        if item_id not in self.item_to_idx:
            return {'error': 'Item not found'}
        
        item_idx = self.item_to_idx[item_id]
        item_ratings = self.rating_matrix[:, item_idx]
        user_ratings = item_ratings[item_ratings > 0]
        
        if len(user_ratings) == 0:
            return {'error': 'Item has no ratings'}
        
        # Get similar items
        neighbors = self.get_item_neighbors(item_id, k=10)
        
        # Rating distribution
        rating_counts = Counter(user_ratings.round().astype(int))
        
        profile = {
            'n_ratings': len(user_ratings),
            'mean_rating': np.mean(user_ratings),
            'std_rating': np.std(user_ratings),
            'min_rating': np.min(user_ratings),
            'max_rating': np.max(user_ratings),
            'rating_distribution': dict(rating_counts),
            'n_similar_items': len(neighbors),
            'top_similar_items': neighbors[:5],
            'popularity_tier': self._classify_item_popularity(len(user_ratings)),
            'rating_pattern': self._classify_rating_pattern(user_ratings)
        }
        
        return profile
    
    def _classify_item_popularity(self, n_ratings: int) -> str:
        """Classify item popularity based on number of ratings."""
        quartiles = np.percentile(self.item_user_counts[self.item_user_counts > 0], [25, 50, 75])
        
        if n_ratings >= quartiles[2]:
            return 'popular'
        elif n_ratings >= quartiles[1]:
            return 'moderate'
        elif n_ratings >= quartiles[0]:
            return 'niche'
        else:
            return 'rare'
    
    def _classify_rating_pattern(self, ratings: np.ndarray) -> str:
        """Classify item's rating pattern."""
        mean_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        
        if mean_rating >= 4.0 and std_rating < 0.8:
            return 'universally_loved'
        elif mean_rating <= 2.0 and std_rating < 0.8:
            return 'universally_disliked'
        elif std_rating > 1.5:
            return 'polarizing'
        elif mean_rating > 3.5:
            return 'generally_liked'
        elif mean_rating < 2.5:
            return 'generally_disliked'
        else:
            return 'mixed_reception'
    
    def get_model_statistics(self) -> Dict:
        """Get comprehensive model statistics."""
        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'matrix_density': np.count_nonzero(self.rating_matrix) / self.rating_matrix.size,
            'global_mean_rating': self.global_mean,
            'similarity_metric': self.similarity_metric,
            'n_neighbors': self.n_neighbors,
            'min_common_users': self.min_common_users,
            'shrinkage_factor': self.shrinkage_factor,
            'use_inverse_user_frequency': self.use_inverse_user_frequency,
            'prediction_method': self.prediction_method
        }
        
        # Item statistics
        stats.update({
            'avg_users_per_item': np.mean(self.item_user_counts),
            'std_users_per_item': np.std(self.item_user_counts),
            'min_users_per_item': np.min(self.item_user_counts),
            'max_users_per_item': np.max(self.item_user_counts)
        })
        
        # Similarity statistics
        if self.item_similarity_matrix is not None:
            similarities = self.item_similarity_matrix[np.triu_indices_from(self.item_similarity_matrix, k=1)]
            positive_sims = similarities[similarities > 0]
            
            stats.update({
                'avg_similarity': np.mean(positive_sims) if len(positive_sims) > 0 else 0,
                'std_similarity': np.std(positive_sims) if len(positive_sims) > 0 else 0,
                'positive_similarities_ratio': len(positive_sims) / len(similarities) if len(similarities) > 0 else 0
            })
        
        return stats
    
    def compare_items(self, item1_id: str, item2_id: str) -> Dict:
        """
        Compare two items in detail.
        
        Args:
            item1_id, item2_id: Item IDs to compare
            
        Returns:
            Dictionary with comparison details
        """
        if item1_id not in self.item_to_idx or item2_id not in self.item_to_idx:
            return {'error': 'One or both items not found'}
        
        item1_idx = self.item_to_idx[item1_id]
        item2_idx = self.item_to_idx[item2_id]
        
        # Get similarity
        similarity = self.item_similarity_matrix[item1_idx, item2_idx]
        
        # Get profiles
        profile1 = self.analyze_item_profile(item1_id)
        profile2 = self.analyze_item_profile(item2_id)
        
        # Find common users
        ratings1 = self.rating_matrix[:, item1_idx]
        ratings2 = self.rating_matrix[:, item2_idx]
        common_mask = (ratings1 > 0) & (ratings2 > 0)
        common_users = np.sum(common_mask)
        
        comparison = {
            'similarity': similarity,
            'common_users': common_users,
            'item1_profile': profile1,
            'item2_profile': profile2
        }
        
        if common_users > 0:
            common_ratings1 = ratings1[common_mask]
            common_ratings2 = ratings2[common_mask]
            
            comparison.update({
                'correlation_on_common_users': np.corrcoef(common_ratings1, common_ratings2)[0, 1],
                'avg_rating_diff_common_users': np.mean(common_ratings1 - common_ratings2)
            })
        
        return comparison

# Optimization utilities for large-scale systems
class ItemBasedCFOptimizer:
    """Utilities for optimizing item-based CF at scale."""
    
    @staticmethod
    def compute_similarities_in_batches(rating_matrix: np.ndarray, 
                                       batch_size: int = 1000,
                                       similarity_metric: str = 'adjusted_cosine') -> np.ndarray:
        """
        Compute item similarities in batches to manage memory.
        
        Args:
            rating_matrix: User-item rating matrix
            batch_size: Number of items per batch
            similarity_metric: Similarity metric to use
            
        Returns:
            Item similarity matrix
        """
        n_items = rating_matrix.shape[1]
        similarity_matrix = np.zeros((n_items, n_items))
        
        n_batches = (n_items + batch_size - 1) // batch_size
        
        print(f"Computing similarities in {n_batches} batches...")
        
        for i in range(n_batches):
            start_i = i * batch_size
            end_i = min((i + 1) * batch_size, n_items)
            
            for j in range(i, n_batches):
                start_j = j * batch_size
                end_j = min((j + 1) * batch_size, n_items)
                
                # Extract batch data
                if similarity_metric == 'adjusted_cosine':
                    # User-mean centered data
                    user_means = np.mean(rating_matrix, axis=1, keepdims=True)
                    user_means[user_means == 0] = 0  # Handle users with no ratings
                    
                    centered_matrix = rating_matrix.copy()
                    mask = rating_matrix > 0
                    centered_matrix[mask] -= user_means[mask[:, 0]]
                    
                    batch_i = centered_matrix[:, start_i:end_i].T
                    batch_j = centered_matrix[:, start_j:end_j].T
                else:
                    batch_i = rating_matrix[:, start_i:end_i].T
                    batch_j = rating_matrix[:, start_j:end_j].T
                
                # Compute similarities for this batch pair
                batch_similarities = cosine_similarity(batch_i, batch_j)
                
                # Store results
                similarity_matrix[start_i:end_i, start_j:end_j] = batch_similarities
                
                if i != j:  # Symmetric matrix
                    similarity_matrix[start_j:end_j, start_i:end_i] = batch_similarities.T
            
            print(f"Completed batch {i + 1}/{n_batches}")
        
        return similarity_matrix
    
    @staticmethod
    def select_top_k_similarities(similarity_matrix: np.ndarray, k: int = 50) -> Dict:
        """
        Select only top-k similarities for each item to save memory.
        
        Args:
            similarity_matrix: Full similarity matrix
            k: Number of top similarities to keep
            
        Returns:
            Dictionary mapping item indices to their top-k neighbors
        """
        top_k_neighbors = {}
        n_items = similarity_matrix.shape[0]
        
        for i in range(n_items):
            similarities = similarity_matrix[i, :]
            
            # Get top k (excluding self)
            top_indices = np.argsort(similarities)[::-1]
            top_indices = top_indices[top_indices != i][:k]
            
            # Keep only positive similarities
            neighbors = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    neighbors.append((idx, similarities[idx]))
                else:
                    break
            
            top_k_neighbors[i] = neighbors
        
        return top_k_neighbors

# Example usage and testing
def create_item_based_test_data():
    """Create test data optimized for item-based CF evaluation."""
    np.random.seed(42)
    
    # Create item categories with different characteristics
    item_categories = {
        'blockbuster_movies': [f'blockbuster_{i}' for i in range(30)],  # Many ratings, high avg
        'indie_films': [f'indie_{i}' for i in range(40)],  # Moderate ratings, diverse
        'classic_movies': [f'classic_{i}' for i in range(25)],  # Fewer ratings, very high avg
        'niche_documentaries': [f'doc_{i}' for i in range(20)]  # Few ratings, polarizing
    }
    
    # Create users with different rating behaviors
    user_types = {
        'mainstream': [f'mainstream_user_{i}' for i in range(50)],
        'cinephile': [f'cinephile_{i}' for i in range(30)],
        'casual': [f'casual_user_{i}' for i in range(40)],
        'critic': [f'critic_{i}' for i in range(20)]
    }
    
    interactions = []
    
    # Generate ratings based on user type and item category compatibility
    compatibility_matrix = {
        'mainstream': {'blockbuster_movies': 4.2, 'indie_films': 3.0, 'classic_movies': 3.5, 'niche_documentaries': 2.5},
        'cinephile': {'blockbuster_movies': 3.5, 'indie_films': 4.3, 'classic_movies': 4.5, 'niche_documentaries': 4.0},
        'casual': {'blockbuster_movies': 4.0, 'indie_films': 3.2, 'classic_movies': 3.8, 'niche_documentaries': 2.8},
        'critic': {'blockbuster_movies': 3.2, 'indie_films': 4.1, 'classic_movies': 4.2, 'niche_documentaries': 3.8}
    }
    
    for user_type, users in user_types.items():
        for user in users:
            for category, items in item_categories.items():
                base_rating = compatibility_matrix[user_type][category]
                
                # Determine how many items from this category the user will rate
                if category == 'blockbuster_movies':
                    n_items = np.random.randint(15, 25)  # Most users rate blockbusters
                elif category == 'indie_films':
                    n_items = np.random.randint(5, 15)
                elif category == 'classic_movies':
                    n_items = np.random.randint(3, 10)
                else:  # niche_documentaries
                    n_items = np.random.randint(1, 6)
                
                rated_items = np.random.choice(items, min(n_items, len(items)), replace=False)
                
                for item in rated_items:
                    # Add noise to base rating
                    noise = np.random.normal(0, 0.7)
                    final_rating = np.clip(base_rating + noise, 1, 5)
                    interactions.append((user, item, round(final_rating, 1)))
    
    return interactions, user_types, item_categories

if __name__ == "__main__":
    # Create test data
    print("Creating item-based test dataset...")
    interactions, user_types, item_categories = create_item_based_test_data()
    
    print(f"Created {len(interactions)} interactions")
    print(f"User types: {[f'{utype}: {len(users)}' for utype, users in user_types.items()]}")
    print(f"Item categories: {[f'{cat}: {len(items)}' for cat, items in item_categories.items()]}")
    
    # Test different similarity metrics
    metrics = ['adjusted_cosine', 'cosine', 'pearson']
    
    for metric in metrics:
        print(f"\nTesting {metric} similarity:")
        
        cf_model = AdvancedItemBasedCF(
            similarity_metric=metric,
            n_neighbors=20,
            min_common_users=5,
            shrinkage_factor=100.0,
            use_inverse_user_frequency=True,
            prediction_method='deviation_from_mean'
        )
        
        start_time = time.time()
        cf_model.fit(interactions)
        fit_time = time.time() - start_time
        
        print(f"Model fitted in {fit_time:.2f} seconds")
        
        # Test predictions
        test_user = 'mainstream_user_0'
        test_item = 'blockbuster_0'
        
        prediction = cf_model.predict_rating(test_user, test_item)
        print(f"Prediction for {test_user} -> {test_item}: {prediction:.2f}")
        
        # Get recommendations
        recommendations = cf_model.recommend_items(test_user, n_recommendations=5)
        print(f"Top 5 recommendations for {test_user}:")
        for item_id, pred_rating in recommendations:
            print(f"  {item_id}: {pred_rating:.2f}")
        
        # Analyze item profile
        profile = cf_model.analyze_item_profile('blockbuster_0')
        print(f"Item profile: {profile['popularity_tier']}, "
              f"Pattern: {profile['rating_pattern']}, "
              f"Mean rating: {profile['mean_rating']:.2f}")
        
        # Compare items
        comparison = cf_model.compare_items('blockbuster_0', 'blockbuster_1')
        print(f"Item similarity: {comparison['similarity']:.4f}, "
              f"Common users: {comparison['common_users']}")
        
        # Get model statistics
        stats = cf_model.get_model_statistics()
        print(f"Model stats - Density: {stats['matrix_density']:.4f}, "
              f"Avg similarity: {stats.get('avg_similarity', 0):.4f}")
```

## 3. Advanced Techniques for Item-Based CF

### 3.1 Slope One Algorithm
A simplified item-based approach that uses average rating differences:

```python
class SlopeOneCF:
    """
    Slope One collaborative filtering algorithm - simple and effective.
    """
    
    def __init__(self):
        self.deviations = {}
        self.frequencies = {}
        self.user_means = {}
    
    def fit(self, interactions: List[Tuple]):
        """Fit Slope One model."""
        # Convert to user-item dictionary
        user_items = defaultdict(dict)
        
        for user_id, item_id, rating in interactions:
            user_items[user_id][item_id] = rating
        
        # Calculate deviations and frequencies
        self.deviations = defaultdict(lambda: defaultdict(float))
        self.frequencies = defaultdict(lambda: defaultdict(int))
        
        for user_ratings in user_items.values():
            for item1, rating1 in user_ratings.items():
                for item2, rating2 in user_ratings.items():
                    if item1 != item2:
                        self.deviations[item1][item2] += rating1 - rating2
                        self.frequencies[item1][item2] += 1
        
        # Average the deviations
        for item1 in self.deviations:
            for item2 in self.deviations[item1]:
                self.deviations[item1][item2] /= self.frequencies[item1][item2]
        
        # Calculate user means
        for user_id, ratings in user_items.items():
            self.user_means[user_id] = np.mean(list(ratings.values()))
    
    def predict_rating(self, user_id: str, item_id: str, user_ratings: Dict[str, float]) -> float:
        """Predict rating using Slope One."""
        numerator = 0.0
        denominator = 0.0
        
        for other_item, rating in user_ratings.items():
            if other_item != item_id and item_id in self.deviations.get(other_item, {}):
                deviation = self.deviations[other_item][item_id]
                frequency = self.frequencies[other_item][item_id]
                
                numerator += (deviation + rating) * frequency
                denominator += frequency
        
        if denominator == 0:
            return self.user_means.get(user_id, 3.0)
        
        return numerator / denominator
```

### 3.2 Asymmetric SVD for Item-Based CF
Incorporate asymmetric relationships between items:

```python
def asymmetric_item_similarity(rating_matrix: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Compute asymmetric item similarities where sim(i,j) != sim(j,i).
    
    Args:
        rating_matrix: User-item rating matrix
        alpha: Asymmetry parameter (0 = symmetric, 1 = fully asymmetric)
    """
    n_items = rating_matrix.shape[1]
    similarity_matrix = np.zeros((n_items, n_items))
    
    # Compute item popularity (number of ratings)
    item_popularity = np.sum(rating_matrix > 0, axis=0)
    
    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                # Standard cosine similarity
                vec_i = rating_matrix[:, i]
                vec_j = rating_matrix[:, j]
                
                common_mask = (vec_i > 0) & (vec_j > 0)
                
                if np.sum(common_mask) > 0:
                    common_i = vec_i[common_mask]
                    common_j = vec_j[common_mask]
                    
                    cosine_sim = np.dot(common_i, common_j) / (np.linalg.norm(common_i) * np.linalg.norm(common_j))
                    
                    # Apply asymmetric weighting based on popularity
                    popularity_ratio = item_popularity[j] / (item_popularity[i] + 1e-10)
                    asymmetric_weight = popularity_ratio ** alpha
                    
                    similarity_matrix[i, j] = cosine_sim * asymmetric_weight
    
    return similarity_matrix
```

## 4. Scalability and Optimization

### 4.1 Model Compression
- **Top-k filtering**: Keep only top-k similarities per item
- **Threshold filtering**: Remove similarities below threshold
- **Matrix factorization**: Approximate similarity matrix

### 4.2 Incremental Updates
- **Online similarity updates**: Update similarities as new ratings arrive
- **Approximate updates**: Use approximation algorithms for efficiency
- **Decay factors**: Apply time-based decay to old similarities

### 4.3 Distributed Computing
- **MapReduce similarity computation**: Distribute similarity calculations
- **Spark-based implementations**: Use Spark for large-scale processing
- **Parameter servers**: Use parameter servers for model synchronization

## 5. Study Questions

### Basic Level
1. Why does item-based CF often outperform user-based CF?
2. What is the intuition behind adjusted cosine similarity?
3. How does shrinkage regularization help with similarity computation?
4. What are the computational advantages of item-based over user-based CF?

### Intermediate Level
5. Implement a version that handles both explicit ratings and implicit feedback (clicks, views).
6. Design a method to detect and handle adversarial item relationships.
7. How would you adapt item-based CF for real-time streaming recommendations?
8. Compare different prediction methods (weighted average vs deviation from mean).

### Advanced Level
9. Implement a distributed item-based CF system using Apache Spark.
10. Design an incremental learning version that updates item similarities online.
11. How would you incorporate content features into item-based similarity computation?
12. Implement a version that handles temporal dynamics in item relationships.

### Tricky Questions
13. An item receives only 5-star ratings from 10 users. How would different similarity measures handle this item?
14. If two items are never rated by the same user, but users who rate item A also tend to rate items similar to item B, how would you capture this relationship?
15. Design an item-based CF system that works well for a catalog with millions of items but each user rates only 10-20 items.
16. How would you detect and handle the case where item characteristics have fundamentally changed (e.g., software updates, movie remasters)?

## 6. Key Takeaways

1. **Item-based CF leverages item relationships** which are often more stable than user preferences
2. **Adjusted cosine similarity** typically works best by removing user bias
3. **Shrinkage regularization** prevents overfitting with sparse similarity estimates
4. **IUF weighting** helps handle popularity bias in recommendations
5. **Item-based approaches scale better** than user-based for most real-world scenarios
6. **Precomputation and optimization** are crucial for production systems

## Next Session Preview
In the next session, we'll explore advanced similarity measures and distance metrics that can be used in both user-based and item-based collaborative filtering, including techniques for handling different data types and scaling challenges.