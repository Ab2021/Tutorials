# Day 3.4: User-Based Collaborative Filtering

## Learning Objectives
By the end of this session, you will:
- Master the mathematical foundations of user-based collaborative filtering
- Implement advanced user similarity measures and neighborhood selection
- Learn optimization techniques for large-scale user-based systems
- Understand bias correction and normalization strategies
- Handle edge cases and improve prediction quality

## 1. User-Based Collaborative Filtering Fundamentals

User-based collaborative filtering (UBCF) predicts a user's preferences based on ratings from similar users. The core idea is: **"Users who agreed in the past will agree in the future."**

### Mathematical Formulation

Given:
- User u wants rating prediction for item i
- N(u) = set of k most similar users to u who have rated item i
- sim(u,v) = similarity between users u and v
- r̄ᵤ = mean rating of user u

**Basic Prediction Formula:**
```
r̂ᵤᵢ = r̄ᵤ + (Σᵥ∈N(u) sim(u,v) × (rᵥᵢ - r̄ᵥ)) / (Σᵥ∈N(u) |sim(u,v)|)
```

Where:
- r̂ᵤᵢ = predicted rating of user u for item i
- rᵥᵢ = actual rating of user v for item i
- r̄ᵥ = mean rating of user v

## 2. Advanced User Similarity Measures

### 2.1 Weighted Similarity Measures
Standard similarity measures can be enhanced with weighting schemes.

### 2.2 Significance Weighting
Penalize similarities based on few common items:
```
sim_weighted(u,v) = sim(u,v) × min(|I(u,v)|/threshold, 1)
```

Where I(u,v) is the set of items rated by both users.

## 3. Implementation: Advanced User-Based Collaborative Filtering

```python
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Set
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import time
from numba import jit
import heapq

class AdvancedUserBasedCF:
    """
    Advanced User-Based Collaborative Filtering with multiple similarity measures,
    optimization techniques, and bias correction methods.
    """
    
    def __init__(self, 
                 similarity_metric: str = 'cosine',
                 n_neighbors: int = 50,
                 min_common_items: int = 5,
                 significance_weighting: bool = True,
                 case_amplification: float = 2.5,
                 use_bias_correction: bool = True,
                 normalization_method: str = 'user_mean'):
        """
        Initialize Advanced User-Based CF.
        
        Args:
            similarity_metric: 'cosine', 'pearson', 'adjusted_cosine', 'jaccard'
            n_neighbors: Number of neighbors for prediction
            min_common_items: Minimum common items for similarity computation
            significance_weighting: Apply significance weighting to similarities
            case_amplification: Amplification factor for strong similarities
            use_bias_correction: Apply bias correction in predictions
            normalization_method: 'user_mean', 'z_score', 'min_max'
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items
        self.significance_weighting = significance_weighting
        self.case_amplification = case_amplification
        self.use_bias_correction = use_bias_correction
        self.normalization_method = normalization_method
        
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
        self.user_stds = None
        self.item_means = None
        self.user_item_counts = None
        
        # Precomputed structures
        self.user_similarity_matrix = None
        self.user_neighborhoods = {}
        self.knn_model = None
        
    def fit(self, interactions: List[Tuple]):
        """
        Fit the user-based collaborative filtering model.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
        """
        print("Fitting Advanced User-Based CF model...")
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
        
        # Normalize ratings
        self._normalize_ratings()
        
        # Precompute user similarities
        self._compute_user_similarities()
        
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
        self.user_stds = np.zeros(self.n_users)
        self.user_item_counts = np.zeros(self.n_users)
        
        for i in range(self.n_users):
            user_ratings = self.rating_matrix[i, :]
            non_zero_user_ratings = user_ratings[user_ratings > 0]
            
            if len(non_zero_user_ratings) > 0:
                self.user_means[i] = np.mean(non_zero_user_ratings)
                self.user_stds[i] = np.std(non_zero_user_ratings) if len(non_zero_user_ratings) > 1 else 1.0
                self.user_item_counts[i] = len(non_zero_user_ratings)
            else:
                self.user_means[i] = self.global_mean
                self.user_stds[i] = 1.0
                self.user_item_counts[i] = 0
        
        # Item statistics
        self.item_means = np.zeros(self.n_items)
        for j in range(self.n_items):
            item_ratings = self.rating_matrix[:, j]
            non_zero_item_ratings = item_ratings[item_ratings > 0]
            self.item_means[j] = np.mean(non_zero_item_ratings) if len(non_zero_item_ratings) > 0 else self.global_mean
    
    def _normalize_ratings(self):
        """Normalize rating matrix based on specified method."""
        self.normalized_matrix = self.rating_matrix.copy()
        
        if self.normalization_method == 'user_mean':
            # Subtract user mean from each user's ratings
            for i in range(self.n_users):
                user_mask = self.normalized_matrix[i, :] > 0
                if np.any(user_mask):
                    self.normalized_matrix[i, user_mask] -= self.user_means[i]
        
        elif self.normalization_method == 'z_score':
            # Z-score normalization per user
            for i in range(self.n_users):
                user_mask = self.normalized_matrix[i, :] > 0
                if np.any(user_mask) and self.user_stds[i] > 0:
                    self.normalized_matrix[i, user_mask] = \
                        (self.normalized_matrix[i, user_mask] - self.user_means[i]) / self.user_stds[i]
        
        elif self.normalization_method == 'min_max':
            # Min-max normalization per user
            for i in range(self.n_users):
                user_ratings = self.normalized_matrix[i, :]
                user_mask = user_ratings > 0
                if np.any(user_mask):
                    min_rating = np.min(user_ratings[user_mask])
                    max_rating = np.max(user_ratings[user_mask])
                    if max_rating > min_rating:
                        self.normalized_matrix[i, user_mask] = \
                            (user_ratings[user_mask] - min_rating) / (max_rating - min_rating)
    
    def _compute_user_similarities(self):
        """Compute user similarity matrix with optimizations."""
        print("Computing user similarity matrix...")
        
        similarity_start = time.time()
        
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = self._compute_cosine_similarities()
        elif self.similarity_metric == 'pearson':
            self.user_similarity_matrix = self._compute_pearson_similarities()
        elif self.similarity_metric == 'adjusted_cosine':
            self.user_similarity_matrix = self._compute_adjusted_cosine_similarities()
        elif self.similarity_metric == 'jaccard':
            self.user_similarity_matrix = self._compute_jaccard_similarities()
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Apply significance weighting
        if self.significance_weighting:
            self._apply_significance_weighting()
        
        # Apply case amplification
        if self.case_amplification != 1.0:
            self._apply_case_amplification()
        
        similarity_time = time.time() - similarity_start
        print(f"Similarity computation completed in {similarity_time:.2f} seconds")
        
        # Precompute neighborhoods for efficiency
        self._precompute_neighborhoods()
    
    def _compute_cosine_similarities(self) -> np.ndarray:
        """Compute cosine similarities efficiently."""
        # Use normalized matrix for cosine similarity
        matrix = self.normalized_matrix.copy()
        
        # Compute norms
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        
        # Normalize
        normalized = matrix / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Apply minimum common items constraint
        self._apply_min_common_items_constraint(similarity_matrix)
        
        # Remove self-similarities
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _compute_pearson_similarities(self) -> np.ndarray:
        """Compute Pearson correlation similarities."""
        similarity_matrix = np.zeros((self.n_users, self.n_users))
        
        for i in range(self.n_users):
            for j in range(i + 1, self.n_users):
                user_i_ratings = self.rating_matrix[i, :]
                user_j_ratings = self.rating_matrix[j, :]
                
                # Find common items
                common_mask = (user_i_ratings > 0) & (user_j_ratings > 0)
                
                if np.sum(common_mask) >= self.min_common_items:
                    common_i = user_i_ratings[common_mask]
                    common_j = user_j_ratings[common_mask]
                    
                    if len(common_i) > 1:
                        correlation, _ = pearsonr(common_i, common_j)
                        if not np.isnan(correlation):
                            similarity_matrix[i, j] = correlation
                            similarity_matrix[j, i] = correlation
        
        return similarity_matrix
    
    def _compute_adjusted_cosine_similarities(self) -> np.ndarray:
        """Compute adjusted cosine similarities (item-mean centered)."""
        # Center by item means instead of user means
        adjusted_matrix = self.rating_matrix.copy()
        
        for j in range(self.n_items):
            item_mask = adjusted_matrix[:, j] > 0
            if np.any(item_mask):
                adjusted_matrix[item_mask, j] -= self.item_means[j]
        
        # Compute cosine similarity on adjusted matrix
        norms = np.linalg.norm(adjusted_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        
        normalized = adjusted_matrix / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        self._apply_min_common_items_constraint(similarity_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _compute_jaccard_similarities(self) -> np.ndarray:
        """Compute Jaccard similarities for binary preferences."""
        # Convert to binary matrix
        binary_matrix = (self.rating_matrix > 0).astype(int)
        
        similarity_matrix = np.zeros((self.n_users, self.n_users))
        
        for i in range(self.n_users):
            for j in range(i + 1, self.n_users):
                user_i_items = set(np.where(binary_matrix[i, :] > 0)[0])
                user_j_items = set(np.where(binary_matrix[j, :] > 0)[0])
                
                intersection = len(user_i_items & user_j_items)
                union = len(user_i_items | user_j_items)
                
                if union > 0 and intersection >= self.min_common_items:
                    jaccard = intersection / union
                    similarity_matrix[i, j] = jaccard
                    similarity_matrix[j, i] = jaccard
        
        return similarity_matrix
    
    def _apply_min_common_items_constraint(self, similarity_matrix: np.ndarray):
        """Apply minimum common items constraint to similarity matrix."""
        if self.min_common_items <= 1:
            return
        
        # Compute common items matrix
        binary_matrix = (self.rating_matrix > 0).astype(int)
        common_items_matrix = np.dot(binary_matrix, binary_matrix.T)
        
        # Zero out similarities with insufficient common items
        mask = common_items_matrix < self.min_common_items
        similarity_matrix[mask] = 0
    
    def _apply_significance_weighting(self):
        """Apply significance weighting to penalize similarities with few common items."""
        binary_matrix = (self.rating_matrix > 0).astype(int)
        common_items_matrix = np.dot(binary_matrix, binary_matrix.T)
        
        # Significance weighting: min(common_items / threshold, 1)
        threshold = max(self.min_common_items, 20)  # Adjust threshold as needed
        significance_weights = np.minimum(common_items_matrix / threshold, 1.0)
        
        self.user_similarity_matrix *= significance_weights
    
    def _apply_case_amplification(self):
        """Apply case amplification to emphasize strong similarities."""
        if self.case_amplification != 1.0:
            # Amplify positive similarities
            positive_mask = self.user_similarity_matrix > 0
            self.user_similarity_matrix[positive_mask] = \
                np.power(self.user_similarity_matrix[positive_mask], self.case_amplification)
    
    def _precompute_neighborhoods(self):
        """Precompute neighborhoods for each user for efficiency."""
        print("Precomputing user neighborhoods...")
        
        self.user_neighborhoods = {}
        
        for user_idx in range(self.n_users):
            similarities = self.user_similarity_matrix[user_idx, :]
            
            # Get top k neighbors (excluding self)
            top_indices = np.argsort(similarities)[::-1]
            top_indices = top_indices[top_indices != user_idx]
            
            # Keep only positive similarities
            neighbors = []
            for idx in top_indices[:self.n_neighbors]:
                if similarities[idx] > 0:
                    neighbors.append((idx, similarities[idx]))
                else:
                    break
            
            user_id = self.idx_to_user[user_idx]
            self.user_neighborhoods[user_id] = neighbors
    
    def get_user_neighbors(self, user_id: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Get k most similar users.
        
        Args:
            user_id: Target user ID
            k: Number of neighbors (default uses self.n_neighbors)
            
        Returns:
            List of (neighbor_user_id, similarity) tuples
        """
        if user_id not in self.user_neighborhoods:
            return []
        
        neighbors = self.user_neighborhoods[user_id]
        
        if k is not None:
            neighbors = neighbors[:k]
        
        # Convert indices to user IDs
        result = []
        for neighbor_idx, similarity in neighbors:
            neighbor_id = self.idx_to_user[neighbor_idx]
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
        
        # Get neighbors who have rated this item
        neighbors = self.get_user_neighbors(user_id)
        
        numerator = 0.0
        denominator = 0.0
        
        user_mean = self.user_means[user_idx]
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.user_to_idx[neighbor_id]
            neighbor_rating = self.rating_matrix[neighbor_idx, item_idx]
            
            if neighbor_rating > 0:  # Neighbor has rated this item
                if self.use_bias_correction:
                    # Use bias-corrected prediction
                    neighbor_mean = self.user_means[neighbor_idx]
                    deviation = neighbor_rating - neighbor_mean
                    numerator += similarity * deviation
                else:
                    # Simple weighted average
                    numerator += similarity * neighbor_rating
                
                denominator += abs(similarity)
        
        if denominator == 0:
            # No neighbors found, return user mean or global mean
            return user_mean if user_mean > 0 else self.global_mean
        
        if self.use_bias_correction:
            prediction = user_mean + (numerator / denominator)
        else:
            prediction = numerator / denominator
        
        # Clamp to valid rating range
        return np.clip(prediction, 1.0, 5.0)
    
    def predict_ratings_batch(self, user_item_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-item pairs efficiently.
        
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
                       exclude_rated: bool = True,
                       candidate_items: List[str] = None) -> List[Tuple[str, float]]:
        """
        Recommend top N items for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            candidate_items: Specific items to consider (None for all items)
            
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
        item_idx = self.item_to_idx[item_id]
        
        # Get neighbors who rated this item
        neighbors = self.get_user_neighbors(user_id)
        contributing_neighbors = []
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.user_to_idx[neighbor_id]
            neighbor_rating = self.rating_matrix[neighbor_idx, item_idx]
            
            if neighbor_rating > 0:
                contributing_neighbors.append({
                    'neighbor_id': neighbor_id,
                    'similarity': similarity,
                    'rating': neighbor_rating,
                    'contribution': similarity * neighbor_rating
                })
        
        # Sort by contribution
        contributing_neighbors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        predicted_rating = self.predict_rating(user_id, item_id)
        user_mean = self.user_means[user_idx]
        
        explanation = {
            'predicted_rating': predicted_rating,
            'user_mean_rating': user_mean,
            'global_mean_rating': self.global_mean,
            'n_contributing_neighbors': len(contributing_neighbors),
            'top_contributing_neighbors': contributing_neighbors[:5],
            'similarity_metric': self.similarity_metric,
            'use_bias_correction': self.use_bias_correction
        }
        
        return explanation
    
    def analyze_user_profile(self, user_id: str) -> Dict:
        """
        Analyze user's rating profile and preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user profile analysis
        """
        if user_id not in self.user_to_idx:
            return {'error': 'User not found'}
        
        user_idx = self.user_to_idx[user_id]
        user_ratings = self.rating_matrix[user_idx, :]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            return {'error': 'User has no ratings'}
        
        # Get neighbors
        neighbors = self.get_user_neighbors(user_id, k=10)
        
        # Rating distribution
        rating_counts = Counter(rated_items.round().astype(int))
        
        profile = {
            'n_rated_items': len(rated_items),
            'mean_rating': np.mean(rated_items),
            'std_rating': np.std(rated_items),
            'min_rating': np.min(rated_items),
            'max_rating': np.max(rated_items),
            'rating_distribution': dict(rating_counts),
            'n_neighbors': len(neighbors),
            'top_neighbors': neighbors[:5],
            'rating_behavior': self._classify_rating_behavior(rated_items)
        }
        
        return profile
    
    def _classify_rating_behavior(self, ratings: np.ndarray) -> str:
        """Classify user's rating behavior."""
        mean_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        
        if std_rating < 0.5:
            return 'consistent'
        elif mean_rating > 4.0:
            return 'positive'
        elif mean_rating < 2.5:
            return 'critical'
        elif std_rating > 1.5:
            return 'diverse'
        else:
            return 'moderate'
    
    def get_model_statistics(self) -> Dict:
        """Get comprehensive model statistics."""
        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'matrix_density': np.count_nonzero(self.rating_matrix) / self.rating_matrix.size,
            'global_mean_rating': self.global_mean,
            'similarity_metric': self.similarity_metric,
            'n_neighbors': self.n_neighbors,
            'min_common_items': self.min_common_items,
            'use_bias_correction': self.use_bias_correction,
            'significance_weighting': self.significance_weighting,
            'case_amplification': self.case_amplification
        }
        
        # User statistics
        stats.update({
            'avg_items_per_user': np.mean(self.user_item_counts),
            'std_items_per_user': np.std(self.user_item_counts),
            'min_items_per_user': np.min(self.user_item_counts),
            'max_items_per_user': np.max(self.user_item_counts)
        })
        
        # Similarity statistics
        if self.user_similarity_matrix is not None:
            similarities = self.user_similarity_matrix[np.triu_indices_from(self.user_similarity_matrix, k=1)]
            positive_sims = similarities[similarities > 0]
            
            stats.update({
                'avg_similarity': np.mean(positive_sims) if len(positive_sims) > 0 else 0,
                'std_similarity': np.std(positive_sims) if len(positive_sims) > 0 else 0,
                'positive_similarities_ratio': len(positive_sims) / len(similarities) if len(similarities) > 0 else 0
            })
        
        return stats

# Performance optimization utilities
@jit(nopython=True)
def fast_cosine_similarity(matrix: np.ndarray, min_common: int) -> np.ndarray:
    """Fast cosine similarity computation with Numba."""
    n_users = matrix.shape[0]
    similarity_matrix = np.zeros((n_users, n_users))
    
    for i in range(n_users):
        for j in range(i + 1, n_users):
            # Find common non-zero items
            common_mask = (matrix[i, :] > 0) & (matrix[j, :] > 0)
            n_common = np.sum(common_mask)
            
            if n_common >= min_common:
                # Compute cosine similarity
                vec_i = matrix[i, common_mask]
                vec_j = matrix[j, common_mask]
                
                norm_i = np.sqrt(np.sum(vec_i * vec_i))
                norm_j = np.sqrt(np.sum(vec_j * vec_j))
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.sum(vec_i * vec_j) / (norm_i * norm_j)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
    
    return similarity_matrix

# Example usage and testing
def create_user_based_test_data():
    """Create test data with clear user similarity patterns."""
    np.random.seed(42)
    
    # Create user groups with similar preferences
    user_groups = {
        'action_lovers': [f'action_user_{i}' for i in range(20)],
        'romance_fans': [f'romance_user_{i}' for i in range(20)],
        'comedy_enthusiasts': [f'comedy_user_{i}' for i in range(20)],
        'drama_critics': [f'drama_user_{i}' for i in range(20)]
    }
    
    # Create items by genre
    items_by_genre = {
        'action': [f'action_movie_{i}' for i in range(25)],
        'romance': [f'romance_movie_{i}' for i in range(25)],
        'comedy': [f'comedy_movie_{i}' for i in range(25)],
        'drama': [f'drama_movie_{i}' for i in range(25)]
    }
    
    interactions = []
    
    # Generate ratings with group preferences
    genre_preference_map = {
        'action_lovers': 'action',
        'romance_fans': 'romance', 
        'comedy_enthusiasts': 'comedy',
        'drama_critics': 'drama'
    }
    
    for group, users in user_groups.items():
        preferred_genre = genre_preference_map[group]
        
        for user in users:
            # Rate items from preferred genre highly
            preferred_items = np.random.choice(items_by_genre[preferred_genre], 
                                             np.random.randint(15, 25), replace=False)
            for item in preferred_items:
                rating = np.random.normal(4.2, 0.8)
                rating = np.clip(rating, 1, 5)
                interactions.append((user, item, round(rating, 1)))
            
            # Rate some items from other genres (lower ratings)
            for other_genre in items_by_genre:
                if other_genre != preferred_genre:
                    other_items = np.random.choice(items_by_genre[other_genre],
                                                 np.random.randint(3, 8), replace=False)
                    for item in other_items:
                        rating = np.random.normal(2.5, 1.0)
                        rating = np.clip(rating, 1, 5)
                        interactions.append((user, item, round(rating, 1)))
    
    return interactions, user_groups, items_by_genre

if __name__ == "__main__":
    # Create test data
    print("Creating user-based test dataset...")
    interactions, user_groups, items_by_genre = create_user_based_test_data()
    
    print(f"Created {len(interactions)} interactions")
    print(f"User groups: {[f'{group}: {len(users)}' for group, users in user_groups.items()]}")
    
    # Test different similarity metrics
    metrics = ['cosine', 'pearson', 'adjusted_cosine']
    
    for metric in metrics:
        print(f"\nTesting {metric} similarity:")
        
        cf_model = AdvancedUserBasedCF(
            similarity_metric=metric,
            n_neighbors=20,
            min_common_items=5,
            significance_weighting=True,
            use_bias_correction=True
        )
        
        start_time = time.time()
        cf_model.fit(interactions)
        fit_time = time.time() - start_time
        
        print(f"Model fitted in {fit_time:.2f} seconds")
        
        # Test predictions
        test_user = 'action_user_0'
        test_item = 'action_movie_0'  # Should be highly rated
        
        prediction = cf_model.predict_rating(test_user, test_item)
        print(f"Prediction for {test_user} -> {test_item}: {prediction:.2f}")
        
        # Get recommendations
        recommendations = cf_model.recommend_items(test_user, n_recommendations=5)
        print(f"Top 5 recommendations for {test_user}:")
        for item_id, pred_rating in recommendations:
            print(f"  {item_id}: {pred_rating:.2f}")
        
        # Analyze user profile
        profile = cf_model.analyze_user_profile(test_user)
        print(f"User profile: {profile['rating_behavior']}, "
              f"Mean rating: {profile['mean_rating']:.2f}, "
              f"Items rated: {profile['n_rated_items']}")
        
        # Get model statistics
        stats = cf_model.get_model_statistics()
        print(f"Model stats - Density: {stats['matrix_density']:.4f}, "
              f"Avg similarity: {stats.get('avg_similarity', 0):.4f}")
```

## 4. Optimization Techniques

### 4.1 Efficient Similarity Computation
- **Vectorized operations**: Use numpy/scipy for batch computations
- **Sparse matrix operations**: Leverage sparsity for memory efficiency
- **Approximation algorithms**: Use sampling for very large datasets

### 4.2 Neighborhood Precomputation
- **Offline computation**: Precompute neighborhoods during model training
- **Incremental updates**: Update neighborhoods efficiently when new data arrives
- **Caching strategies**: Cache frequently accessed similarities and neighborhoods

### 4.3 Memory Optimization
- **Lazy loading**: Load similarity matrices on demand
- **Compression**: Use appropriate data types and compression
- **Chunked processing**: Process data in chunks for large datasets

## 5. Study Questions

### Basic Level
1. What is the intuition behind user-based collaborative filtering?
2. How does significance weighting improve similarity computation?
3. Why is bias correction important in user-based CF?
4. What are the computational complexity challenges of user-based CF?

### Intermediate Level
5. Implement a version that handles both explicit and implicit feedback.
6. Design a method to detect and handle fake or spam users.
7. How would you adapt user-based CF for real-time recommendations?
8. Compare the effectiveness of different similarity measures on your dataset.

### Advanced Level
9. Implement a distributed user-based CF system using MapReduce.
10. Design an online learning version that updates user similarities incrementally.
11. How would you incorporate temporal factors into user-based CF?
12. Implement a version that handles missing ratings intelligently.

### Tricky Questions
13. A user has rated only very popular items. How would this affect their similarity computation with other users?
14. If two users have very different rating scales but similar preferences, how would different similarity measures handle this?
15. Design a user-based CF system that works well when users have rated fewer than 10 items on average.
16. How would you detect and handle the case where user preferences have fundamentally changed over time?

## 6. Key Takeaways

1. **User-based CF finds similar users** and leverages their preferences
2. **Similarity computation is the bottleneck** requiring optimization
3. **Bias correction significantly improves** prediction accuracy
4. **Significance weighting helps** with sparse similarity estimates
5. **Precomputation and caching** are essential for scalability
6. **Different similarity measures** work better for different data characteristics

## Next Session Preview
In the next session, we'll explore item-based collaborative filtering, which often performs better than user-based approaches and has different computational and scalability characteristics.