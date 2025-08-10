# Day 3.3: Collaborative Filtering Fundamentals

## Learning Objectives
By the end of this session, you will:
- Understand the core principles and assumptions of collaborative filtering
- Master the mathematical foundations of neighborhood-based methods
- Learn the difference between memory-based and model-based approaches
- Implement fundamental collaborative filtering algorithms
- Analyze the strengths and limitations of collaborative filtering

## 1. What is Collaborative Filtering?

Collaborative Filtering (CF) is a method of making automatic predictions about interests by collecting preferences from many users. The underlying assumption is that if two users have similar preferences in the past, they will have similar preferences in the future.

### Core Principle
**"People who agreed in the past will agree in the future"**

### Mathematical Foundation
Given:
- U = {u₁, u₂, ..., uₘ} set of users
- I = {i₁, i₂, ..., iₙ} set of items  
- R = m×n user-item rating matrix

Goal: Predict r̂ᵤᵢ (rating of user u for item i) for unknown entries

## 2. Types of Collaborative Filtering

### 2.1 Memory-Based (Neighborhood-Based)
- **User-Based CF**: Find similar users and recommend items they liked
- **Item-Based CF**: Find similar items to those the user liked

### 2.2 Model-Based
- **Matrix Factorization**: Decompose rating matrix into latent factors
- **Clustering**: Group users/items and make predictions within clusters
- **Deep Learning**: Neural networks for complex pattern learning

## 3. Core Assumptions of Collaborative Filtering

### Assumption 1: User Consistency
Users' preferences remain relatively stable over time.

### Assumption 2: Preference Transitivity  
If user A likes items similar to user B, and user B likes an item, then user A will likely enjoy that item.

### Assumption 3: Sufficient Data
Enough user-item interactions exist to find meaningful patterns.

### Assumption 4: Community Effect
Users' preferences are influenced by community preferences.

## 4. Implementation: Fundamental Collaborative Filtering Framework

```python
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Union
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import heapq

class CollaborativeFilteringBase:
    """
    Base class for collaborative filtering algorithms with common functionality.
    """
    
    def __init__(self, similarity_metric: str = 'cosine', 
                 n_neighbors: int = 50,
                 min_common_items: int = 5):
        """
        Initialize CF base class.
        
        Args:
            similarity_metric: 'cosine', 'pearson', 'euclidean', 'jaccard'
            n_neighbors: Number of neighbors to consider
            min_common_items: Minimum common items/users for similarity computation
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items
        
        # Data structures
        self.rating_matrix = None
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
        
        # Similarity matrices (computed on demand)
        self.user_similarity = None
        self.item_similarity = None
        
    def fit(self, interactions: List[Tuple]):
        """
        Fit the collaborative filtering model.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
        """
        print("Fitting collaborative filtering model...")
        
        # Convert to DataFrame for easier processing
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
        
        print(f"Model fitted: {self.n_users} users, {self.n_items} items")
        print(f"Matrix density: {np.count_nonzero(self.rating_matrix) / self.rating_matrix.size:.4f}")
    
    def _calculate_statistics(self):
        """Calculate and cache statistics."""
        # Global mean (excluding zeros)
        non_zero_ratings = self.rating_matrix[self.rating_matrix > 0]
        self.global_mean = np.mean(non_zero_ratings) if len(non_zero_ratings) > 0 else 0.0
        
        # User means
        self.user_means = np.zeros(self.n_users)
        for i in range(self.n_users):
            user_ratings = self.rating_matrix[i, :]
            non_zero_user_ratings = user_ratings[user_ratings > 0]
            self.user_means[i] = np.mean(non_zero_user_ratings) if len(non_zero_user_ratings) > 0 else self.global_mean
        
        # Item means
        self.item_means = np.zeros(self.n_items)
        for j in range(self.n_items):
            item_ratings = self.rating_matrix[:, j]
            non_zero_item_ratings = item_ratings[item_ratings > 0]
            self.item_means[j] = np.mean(non_zero_item_ratings) if len(non_zero_item_ratings) > 0 else self.global_mean
    
    def _compute_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute similarity between two rating vectors.
        
        Args:
            vector1, vector2: Rating vectors
            
        Returns:
            Similarity score
        """
        # Find common non-zero entries
        mask1 = vector1 > 0
        mask2 = vector2 > 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_common_items:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            norm1 = np.linalg.norm(v1_common)
            norm2 = np.linalg.norm(v2_common)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(v1_common, v2_common) / (norm1 * norm2)
        
        elif self.similarity_metric == 'pearson':
            # Pearson correlation coefficient
            if len(v1_common) < 2:
                return 0.0
            
            correlation, _ = pearsonr(v1_common, v2_common)
            return correlation if not np.isnan(correlation) else 0.0
        
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance converted to similarity
            distance = euclidean(v1_common, v2_common)
            return 1 / (1 + distance)
        
        elif self.similarity_metric == 'jaccard':
            # Jaccard similarity (for binary preferences)
            set1 = set(np.where(vector1 > 0)[0])
            set2 = set(np.where(vector2 > 0)[0])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def _compute_user_similarity_matrix(self):
        """Compute and cache user-user similarity matrix."""
        if self.user_similarity is not None:
            return
        
        print("Computing user similarity matrix...")
        self.user_similarity = np.zeros((self.n_users, self.n_users))
        
        for i in range(self.n_users):
            if i % 100 == 0:
                print(f"  Processing user {i}/{self.n_users}")
            
            for j in range(i + 1, self.n_users):
                similarity = self._compute_similarity(
                    self.rating_matrix[i, :], 
                    self.rating_matrix[j, :]
                )
                self.user_similarity[i, j] = similarity
                self.user_similarity[j, i] = similarity  # Symmetric
        
        print("User similarity matrix computation complete")
    
    def _compute_item_similarity_matrix(self):
        """Compute and cache item-item similarity matrix."""
        if self.item_similarity is not None:
            return
        
        print("Computing item similarity matrix...")
        self.item_similarity = np.zeros((self.n_items, self.n_items))
        
        for i in range(self.n_items):
            if i % 100 == 0:
                print(f"  Processing item {i}/{self.n_items}")
            
            for j in range(i + 1, self.n_items):
                similarity = self._compute_similarity(
                    self.rating_matrix[:, i], 
                    self.rating_matrix[:, j]
                )
                self.item_similarity[i, j] = similarity
                self.item_similarity[j, i] = similarity  # Symmetric
        
        print("Item similarity matrix computation complete")
    
    def get_user_neighbors(self, user_id: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Get k most similar users to the given user.
        
        Args:
            user_id: Target user ID
            k: Number of neighbors (default: self.n_neighbors)
            
        Returns:
            List of (user_id, similarity) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        if k is None:
            k = self.n_neighbors
        
        self._compute_user_similarity_matrix()
        
        user_idx = self.user_to_idx[user_id]
        similarities = self.user_similarity[user_idx, :]
        
        # Get top k similar users (excluding self)
        top_indices = np.argsort(similarities)[::-1]
        top_indices = top_indices[top_indices != user_idx][:k]
        
        neighbors = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only positive similarities
                neighbor_id = self.idx_to_user[idx]
                neighbors.append((neighbor_id, similarities[idx]))
        
        return neighbors
    
    def get_item_neighbors(self, item_id: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Get k most similar items to the given item.
        
        Args:
            item_id: Target item ID
            k: Number of neighbors (default: self.n_neighbors)
            
        Returns:
            List of (item_id, similarity) tuples
        """
        if item_id not in self.item_to_idx:
            return []
        
        if k is None:
            k = self.n_neighbors
        
        self._compute_item_similarity_matrix()
        
        item_idx = self.item_to_idx[item_id]
        similarities = self.item_similarity[item_idx, :]
        
        # Get top k similar items (excluding self)
        top_indices = np.argsort(similarities)[::-1]
        top_indices = top_indices[top_indices != item_idx][:k]
        
        neighbors = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only positive similarities
                neighbor_id = self.idx_to_item[idx]
                neighbors.append((neighbor_id, similarities[idx]))
        
        return neighbors
    
    def predict_rating(self, user_id: str, item_id: str, method: str = 'user_based') -> float:
        """
        Predict rating for user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            method: 'user_based' or 'item_based'
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # If rating already exists, return it
        if self.rating_matrix[user_idx, item_idx] > 0:
            return self.rating_matrix[user_idx, item_idx]
        
        if method == 'user_based':
            return self._predict_user_based(user_id, item_id)
        elif method == 'item_based':
            return self._predict_item_based(user_id, item_id)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
    
    def _predict_user_based(self, user_id: str, item_id: str) -> float:
        """User-based collaborative filtering prediction."""
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        user_mean = self.user_means[user_idx]
        
        # Get similar users who have rated this item
        neighbors = self.get_user_neighbors(user_id)
        
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.user_to_idx[neighbor_id]
            neighbor_rating = self.rating_matrix[neighbor_idx, item_idx]
            
            if neighbor_rating > 0:  # Neighbor has rated this item
                neighbor_mean = self.user_means[neighbor_idx]
                numerator += similarity * (neighbor_rating - neighbor_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return user_mean  # No similar users found
        
        prediction = user_mean + (numerator / denominator)
        
        # Clamp prediction to valid range
        return np.clip(prediction, 1.0, 5.0)
    
    def _predict_item_based(self, user_id: str, item_id: str) -> float:
        """Item-based collaborative filtering prediction."""
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
            return self.item_means[item_idx]  # No similar items found
        
        prediction = numerator / denominator
        
        # Clamp prediction to valid range
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend_items(self, user_id: str, n_recommendations: int = 10,
                       method: str = 'user_based',
                       exclude_rated: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend top N items for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            method: 'user_based' or 'item_based'  
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        recommendations = []
        
        for item_id in self.item_to_idx.keys():
            item_idx = self.item_to_idx[item_id]
            
            # Skip if user already rated this item and exclude_rated is True
            if exclude_rated and self.rating_matrix[user_idx, item_idx] > 0:
                continue
            
            predicted_rating = self.predict_rating(user_id, item_id, method)
            recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def evaluate(self, test_interactions: List[Tuple], 
                 method: str = 'user_based') -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_interactions: List of (user_id, item_id, rating) tuples
            method: Prediction method
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = []
        actuals = []
        
        for user_id, item_id, actual_rating in test_interactions:
            if user_id in self.user_to_idx and item_id in self.item_to_idx:
                predicted_rating = self.predict_rating(user_id, item_id, method)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
        
        if len(predictions) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        coverage = len(predictions) / len(test_interactions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'n_predictions': len(predictions)
        }
    
    def analyze_similarity_distribution(self):
        """Analyze and visualize similarity distributions."""
        self._compute_user_similarity_matrix()
        self._compute_item_similarity_matrix()
        
        plt.figure(figsize=(15, 5))
        
        # User similarity distribution
        plt.subplot(1, 3, 1)
        user_sims = self.user_similarity[np.triu_indices_from(self.user_similarity, k=1)]
        user_sims = user_sims[user_sims > 0]  # Only positive similarities
        plt.hist(user_sims, bins=50, alpha=0.7, edgecolor='black')
        plt.title('User Similarity Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        
        # Item similarity distribution  
        plt.subplot(1, 3, 2)
        item_sims = self.item_similarity[np.triu_indices_from(self.item_similarity, k=1)]
        item_sims = item_sims[item_sims > 0]  # Only positive similarities
        plt.hist(item_sims, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Item Similarity Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        
        # Comparison
        plt.subplot(1, 3, 3)
        plt.hist(user_sims, bins=30, alpha=0.5, label='User Similarities', edgecolor='black')
        plt.hist(item_sims, bins=30, alpha=0.5, label='Item Similarities', edgecolor='black')
        plt.title('Similarity Comparison')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("Similarity Statistics:")
        print(f"User similarities - Mean: {np.mean(user_sims):.4f}, Std: {np.std(user_sims):.4f}")
        print(f"Item similarities - Mean: {np.mean(item_sims):.4f}, Std: {np.std(item_sims):.4f}")
        print(f"Positive user similarities: {len(user_sims)} / {len(self.user_similarity[np.triu_indices_from(self.user_similarity, k=1)])}")
        print(f"Positive item similarities: {len(item_sims)} / {len(self.item_similarity[np.triu_indices_from(self.item_similarity, k=1)])}")

# Advanced CF implementation with optimizations
class OptimizedCollaborativeFiltering(CollaborativeFilteringBase):
    """
    Optimized version with efficient similarity computation and caching.
    """
    
    def __init__(self, similarity_metric='cosine', n_neighbors=50, 
                 min_common_items=5, use_bias=True):
        super().__init__(similarity_metric, n_neighbors, min_common_items)
        self.use_bias = use_bias
        self.similarity_cache = {}
        
    def _compute_similarities_vectorized(self, matrix: np.ndarray) -> np.ndarray:
        """Compute similarity matrix using vectorized operations."""
        if self.similarity_metric == 'cosine':
            # Normalize rows
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized = matrix / norms
            
            # Compute cosine similarity
            similarity_matrix = np.dot(normalized, normalized.T)
            
            # Zero out similarities with insufficient common items
            if self.min_common_items > 1:
                binary_matrix = (matrix > 0).astype(int)
                common_items = np.dot(binary_matrix, binary_matrix.T)
                mask = common_items < self.min_common_items
                similarity_matrix[mask] = 0
            
            # Remove self-similarities
            np.fill_diagonal(similarity_matrix, 0)
            
            return similarity_matrix
        
        else:
            # Fall back to pairwise computation for other metrics
            return super()._compute_user_similarity_matrix()
    
    def _compute_user_similarity_matrix(self):
        """Optimized user similarity computation."""
        if self.user_similarity is not None:
            return
        
        print("Computing user similarity matrix (vectorized)...")
        self.user_similarity = self._compute_similarities_vectorized(self.rating_matrix)
        print("User similarity matrix computation complete")
    
    def _compute_item_similarity_matrix(self):
        """Optimized item similarity computation."""
        if self.item_similarity is not None:
            return
        
        print("Computing item similarity matrix (vectorized)...")
        self.item_similarity = self._compute_similarities_vectorized(self.rating_matrix.T)
        print("Item similarity matrix computation complete")
    
    def _predict_user_based_with_bias(self, user_id: str, item_id: str) -> float:
        """User-based prediction with bias correction."""
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Baseline prediction: global_mean + user_bias + item_bias
        user_bias = self.user_means[user_idx] - self.global_mean
        item_bias = self.item_means[item_idx] - self.global_mean
        baseline = self.global_mean + user_bias + item_bias
        
        # Get similar users who have rated this item
        neighbors = self.get_user_neighbors(user_id)
        
        numerator = 0.0
        denominator = 0.0
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.user_to_idx[neighbor_id]
            neighbor_rating = self.rating_matrix[neighbor_idx, item_idx]
            
            if neighbor_rating > 0:
                # Compute neighbor's baseline for this item
                neighbor_bias = self.user_means[neighbor_idx] - self.global_mean
                neighbor_baseline = self.global_mean + neighbor_bias + item_bias
                
                # Use deviation from baseline
                deviation = neighbor_rating - neighbor_baseline
                numerator += similarity * deviation
                denominator += abs(similarity)
        
        if denominator == 0:
            return baseline
        
        prediction = baseline + (numerator / denominator)
        return np.clip(prediction, 1.0, 5.0)
    
    def predict_rating(self, user_id: str, item_id: str, method: str = 'user_based') -> float:
        """Enhanced prediction with bias correction option."""
        if self.use_bias and method == 'user_based':
            return self._predict_user_based_with_bias(user_id, item_id)
        else:
            return super().predict_rating(user_id, item_id, method)

# Example usage and comparison
def create_movielens_sample():
    """Create a sample dataset similar to MovieLens."""
    np.random.seed(42)
    
    users = [f'user_{i}' for i in range(1, 201)]  # 200 users
    movies = [f'movie_{i}' for i in range(1, 101)]  # 100 movies
    
    interactions = []
    
    # Create user profiles with preferences
    user_profiles = {}
    genres = ['action', 'comedy', 'drama', 'horror', 'romance']
    
    for user in users:
        # Each user prefers 2-3 genres
        preferred_genres = np.random.choice(genres, np.random.randint(2, 4), replace=False)
        user_profiles[user] = preferred_genres
    
    # Create movie profiles
    movie_profiles = {}
    for movie in movies:
        movie_genre = np.random.choice(genres)
        movie_profiles[movie] = movie_genre
    
    # Generate ratings based on user-movie genre compatibility
    for user in users:
        n_ratings = np.random.randint(10, 40)  # Each user rates 10-40 movies
        rated_movies = np.random.choice(movies, n_ratings, replace=False)
        
        for movie in rated_movies:
            # Base rating from genre compatibility
            if movie_profiles[movie] in user_profiles[user]:
                base_rating = np.random.normal(4, 0.8)  # Higher rating for preferred genres
            else:
                base_rating = np.random.normal(2.5, 1.0)  # Lower rating for non-preferred
            
            # Add some randomness
            final_rating = np.clip(base_rating + np.random.normal(0, 0.3), 1, 5)
            interactions.append((user, movie, round(final_rating, 1)))
    
    return interactions, user_profiles, movie_profiles

if __name__ == "__main__":
    # Create sample data
    print("Creating sample MovieLens-style dataset...")
    interactions, user_profiles, movie_profiles = create_movielens_sample()
    
    print(f"Created {len(interactions)} interactions")
    print(f"Sample interactions: {interactions[:5]}")
    
    # Split into train/test
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(train_interactions)}, Test: {len(test_interactions)}")
    
    # Test basic CF
    print("\nTesting Basic Collaborative Filtering:")
    cf_basic = CollaborativeFilteringBase(similarity_metric='cosine', n_neighbors=30)
    cf_basic.fit(train_interactions)
    
    # Evaluate
    metrics_user = cf_basic.evaluate(test_interactions, method='user_based')
    metrics_item = cf_basic.evaluate(test_interactions, method='item_based')
    
    print(f"User-based CF - RMSE: {metrics_user['rmse']:.4f}, MAE: {metrics_user['mae']:.4f}")
    print(f"Item-based CF - RMSE: {metrics_item['rmse']:.4f}, MAE: {metrics_item['mae']:.4f}")
    
    # Test optimized CF
    print("\nTesting Optimized Collaborative Filtering:")
    cf_opt = OptimizedCollaborativeFiltering(
        similarity_metric='cosine', n_neighbors=30, use_bias=True
    )
    cf_opt.fit(train_interactions)
    
    metrics_opt = cf_opt.evaluate(test_interactions, method='user_based')
    print(f"Optimized CF - RMSE: {metrics_opt['rmse']:.4f}, MAE: {metrics_opt['mae']:.4f}")
    
    # Sample recommendations
    print("\nSample Recommendations:")
    sample_user = 'user_1'
    recommendations = cf_basic.recommend_items(sample_user, n_recommendations=5)
    
    print(f"Top 5 recommendations for {sample_user}:")
    for item_id, predicted_rating in recommendations:
        print(f"  {item_id}: {predicted_rating:.2f}")
    
    # Analyze similarities
    print("\nAnalyzing similarity patterns...")
    cf_basic.analyze_similarity_distribution()
```

## 5. Challenges and Limitations

### 5.1 Sparsity Problem
- Most users rate very few items
- Most items are rated by very few users
- Leads to poor similarity estimates

### 5.2 Cold Start Problem
- **New users**: No rating history for recommendations
- **New items**: No ratings to base recommendations on
- **System cold start**: Insufficient data for any user/item

### 5.3 Scalability Issues
- O(m²) or O(n²) similarity computations
- Memory requirements for similarity matrices
- Real-time recommendation challenges

### 5.4 Data Quality Issues
- **Rating bias**: Different users use scales differently
- **Popularity bias**: Popular items get more attention
- **Temporal effects**: User preferences change over time

## 6. Study Questions

### Basic Level
1. What is the fundamental assumption behind collaborative filtering?
2. Explain the difference between user-based and item-based collaborative filtering.
3. Why is the sparsity problem particularly challenging for collaborative filtering?
4. What role does the similarity metric play in CF performance?

### Intermediate Level  
5. Implement a hybrid approach that combines user-based and item-based predictions.
6. How would you handle the case where a user has very few neighbors with high similarity?
7. Design a method to detect and handle rating spam in collaborative filtering.
8. Compare the computational complexity of user-based vs item-based approaches.

### Advanced Level
9. Implement a temporal collaborative filtering system that accounts for time-based preference changes.
10. Design a distributed collaborative filtering system that can handle millions of users.
11. How would you incorporate implicit feedback (clicks, views) into traditional CF?
12. Implement confidence-weighted collaborative filtering for implicit feedback data.

### Tricky Questions
13. A new user rates only very popular items highly. How would this affect recommendations from user-based CF?
14. If most similar users for a target user have very few ratings, how would you improve prediction quality?
15. Design a collaborative filtering system that works well when 99% of the rating matrix is empty.
16. How would you detect whether user-based or item-based CF would work better for a given dataset without implementing both?

## 7. Key Takeaways

1. **Collaborative filtering relies on community wisdom** - similar users/items provide recommendation signals
2. **Similarity computation is the heart** of neighborhood-based methods
3. **Sparsity and cold-start are fundamental challenges** requiring careful handling
4. **User-based vs item-based choice depends** on data characteristics and use case
5. **Bias correction and normalization** significantly improve prediction quality
6. **Scalability requires algorithmic and architectural** optimizations

## Next Session Preview
In the next session, we'll dive deep into user-based collaborative filtering, exploring advanced similarity measures, neighborhood selection strategies, and optimization techniques for large-scale implementations.