# Day 3.1: Introduction to User-Item Matrices

## Learning Objectives
By the end of this session, you will:
- Understand the fundamental concept of user-item matrices in recommendation systems
- Learn different types of feedback data (explicit vs implicit)
- Master the mathematical representation of user preferences
- Implement basic user-item matrix operations
- Recognize the importance of data preprocessing in recommendation systems

## 1. What is a User-Item Matrix?

A user-item matrix (also called a utility matrix) is the fundamental data structure in collaborative filtering recommendation systems. It represents the relationship between users and items through ratings, purchases, clicks, or other interaction data.

### Mathematical Definition
Let U = {u₁, u₂, ..., uₘ} be a set of m users and I = {i₁, i₂, ..., iₙ} be a set of n items.

The user-item matrix R is an m × n matrix where:
- R[u,i] represents the rating/preference of user u for item i
- R[u,i] = 0 or null indicates no interaction between user u and item i

```
       Item1  Item2  Item3  Item4  Item5
User1    5      3      0      1      4
User2    4      0      2      5      0
User3    0      4      3      0      2
User4    2      5      4      3      0
```

## 2. Types of Feedback Data

### 2.1 Explicit Feedback
Direct user ratings or preferences:
- **Rating scales**: 1-5 stars, 1-10 points, thumbs up/down
- **Reviews**: Text with numerical ratings
- **Preferences**: Like/dislike, favorite/not favorite

**Advantages:**
- Clear indication of user preference intensity
- High quality signal for recommendation
- Easy to interpret and use

**Disadvantages:**
- Sparse data (users rate few items)
- Rating bias (different users use scales differently)
- Requires user effort to provide ratings

### 2.2 Implicit Feedback
Indirect signals of user preferences:
- **Behavioral data**: Clicks, views, time spent, purchases
- **Usage patterns**: Download, share, bookmark
- **Contextual signals**: Search queries, navigation paths

**Advantages:**
- Abundant data available
- No additional user effort required
- Reflects actual user behavior

**Disadvantages:**
- Ambiguous preference indication
- Noisy data (accidental clicks, negative feedback)
- Binary nature (hard to quantify intensity)

## 3. Implementation: Basic User-Item Matrix Operations

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class UserItemMatrix:
    """
    A comprehensive user-item matrix implementation supporting both
    explicit and implicit feedback with various utility functions.
    """
    
    def __init__(self, data_type: str = 'explicit'):
        """
        Initialize user-item matrix.
        
        Args:
            data_type: 'explicit' for ratings, 'implicit' for binary interactions
        """
        self.data_type = data_type
        self.matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        
    def fit(self, interactions: List[Tuple], users: List = None, items: List = None):
        """
        Build user-item matrix from interaction data.
        
        Args:
            interactions: List of (user_id, item_id, rating/interaction) tuples
            users: Optional list of all users (for handling cold users)
            items: Optional list of all items (for handling cold items)
        """
        # Extract unique users and items
        unique_users = set([interaction[0] for interaction in interactions])
        unique_items = set([interaction[1] for interaction in interactions])
        
        # Add cold users/items if provided
        if users:
            unique_users.update(users)
        if items:
            unique_items.update(items)
            
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(unique_users))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(unique_items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        # Initialize matrix
        self.matrix = np.zeros((self.n_users, self.n_items))
        
        # Fill matrix with interactions
        for user_id, item_id, rating in interactions:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            if self.data_type == 'explicit':
                self.matrix[user_idx, item_idx] = rating
            else:  # implicit
                self.matrix[user_idx, item_idx] = 1
                
        print(f"Created {self.n_users} x {self.n_items} user-item matrix")
        print(f"Sparsity: {self.get_sparsity():.2%}")
        
    def get_sparsity(self) -> float:
        """Calculate matrix sparsity (percentage of zero entries)."""
        if self.matrix is None:
            return 0.0
        
        total_entries = self.matrix.size
        non_zero_entries = np.count_nonzero(self.matrix)
        return 1 - (non_zero_entries / total_entries)
    
    def get_user_ratings(self, user_id) -> np.ndarray:
        """Get all ratings for a specific user."""
        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} not found in matrix")
        
        user_idx = self.user_to_idx[user_id]
        return self.matrix[user_idx, :]
    
    def get_item_ratings(self, item_id) -> np.ndarray:
        """Get all ratings for a specific item."""
        if item_id not in self.item_to_idx:
            raise ValueError(f"Item {item_id} not found in matrix")
        
        item_idx = self.item_to_idx[item_id]
        return self.matrix[:, item_idx]
    
    def get_user_mean_rating(self, user_id) -> float:
        """Calculate mean rating for a user (excluding zeros)."""
        ratings = self.get_user_ratings(user_id)
        non_zero_ratings = ratings[ratings > 0]
        
        if len(non_zero_ratings) == 0:
            return 0.0
        
        return np.mean(non_zero_ratings)
    
    def get_item_mean_rating(self, item_id) -> float:
        """Calculate mean rating for an item (excluding zeros)."""
        ratings = self.get_item_ratings(item_id)
        non_zero_ratings = ratings[ratings > 0]
        
        if len(non_zero_ratings) == 0:
            return 0.0
        
        return np.mean(non_zero_ratings)
    
    def get_user_item_count(self, user_id) -> int:
        """Get number of items rated by user."""
        ratings = self.get_user_ratings(user_id)
        return np.count_nonzero(ratings)
    
    def get_item_user_count(self, item_id) -> int:
        """Get number of users who rated item."""
        ratings = self.get_item_ratings(item_id)
        return np.count_nonzero(ratings)
    
    def normalize_ratings(self, method: str = 'user_mean'):
        """
        Normalize ratings using various methods.
        
        Args:
            method: 'user_mean', 'item_mean', 'global_mean', 'z_score'
        """
        if self.matrix is None:
            raise ValueError("Matrix not initialized. Call fit() first.")
        
        normalized_matrix = self.matrix.copy()
        
        if method == 'user_mean':
            # Subtract user mean from each user's ratings
            for user_idx in range(self.n_users):
                user_ratings = normalized_matrix[user_idx, :]
                non_zero_mask = user_ratings > 0
                
                if np.any(non_zero_mask):
                    user_mean = np.mean(user_ratings[non_zero_mask])
                    normalized_matrix[user_idx, non_zero_mask] -= user_mean
                    
        elif method == 'item_mean':
            # Subtract item mean from each item's ratings
            for item_idx in range(self.n_items):
                item_ratings = normalized_matrix[:, item_idx]
                non_zero_mask = item_ratings > 0
                
                if np.any(non_zero_mask):
                    item_mean = np.mean(item_ratings[non_zero_mask])
                    normalized_matrix[non_zero_mask, item_idx] -= item_mean
                    
        elif method == 'global_mean':
            # Subtract global mean from all non-zero ratings
            non_zero_ratings = normalized_matrix[normalized_matrix > 0]
            if len(non_zero_ratings) > 0:
                global_mean = np.mean(non_zero_ratings)
                mask = normalized_matrix > 0
                normalized_matrix[mask] -= global_mean
                
        elif method == 'z_score':
            # Z-score normalization per user
            for user_idx in range(self.n_users):
                user_ratings = normalized_matrix[user_idx, :]
                non_zero_mask = user_ratings > 0
                
                if np.any(non_zero_mask):
                    user_mean = np.mean(user_ratings[non_zero_mask])
                    user_std = np.std(user_ratings[non_zero_mask])
                    
                    if user_std > 0:
                        normalized_matrix[user_idx, non_zero_mask] = \
                            (user_ratings[non_zero_mask] - user_mean) / user_std
        
        return normalized_matrix
    
    def to_sparse(self):
        """Convert to sparse matrix representation for memory efficiency."""
        if self.matrix is None:
            raise ValueError("Matrix not initialized. Call fit() first.")
        
        return csr_matrix(self.matrix)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive matrix statistics."""
        if self.matrix is None:
            return {}
        
        non_zero_ratings = self.matrix[self.matrix > 0]
        
        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'total_entries': self.matrix.size,
            'non_zero_entries': len(non_zero_ratings),
            'sparsity': self.get_sparsity(),
            'density': 1 - self.get_sparsity(),
            'min_rating': np.min(non_zero_ratings) if len(non_zero_ratings) > 0 else 0,
            'max_rating': np.max(non_zero_ratings) if len(non_zero_ratings) > 0 else 0,
            'mean_rating': np.mean(non_zero_ratings) if len(non_zero_ratings) > 0 else 0,
            'std_rating': np.std(non_zero_ratings) if len(non_zero_ratings) > 0 else 0,
        }
        
        # User statistics
        user_item_counts = [self.get_user_item_count(user) for user in self.user_to_idx.keys()]
        stats.update({
            'avg_items_per_user': np.mean(user_item_counts),
            'std_items_per_user': np.std(user_item_counts),
            'min_items_per_user': np.min(user_item_counts),
            'max_items_per_user': np.max(user_item_counts),
        })
        
        # Item statistics
        item_user_counts = [self.get_item_user_count(item) for item in self.item_to_idx.keys()]
        stats.update({
            'avg_users_per_item': np.mean(item_user_counts),
            'std_users_per_item': np.std(item_user_counts),
            'min_users_per_item': np.min(item_user_counts),
            'max_users_per_item': np.max(item_user_counts),
        })
        
        return stats
    
    def visualize_matrix(self, sample_size: int = 100):
        """Visualize a sample of the user-item matrix."""
        if self.matrix is None:
            raise ValueError("Matrix not initialized. Call fit() first.")
        
        # Sample users and items for visualization
        n_sample_users = min(sample_size, self.n_users)
        n_sample_items = min(sample_size, self.n_items)
        
        user_indices = np.random.choice(self.n_users, n_sample_users, replace=False)
        item_indices = np.random.choice(self.n_items, n_sample_items, replace=False)
        
        sample_matrix = self.matrix[np.ix_(user_indices, item_indices)]
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(sample_matrix, cmap='viridis', cbar=True)
        plt.title(f'User-Item Matrix Sample ({n_sample_users}x{n_sample_items})')
        plt.xlabel('Items')
        plt.ylabel('Users')
        
        # Rating distribution
        plt.subplot(2, 2, 2)
        non_zero_ratings = self.matrix[self.matrix > 0]
        plt.hist(non_zero_ratings, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        # Items per user distribution
        plt.subplot(2, 2, 3)
        user_item_counts = [self.get_user_item_count(user) for user in self.user_to_idx.keys()]
        plt.hist(user_item_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Items per User Distribution')
        plt.xlabel('Number of Items Rated')
        plt.ylabel('Number of Users')
        
        # Users per item distribution
        plt.subplot(2, 2, 4)
        item_user_counts = [self.get_item_user_count(item) for item in self.item_to_idx.keys()]
        plt.hist(item_user_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Users per Item Distribution')
        plt.xlabel('Number of Users')
        plt.ylabel('Number of Items')
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def create_sample_data():
    """Create sample movie rating data for testing."""
    np.random.seed(42)
    
    users = [f'user_{i}' for i in range(1, 101)]  # 100 users
    movies = [f'movie_{i}' for i in range(1, 51)]  # 50 movies
    
    interactions = []
    
    # Generate ratings with some patterns
    for user in users:
        # Each user rates 5-20 movies
        n_ratings = np.random.randint(5, 21)
        rated_movies = np.random.choice(movies, n_ratings, replace=False)
        
        # User preference bias (some users like higher ratings)
        user_bias = np.random.normal(0, 0.5)
        
        for movie in rated_movies:
            # Base rating from normal distribution
            base_rating = np.random.normal(3, 1)
            final_rating = np.clip(base_rating + user_bias, 1, 5)
            interactions.append((user, movie, round(final_rating, 1)))
    
    return interactions, users, movies

# Test the implementation
if __name__ == "__main__":
    # Create sample data
    interactions, users, movies = create_sample_data()
    
    print("Sample interactions:")
    for i in range(5):
        print(f"  {interactions[i]}")
    
    # Create and fit user-item matrix
    matrix = UserItemMatrix(data_type='explicit')
    matrix.fit(interactions, users, movies)
    
    # Get statistics
    stats = matrix.get_statistics()
    print("\nMatrix Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test specific operations
    print(f"\nUser 'user_1' mean rating: {matrix.get_user_mean_rating('user_1'):.2f}")
    print(f"Movie 'movie_1' mean rating: {matrix.get_item_mean_rating('movie_1'):.2f}")
    print(f"User 'user_1' rated {matrix.get_user_item_count('user_1')} movies")
    print(f"Movie 'movie_1' rated by {matrix.get_item_user_count('movie_1')} users")
    
    # Test normalization
    normalized = matrix.normalize_ratings('user_mean')
    print(f"\nAfter user-mean normalization:")
    print(f"  Min rating: {np.min(normalized[normalized != 0]):.3f}")
    print(f"  Max rating: {np.max(normalized):.3f}")
    print(f"  Mean rating: {np.mean(normalized[normalized != 0]):.3f}")
```

## 4. Data Preprocessing Considerations

### 4.1 Handling Missing Data
```python
def preprocess_ratings(matrix: UserItemMatrix, min_user_ratings: int = 5, 
                      min_item_ratings: int = 5):
    """
    Preprocess user-item matrix by filtering out users/items with insufficient data.
    
    Args:
        matrix: UserItemMatrix object
        min_user_ratings: Minimum number of ratings per user
        min_item_ratings: Minimum number of ratings per item
    """
    if matrix.matrix is None:
        raise ValueError("Matrix not initialized")
    
    # Iteratively remove users/items until criteria met
    changed = True
    iteration = 0
    
    while changed and iteration < 10:  # Prevent infinite loops
        changed = False
        iteration += 1
        
        # Remove users with too few ratings
        users_to_remove = []
        for user_id in matrix.user_to_idx.keys():
            if matrix.get_user_item_count(user_id) < min_user_ratings:
                users_to_remove.append(user_id)
        
        # Remove items with too few ratings
        items_to_remove = []
        for item_id in matrix.item_to_idx.keys():
            if matrix.get_item_user_count(item_id) < min_item_ratings:
                items_to_remove.append(item_id)
        
        if users_to_remove or items_to_remove:
            changed = True
            # Create new interactions without removed users/items
            # Implementation would involve rebuilding the matrix
            print(f"Iteration {iteration}: Removing {len(users_to_remove)} users, "
                  f"{len(items_to_remove)} items")
    
    return matrix
```

### 4.2 Rating Scale Normalization
Different users may use rating scales differently:
- **Conservative users**: Use only 3-4 on a 5-point scale
- **Liberal users**: Use full 1-5 range
- **Extreme users**: Use only 1 and 5

## 5. Study Questions

### Basic Level
1. What is the difference between explicit and implicit feedback?
2. Why are user-item matrices typically sparse?
3. How would you handle a new user who hasn't rated any items?
4. What are the advantages and disadvantages of explicit vs implicit feedback?

### Intermediate Level
5. Implement a function to convert implicit feedback (clicks) to explicit ratings using click frequency.
6. How would you handle the case where a user rates the same item multiple times?
7. Design a preprocessing pipeline for a real-world e-commerce dataset.
8. What normalization method would you choose for a dataset with high rating variance between users?

### Advanced Level
9. How would you efficiently store and query a user-item matrix with millions of users and items?
10. Design a system to handle both explicit ratings and implicit feedback simultaneously.
11. Implement a method to detect and handle rating spam or fake reviews in the matrix.
12. How would you adapt the user-item matrix concept for temporal recommendations (time-aware systems)?

### Tricky Questions
13. A user gives a 1-star rating. In implicit feedback, they click on an item but immediately leave. How do these scenarios differ in terms of preference indication?
14. You have a user-item matrix where 99% of entries are missing. Is this matrix still useful for recommendations? Justify your answer.
15. Design a method to handle the case where item features change over time (e.g., a software application gets updated features).
16. How would you create a user-item matrix for a multi-stakeholder recommendation system (e.g., recommendations for families where multiple people use the same account)?

## 6. Key Takeaways

1. **User-item matrices are the foundation** of collaborative filtering systems
2. **Sparsity is the biggest challenge** - most entries are missing
3. **Data preprocessing is crucial** for system performance
4. **Different feedback types** require different handling strategies
5. **Normalization helps** handle user rating biases
6. **Storage efficiency** becomes critical at scale

## Next Session Preview
In the next session, we'll dive deeper into user-item matrix representations, explore advanced sparsity handling techniques, and learn about efficient storage strategies for large-scale systems.