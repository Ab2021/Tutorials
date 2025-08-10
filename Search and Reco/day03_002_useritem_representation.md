# Day 3.2: User-Item Matrix Representations and Sparsity

## Learning Objectives
By the end of this session, you will:
- Master different representations of user-item matrices
- Understand the sparsity problem and its implications
- Learn efficient storage and computation techniques for sparse matrices
- Implement advanced sparsity handling methods
- Optimize memory usage for large-scale recommendation systems

## 1. Matrix Representation Formats

### 1.1 Dense Matrix Representation
Traditional numpy arrays where every entry is stored in memory.

**Advantages:**
- Simple indexing: O(1) access time
- Standard linear algebra operations
- Easy to understand and debug

**Disadvantages:**
- Memory inefficient for sparse data
- Scales poorly: O(mn) memory for m users, n items
- Wastes computation on zero entries

### 1.2 Sparse Matrix Representations

#### Coordinate Format (COO)
Stores (row, col, value) triplets for non-zero entries.

```python
# COO Format Example
rows = [0, 0, 1, 2, 2, 2]
cols = [1, 3, 2, 0, 1, 3]
data = [5, 4, 3, 2, 1, 4]
# Represents:
# [[0, 5, 0, 4],
#  [0, 0, 3, 0],
#  [2, 1, 0, 4]]
```

#### Compressed Sparse Row (CSR)
Efficient for row-wise operations (common in user-based CF).

```python
# CSR Format: indptr, indices, data arrays
# indptr[i] points to start of row i in indices/data arrays
indptr = [0, 2, 3, 6]  # Row pointers
indices = [1, 3, 2, 0, 1, 3]  # Column indices
data = [5, 4, 3, 2, 1, 4]  # Values
```

#### Compressed Sparse Column (CSC)
Efficient for column-wise operations (common in item-based CF).

## 2. Implementation: Advanced User-Item Matrix with Sparsity Handling

```python
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, Union
import pickle
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class SparseUserItemMatrix:
    """
    Advanced user-item matrix with efficient sparse representations
    and comprehensive sparsity handling capabilities.
    """
    
    def __init__(self, format_type: str = 'csr', dtype=np.float32):
        """
        Initialize sparse user-item matrix.
        
        Args:
            format_type: 'csr', 'csc', or 'coo' for different sparse formats
            dtype: Data type for matrix values (float32 saves memory)
        """
        self.format_type = format_type
        self.dtype = dtype
        self.matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.n_users = 0
        self.n_items = 0
        self.global_mean = 0.0
        self.user_means = None
        self.item_means = None
        
    def fit(self, interactions: List[Tuple], 
            min_user_interactions: int = 5,
            min_item_interactions: int = 5,
            filter_iterations: int = 3):
        """
        Build sparse user-item matrix with iterative filtering.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            filter_iterations: Number of filtering iterations
        """
        print("Building user-item matrix...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
        print(f"Initial data: {len(df)} interactions, "
              f"{df['user_id'].nunique()} users, "
              f"{df['item_id'].nunique()} items")
        
        # Iterative filtering to remove sparse users/items
        for iteration in range(filter_iterations):
            print(f"Filtering iteration {iteration + 1}...")
            
            # Count interactions per user and item
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()
            
            # Filter users and items
            valid_users = user_counts[user_counts >= min_user_interactions].index
            valid_items = item_counts[item_counts >= min_item_interactions].index
            
            # Keep only valid interactions
            df_filtered = df[
                (df['user_id'].isin(valid_users)) & 
                (df['item_id'].isin(valid_items))
            ].copy()
            
            print(f"  After filtering: {len(df_filtered)} interactions, "
                  f"{df_filtered['user_id'].nunique()} users, "
                  f"{df_filtered['item_id'].nunique()} items")
            
            # Check if we removed anything
            if len(df_filtered) == len(df):
                print(f"  No change in iteration {iteration + 1}, stopping early")
                break
                
            df = df_filtered
        
        # Create mappings
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        # Convert to matrix indices
        row_indices = [self.user_to_idx[user] for user in df['user_id']]
        col_indices = [self.item_to_idx[item] for item in df['item_id']]
        data = df['rating'].astype(self.dtype).values
        
        # Create sparse matrix
        coo = coo_matrix((data, (row_indices, col_indices)), 
                        shape=(self.n_users, self.n_items))
        
        # Convert to desired format
        if self.format_type == 'csr':
            self.matrix = coo.tocsr()
        elif self.format_type == 'csc':
            self.matrix = coo.tocsc()
        else:
            self.matrix = coo
        
        # Calculate statistics
        self._calculate_statistics()
        
        print(f"Final matrix: {self.n_users} x {self.n_items}, "
              f"density: {self.get_density():.4f}")
    
    def _calculate_statistics(self):
        """Calculate and cache matrix statistics."""
        if self.matrix is None:
            return
        
        # Global mean (excluding zeros)
        self.global_mean = self.matrix.data.mean()
        
        # User means
        self.user_means = np.zeros(self.n_users)
        for i in range(self.n_users):
            user_ratings = self.matrix.getrow(i).data
            if len(user_ratings) > 0:
                self.user_means[i] = user_ratings.mean()
        
        # Item means  
        self.item_means = np.zeros(self.n_items)
        for j in range(self.n_items):
            item_ratings = self.matrix.getcol(j).data
            if len(item_ratings) > 0:
                self.item_means[j] = item_ratings.mean()
    
    def get_density(self) -> float:
        """Calculate matrix density (percentage of non-zero entries)."""
        if self.matrix is None:
            return 0.0
        return self.matrix.nnz / (self.n_users * self.n_items)
    
    def get_sparsity(self) -> float:
        """Calculate matrix sparsity (percentage of zero entries)."""
        return 1 - self.get_density()
    
    def get_user_vector(self, user_id, format_type: str = None) -> Union[np.ndarray, sparse.spmatrix]:
        """
        Get user's rating vector.
        
        Args:
            user_id: User identifier
            format_type: Return format ('dense', 'sparse', or None for matrix format)
        """
        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} not found")
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.matrix.getrow(user_idx)
        
        if format_type == 'dense':
            return user_vector.toarray().flatten()
        elif format_type == 'sparse':
            return user_vector
        else:
            return user_vector if self.format_type in ['csr', 'coo'] else user_vector.toarray().flatten()
    
    def get_item_vector(self, item_id, format_type: str = None) -> Union[np.ndarray, sparse.spmatrix]:
        """
        Get item's rating vector.
        
        Args:
            item_id: Item identifier  
            format_type: Return format ('dense', 'sparse', or None for matrix format)
        """
        if item_id not in self.item_to_idx:
            raise ValueError(f"Item {item_id} not found")
        
        item_idx = self.item_to_idx[item_id]
        item_vector = self.matrix.getcol(item_idx)
        
        if format_type == 'dense':
            return item_vector.toarray().flatten()
        elif format_type == 'sparse':
            return item_vector
        else:
            return item_vector if self.format_type in ['csc', 'coo'] else item_vector.toarray().flatten()
    
    def get_rating(self, user_id, item_id) -> float:
        """Get specific user-item rating."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        return self.matrix[user_idx, item_idx]
    
    def set_rating(self, user_id, item_id, rating: float):
        """Set specific user-item rating."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            raise ValueError("User or item not found in matrix")
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Convert to lil_matrix for efficient single element updates
        if not isinstance(self.matrix, sparse.lil_matrix):
            original_format = type(self.matrix)
            self.matrix = self.matrix.tolil()
        else:
            original_format = None
        
        self.matrix[user_idx, item_idx] = rating
        
        # Convert back to original format if needed
        if original_format is not None:
            if self.format_type == 'csr':
                self.matrix = self.matrix.tocsr()
            elif self.format_type == 'csc':
                self.matrix = self.matrix.tocsc()
            elif self.format_type == 'coo':
                self.matrix = self.matrix.tocoo()
    
    def get_nonzero_users_for_item(self, item_id) -> List[Tuple[str, float]]:
        """Get all users who rated a specific item."""
        if item_id not in self.item_to_idx:
            return []
        
        item_idx = self.item_to_idx[item_id]
        item_col = self.matrix.getcol(item_idx).tocoo()
        
        users_ratings = []
        for user_idx, rating in zip(item_col.row, item_col.data):
            user_id = self.idx_to_user[user_idx]
            users_ratings.append((user_id, rating))
        
        return users_ratings
    
    def get_nonzero_items_for_user(self, user_id) -> List[Tuple[str, float]]:
        """Get all items rated by a specific user."""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_row = self.matrix.getrow(user_idx).tocoo()
        
        items_ratings = []
        for item_idx, rating in zip(user_row.col, user_row.data):
            item_id = self.idx_to_item[item_idx]
            items_ratings.append((item_id, rating))
        
        return items_ratings
    
    def compute_user_similarity(self, user1_id, user2_id, metric: str = 'cosine') -> float:
        """
        Compute similarity between two users.
        
        Args:
            user1_id, user2_id: User identifiers
            metric: 'cosine', 'pearson', or 'jaccard'
        """
        if user1_id not in self.user_to_idx or user2_id not in self.user_to_idx:
            return 0.0
        
        user1_vector = self.get_user_vector(user1_id, 'sparse')
        user2_vector = self.get_user_vector(user2_id, 'sparse')
        
        if metric == 'cosine':
            similarity = cosine_similarity(user1_vector, user2_vector)[0, 0]
            return similarity if not np.isnan(similarity) else 0.0
        
        elif metric == 'pearson':
            # Convert to dense for easier computation
            u1_dense = user1_vector.toarray().flatten()
            u2_dense = user2_vector.toarray().flatten()
            
            # Find common non-zero items
            common_mask = (u1_dense > 0) & (u2_dense > 0)
            
            if np.sum(common_mask) < 2:
                return 0.0
            
            u1_common = u1_dense[common_mask]
            u2_common = u2_dense[common_mask]
            
            # Pearson correlation
            correlation = np.corrcoef(u1_common, u2_common)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        elif metric == 'jaccard':
            # Jaccard similarity for binary data
            u1_dense = user1_vector.toarray().flatten()
            u2_dense = user2_vector.toarray().flatten()
            
            u1_binary = (u1_dense > 0).astype(int)
            u2_binary = (u2_dense > 0).astype(int)
            
            intersection = np.sum(u1_binary & u2_binary)
            union = np.sum(u1_binary | u2_binary)
            
            return intersection / union if union > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def normalize_matrix(self, method: str = 'user_mean') -> 'SparseUserItemMatrix':
        """
        Create normalized version of the matrix.
        
        Args:
            method: 'user_mean', 'item_mean', 'global_mean', 'z_score'
        """
        if self.matrix is None:
            raise ValueError("Matrix not initialized")
        
        # Convert to COO for easy manipulation
        coo = self.matrix.tocoo()
        normalized_data = coo.data.copy()
        
        if method == 'user_mean':
            for i, (row, col, rating) in enumerate(zip(coo.row, coo.col, coo.data)):
                user_mean = self.user_means[row]
                normalized_data[i] = rating - user_mean
                
        elif method == 'item_mean':
            for i, (row, col, rating) in enumerate(zip(coo.row, coo.col, coo.data)):
                item_mean = self.item_means[col]
                normalized_data[i] = rating - item_mean
                
        elif method == 'global_mean':
            normalized_data = normalized_data - self.global_mean
            
        elif method == 'z_score':
            for i, (row, col, rating) in enumerate(zip(coo.row, coo.col, coo.data)):
                user_ratings = self.matrix.getrow(row).data
                if len(user_ratings) > 1:
                    user_mean = user_ratings.mean()
                    user_std = user_ratings.std()
                    if user_std > 0:
                        normalized_data[i] = (rating - user_mean) / user_std
                    else:
                        normalized_data[i] = 0
                else:
                    normalized_data[i] = 0
        
        # Create new matrix object
        normalized_coo = coo_matrix((normalized_data, (coo.row, coo.col)), 
                                   shape=coo.shape)
        
        # Create new SparseUserItemMatrix object
        result = SparseUserItemMatrix(self.format_type, self.dtype)
        result.user_to_idx = self.user_to_idx.copy()
        result.idx_to_user = self.idx_to_user.copy()
        result.item_to_idx = self.item_to_idx.copy()
        result.idx_to_item = self.idx_to_item.copy()
        result.n_users = self.n_users
        result.n_items = self.n_items
        
        # Convert to desired format
        if self.format_type == 'csr':
            result.matrix = normalized_coo.tocsr()
        elif self.format_type == 'csc':
            result.matrix = normalized_coo.tocsc()
        else:
            result.matrix = normalized_coo
        
        result._calculate_statistics()
        
        return result
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        if self.matrix is None:
            return {}
        
        # Matrix memory usage
        matrix_memory = self.matrix.data.nbytes + self.matrix.indices.nbytes
        if hasattr(self.matrix, 'indptr'):
            matrix_memory += self.matrix.indptr.nbytes
        elif hasattr(self.matrix, 'row') and hasattr(self.matrix, 'col'):
            matrix_memory += self.matrix.row.nbytes + self.matrix.col.nbytes
        
        # Mapping memory usage
        mapping_memory = (
            len(pickle.dumps(self.user_to_idx)) +
            len(pickle.dumps(self.idx_to_user)) +
            len(pickle.dumps(self.item_to_idx)) +
            len(pickle.dumps(self.idx_to_item))
        )
        
        # Statistics memory
        stats_memory = 0
        if self.user_means is not None:
            stats_memory += self.user_means.nbytes
        if self.item_means is not None:
            stats_memory += self.item_means.nbytes
        
        # Dense equivalent memory
        dense_memory = self.n_users * self.n_items * np.dtype(self.dtype).itemsize
        
        total_memory = matrix_memory + mapping_memory + stats_memory
        
        return {
            'matrix_mb': matrix_memory / (1024 * 1024),
            'mappings_mb': mapping_memory / (1024 * 1024),
            'statistics_mb': stats_memory / (1024 * 1024),
            'total_mb': total_memory / (1024 * 1024),
            'dense_equivalent_mb': dense_memory / (1024 * 1024),
            'compression_ratio': dense_memory / total_memory if total_memory > 0 else 0,
        }
    
    def analyze_sparsity_patterns(self) -> Dict:
        """Analyze sparsity patterns in the matrix."""
        if self.matrix is None:
            return {}
        
        coo = self.matrix.tocoo()
        
        # User interaction counts
        user_interactions = np.bincount(coo.row, minlength=self.n_users)
        
        # Item interaction counts  
        item_interactions = np.bincount(coo.col, minlength=self.n_items)
        
        # Rating distribution
        rating_dist = np.histogram(coo.data, bins=20)
        
        analysis = {
            'total_interactions': len(coo.data),
            'density': self.get_density(),
            'sparsity': self.get_sparsity(),
            
            # User patterns
            'avg_interactions_per_user': np.mean(user_interactions),
            'std_interactions_per_user': np.std(user_interactions),
            'min_interactions_per_user': np.min(user_interactions),
            'max_interactions_per_user': np.max(user_interactions),
            'users_with_single_interaction': np.sum(user_interactions == 1),
            
            # Item patterns
            'avg_interactions_per_item': np.mean(item_interactions),
            'std_interactions_per_item': np.std(item_interactions),
            'min_interactions_per_item': np.min(item_interactions),
            'max_interactions_per_item': np.max(item_interactions),
            'items_with_single_interaction': np.sum(item_interactions == 1),
            
            # Rating patterns
            'mean_rating': np.mean(coo.data),
            'std_rating': np.std(coo.data),
            'min_rating': np.min(coo.data),
            'max_rating': np.max(coo.data),
            'rating_distribution': rating_dist,
        }
        
        return analysis
    
    def save(self, filepath: str):
        """Save the matrix to disk."""
        save_data = {
            'matrix': self.matrix,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'format_type': self.format_type,
            'dtype': self.dtype,
            'global_mean': self.global_mean,
            'user_means': self.user_means,
            'item_means': self.item_means,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """Load the matrix from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.matrix = data['matrix']
        self.user_to_idx = data['user_to_idx']
        self.idx_to_user = data['idx_to_user']
        self.item_to_idx = data['item_to_idx']
        self.idx_to_item = data['idx_to_item']
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.format_type = data['format_type']
        self.dtype = data['dtype']
        self.global_mean = data['global_mean']
        self.user_means = data['user_means']
        self.item_means = data['item_means']

# Specialized sparse matrix operations
class SparseMatrixOperations:
    """Utility class for efficient sparse matrix operations."""
    
    @staticmethod
    def efficient_user_similarity_matrix(matrix: SparseUserItemMatrix, 
                                       metric: str = 'cosine',
                                       min_common_items: int = 5) -> sparse.csr_matrix:
        """
        Compute user-user similarity matrix efficiently.
        
        Args:
            matrix: SparseUserItemMatrix object
            metric: Similarity metric
            min_common_items: Minimum common items required for similarity computation
        """
        if metric == 'cosine':
            # Normalize user vectors
            normalized = matrix.matrix.copy().astype(np.float64)
            
            # L2 normalize each row
            row_norms = np.array(normalized.multiply(normalized).sum(axis=1)).flatten()
            row_norms = np.sqrt(row_norms)
            row_norms[row_norms == 0] = 1  # Avoid division by zero
            
            # Create diagonal matrix for normalization
            norm_matrix = sparse.diags(1.0 / row_norms)
            normalized = norm_matrix @ normalized
            
            # Compute similarity matrix
            similarity_matrix = normalized @ normalized.T
            
            # Zero out similarities with insufficient common items
            if min_common_items > 1:
                # Compute common items matrix
                binary_matrix = (matrix.matrix > 0).astype(np.int32)
                common_items = binary_matrix @ binary_matrix.T
                
                # Mask insufficient similarities
                mask = common_items < min_common_items
                similarity_matrix[mask] = 0
            
            # Remove self-similarities
            similarity_matrix.setdiag(0)
            
            return similarity_matrix.tocsr()
        
        else:
            raise NotImplementedError(f"Metric {metric} not implemented for batch computation")
    
    @staticmethod
    def efficient_item_similarity_matrix(matrix: SparseUserItemMatrix,
                                       metric: str = 'cosine',
                                       min_common_users: int = 5) -> sparse.csr_matrix:
        """
        Compute item-item similarity matrix efficiently.
        
        Args:
            matrix: SparseUserItemMatrix object  
            metric: Similarity metric
            min_common_users: Minimum common users required for similarity computation
        """
        if metric == 'cosine':
            # Transpose for item-item computation
            item_matrix = matrix.matrix.T.copy().astype(np.float64)
            
            # L2 normalize each row (item)
            row_norms = np.array(item_matrix.multiply(item_matrix).sum(axis=1)).flatten()
            row_norms = np.sqrt(row_norms)
            row_norms[row_norms == 0] = 1  # Avoid division by zero
            
            # Create diagonal matrix for normalization
            norm_matrix = sparse.diags(1.0 / row_norms)
            normalized = norm_matrix @ item_matrix
            
            # Compute similarity matrix
            similarity_matrix = normalized @ normalized.T
            
            # Zero out similarities with insufficient common users
            if min_common_users > 1:
                # Compute common users matrix
                binary_matrix = (item_matrix > 0).astype(np.int32)
                common_users = binary_matrix @ binary_matrix.T
                
                # Mask insufficient similarities
                mask = common_users < min_common_users
                similarity_matrix[mask] = 0
            
            # Remove self-similarities
            similarity_matrix.setdiag(0)
            
            return similarity_matrix.tocsr()
        
        else:
            raise NotImplementedError(f"Metric {metric} not implemented for batch computation")

# Example usage and performance comparison
def create_large_sparse_dataset(n_users: int = 10000, n_items: int = 5000, 
                               density: float = 0.001) -> List[Tuple]:
    """Create a large sparse dataset for testing."""
    np.random.seed(42)
    
    n_interactions = int(n_users * n_items * density)
    
    interactions = []
    for _ in range(n_interactions):
        user = f'user_{np.random.randint(0, n_users)}'
        item = f'item_{np.random.randint(0, n_items)}'
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
        interactions.append((user, item, rating))
    
    # Remove duplicates, keeping last rating
    interaction_dict = {}
    for user, item, rating in interactions:
        interaction_dict[(user, item)] = rating
    
    return [(user, item, rating) for (user, item), rating in interaction_dict.items()]

if __name__ == "__main__":
    # Create test dataset
    print("Creating large sparse dataset...")
    interactions = create_large_sparse_dataset(n_users=1000, n_items=500, density=0.01)
    
    # Test different formats
    formats = ['csr', 'csc', 'coo']
    matrices = {}
    
    for fmt in formats:
        print(f"\nTesting {fmt.upper()} format:")
        matrix = SparseUserItemMatrix(format_type=fmt)
        matrix.fit(interactions)
        matrices[fmt] = matrix
        
        # Memory usage
        memory = matrix.get_memory_usage()
        print(f"  Memory usage: {memory['total_mb']:.2f} MB")
        print(f"  Compression ratio: {memory['compression_ratio']:.1f}x")
        
        # Sparsity analysis
        analysis = matrix.analyze_sparsity_patterns()
        print(f"  Density: {analysis['density']:.4f}")
        print(f"  Avg interactions per user: {analysis['avg_interactions_per_user']:.1f}")
        print(f"  Avg interactions per item: {analysis['avg_interactions_per_item']:.1f}")
    
    # Test operations efficiency
    print("\nTesting operation efficiency:")
    matrix = matrices['csr']
    
    # User similarity computation
    ops = SparseMatrixOperations()
    similarity_matrix = ops.efficient_user_similarity_matrix(matrix)
    print(f"User similarity matrix shape: {similarity_matrix.shape}")
    print(f"User similarity matrix density: {similarity_matrix.nnz / similarity_matrix.size:.6f}")
```

## 3. Sparsity Mitigation Strategies

### 3.1 Matrix Factorization Preprocessing
```python
def implicit_feedback_weighting(matrix: SparseUserItemMatrix, 
                              alpha: float = 40.0) -> SparseUserItemMatrix:
    """
    Apply confidence weighting for implicit feedback data.
    Confidence = 1 + alpha * r_ui where r_ui is the raw preference value.
    """
    if matrix.matrix is None:
        raise ValueError("Matrix not initialized")
    
    # Convert to COO for manipulation
    coo = matrix.matrix.tocoo()
    
    # Apply confidence weighting: c_ui = 1 + alpha * r_ui
    confidence_data = 1 + alpha * coo.data
    
    # Create new matrix with confidence values
    confidence_coo = coo_matrix((confidence_data, (coo.row, coo.col)), 
                               shape=coo.shape)
    
    # Create result matrix
    result = SparseUserItemMatrix(matrix.format_type, matrix.dtype)
    result.user_to_idx = matrix.user_to_idx.copy()
    result.idx_to_user = matrix.idx_to_user.copy()
    result.item_to_idx = matrix.item_to_idx.copy()
    result.idx_to_item = matrix.idx_to_item.copy()
    result.n_users = matrix.n_users
    result.n_items = matrix.n_items
    
    # Convert to desired format
    if matrix.format_type == 'csr':
        result.matrix = confidence_coo.tocsr()
    elif matrix.format_type == 'csc':
        result.matrix = confidence_coo.tocsc()
    else:
        result.matrix = confidence_coo
    
    result._calculate_statistics()
    
    return result
```

### 3.2 Data Augmentation Techniques
```python
def augment_with_global_averages(matrix: SparseUserItemMatrix,
                               fill_ratio: float = 0.01) -> SparseUserItemMatrix:
    """
    Augment sparse matrix by filling some zero entries with global averages.
    
    Args:
        matrix: Input sparse matrix
        fill_ratio: Fraction of zero entries to fill with predicted values
    """
    if matrix.matrix is None:
        raise ValueError("Matrix not initialized")
    
    # Get current non-zero entries
    coo = matrix.matrix.tocoo()
    existing_entries = set(zip(coo.row, coo.col))
    
    # Generate potential new entries
    total_entries = matrix.n_users * matrix.n_items
    n_existing = len(existing_entries)
    n_zeros = total_entries - n_existing
    n_fill = int(n_zeros * fill_ratio)
    
    # Sample zero entries to fill
    new_entries = []
    attempts = 0
    max_attempts = n_fill * 10
    
    while len(new_entries) < n_fill and attempts < max_attempts:
        row = np.random.randint(0, matrix.n_users)
        col = np.random.randint(0, matrix.n_items)
        
        if (row, col) not in existing_entries:
            # Use user mean or item mean or global mean as prediction
            user_mean = matrix.user_means[row] if matrix.user_means[row] > 0 else matrix.global_mean
            item_mean = matrix.item_means[col] if matrix.item_means[col] > 0 else matrix.global_mean
            
            # Weighted average of user and item means
            predicted_rating = 0.6 * user_mean + 0.4 * item_mean
            
            if predicted_rating > 0:  # Only add positive predictions
                new_entries.append((row, col, predicted_rating))
                existing_entries.add((row, col))
        
        attempts += 1
    
    if new_entries:
        # Combine existing and new entries
        all_rows = list(coo.row) + [entry[0] for entry in new_entries]
        all_cols = list(coo.col) + [entry[1] for entry in new_entries]
        all_data = list(coo.data) + [entry[2] for entry in new_entries]
        
        # Create augmented matrix
        augmented_coo = coo_matrix((all_data, (all_rows, all_cols)), 
                                  shape=(matrix.n_users, matrix.n_items))
        
        # Create result matrix
        result = SparseUserItemMatrix(matrix.format_type, matrix.dtype)
        result.user_to_idx = matrix.user_to_idx.copy()
        result.idx_to_user = matrix.idx_to_user.copy()
        result.item_to_idx = matrix.item_to_idx.copy()
        result.idx_to_item = matrix.idx_to_item.copy()
        result.n_users = matrix.n_users
        result.n_items = matrix.n_items
        
        # Convert to desired format
        if matrix.format_type == 'csr':
            result.matrix = augmented_coo.tocsr()
        elif matrix.format_type == 'csc':
            result.matrix = augmented_coo.tocsc()
        else:
            result.matrix = augmented_coo
        
        result._calculate_statistics()
        
        print(f"Added {len(new_entries)} predicted entries "
              f"({len(new_entries)/n_zeros:.4f} of zero entries)")
        
        return result
    
    else:
        print("No entries were added during augmentation")
        return matrix
```

## 4. Performance Optimization Techniques

### 4.1 Block-wise Processing
For very large matrices, process in blocks to manage memory:

```python
def process_matrix_in_blocks(matrix: SparseUserItemMatrix, 
                           block_size: int = 1000,
                           operation: str = 'similarity'):
    """
    Process large matrix operations in blocks to manage memory.
    
    Args:
        matrix: Input matrix
        block_size: Size of each block
        operation: Type of operation ('similarity', 'normalization', etc.)
    """
    n_blocks = (matrix.n_users + block_size - 1) // block_size
    
    results = []
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, matrix.n_users)
        
        # Extract block
        block_matrix = matrix.matrix[start_idx:end_idx, :]
        
        if operation == 'similarity':
            # Compute similarities for this block
            block_similarities = cosine_similarity(block_matrix)
            results.append(block_similarities)
        
        print(f"Processed block {block_idx + 1}/{n_blocks}")
    
    return results
```

## 5. Study Questions

### Basic Level
1. What are the advantages of CSR format over COO format for user-based collaborative filtering?
2. How does matrix sparsity affect memory usage and computation time?
3. Why might you choose CSC format for item-based collaborative filtering?
4. What is the trade-off between storage efficiency and access speed in sparse matrices?

### Intermediate Level
5. Implement a function to convert between different sparse matrix formats efficiently.
6. Design a strategy to handle matrices where 99.9% of entries are zero.
7. How would you optimize similarity computation for a matrix with 1M users and 100K items?
8. Compare memory usage of different sparse formats for your specific use case.

### Advanced Level
9. Design a distributed storage system for user-item matrices across multiple machines.
10. Implement incremental updates to sparse matrices without full reconstruction.
11. How would you handle matrices that don't fit in memory?
12. Design a caching strategy for frequently accessed user/item vectors.

### Tricky Questions
13. A user rates 10,000 items out of 1 million available. Is this user dense or sparse in the context of the overall matrix?
14. You have a choice between 80% accuracy with 0.1% density or 60% accuracy with 1% density. Which would you choose for a real-time recommendation system?
15. How would you detect and handle adversarial patterns in sparse user-item matrices (e.g., fake accounts, rating farms)?
16. Design a compression algorithm specifically for user-item matrices that preserves recommendation quality while minimizing storage.

## 6. Key Takeaways

1. **Choose the right sparse format** based on your access patterns
2. **Memory efficiency is crucial** for large-scale systems  
3. **Sparsity requires specialized algorithms** for efficiency
4. **Preprocessing can significantly impact** system performance
5. **Trade-offs exist** between storage efficiency and computation speed
6. **Block-wise processing** enables handling of very large matrices

## Next Session Preview
In the next session, we'll explore collaborative filtering fundamentals, including the mathematical foundations and core assumptions that make collaborative filtering effective for sparse data.