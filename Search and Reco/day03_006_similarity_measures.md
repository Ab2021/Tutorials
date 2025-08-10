# Day 3.6: Similarity Measures and Distance Metrics

## Learning Objectives
By the end of this session, you will:
- Master various similarity measures and their mathematical foundations
- Understand when to use different similarity metrics for different data types
- Implement advanced similarity measures for collaborative filtering
- Learn about distance metrics and their relationship to similarity
- Handle different data distributions and scaling issues in similarity computation

## 1. Mathematical Foundations of Similarity

### Definition
Similarity measures quantify how alike two entities are. In recommendation systems, we measure similarity between:
- **Users**: Based on their rating patterns
- **Items**: Based on how users rate them
- **Content**: Based on features and attributes

### Properties of Good Similarity Measures
1. **Range**: Typically [0,1] or [-1,1]
2. **Symmetry**: sim(A,B) = sim(B,A) (for most measures)
3. **Self-similarity**: sim(A,A) = 1 (maximum similarity)
4. **Triangle inequality**: For some measures, sim(A,C) ≥ sim(A,B) + sim(B,C) - 1

### Relationship Between Similarity and Distance
- **Similarity** measures how alike two entities are (higher = more similar)
- **Distance** measures how far apart two entities are (lower = more similar)
- **Conversion**: sim = 1/(1+distance) or sim = 1-normalized_distance

## 2. Classical Similarity Measures

### 2.1 Cosine Similarity
Measures the cosine of the angle between two vectors.

**Formula:**
```
cos_sim(A,B) = (A·B) / (||A|| × ||B||)
```

**Properties:**
- Range: [-1, 1]
- Ignores magnitude, focuses on direction
- Good for high-dimensional sparse data

### 2.2 Pearson Correlation Coefficient
Measures linear correlation between two variables.

**Formula:**
```
pearson(A,B) = Σ(Aᵢ - Ā)(Bᵢ - B̄) / √(Σ(Aᵢ - Ā)² × Σ(Bᵢ - B̄)²)
```

**Properties:**
- Range: [-1, 1]
- Handles different rating scales well
- Sensitive to outliers

### 2.3 Jaccard Similarity
For binary or set-based data.

**Formula:**
```
jaccard(A,B) = |A ∩ B| / |A ∪ B|
```

**Properties:**
- Range: [0, 1]
- Good for implicit feedback
- Doesn't consider rating values, only presence/absence

## 3. Implementation: Comprehensive Similarity Measures Framework

```python
import numpy as np
import pandas as pd
from scipy import stats, spatial
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
import time
from collections import defaultdict

class SimilarityMeasures:
    """
    Comprehensive implementation of similarity measures for recommendation systems.
    """
    
    def __init__(self, handle_missing: str = 'pairwise', min_overlap: int = 2):
        """
        Initialize similarity measures framework.
        
        Args:
            handle_missing: How to handle missing values ('pairwise', 'listwise', 'zero')
            min_overlap: Minimum overlap required for similarity computation
        """
        self.handle_missing = handle_missing
        self.min_overlap = min_overlap
        
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray, 
                         sparse_aware: bool = True) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vector1, vector2: Input vectors
            sparse_aware: Whether to handle sparse vectors efficiently
            
        Returns:
            Cosine similarity score
        """
        if sparse_aware:
            # Handle sparse vectors (only non-zero elements)
            mask1 = vector1 != 0
            mask2 = vector2 != 0
            common_mask = mask1 | mask2
            
            if np.sum(common_mask) == 0:
                return 0.0
            
            v1 = vector1[common_mask]
            v2 = vector2[common_mask]
        else:
            v1, v2 = vector1, vector2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def adjusted_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray,
                                  user_means: np.ndarray = None) -> float:
        """
        Compute adjusted cosine similarity (user-mean centered).
        
        Args:
            vector1, vector2: Input vectors (typically item rating vectors)
            user_means: Mean ratings for each user
            
        Returns:
            Adjusted cosine similarity score
        """
        # Find common non-zero elements
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        # Subtract user means if provided
        if user_means is not None:
            user_means_common = user_means[common_mask]
            v1_adjusted = v1_common - user_means_common
            v2_adjusted = v2_common - user_means_common
        else:
            v1_adjusted = v1_common
            v2_adjusted = v2_common
        
        return self.cosine_similarity(v1_adjusted, v2_adjusted, sparse_aware=False)
    
    def pearson_correlation(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            Pearson correlation coefficient
        """
        # Find common non-zero elements
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        if len(v1_common) < 2:
            return 0.0
        
        correlation, _ = stats.pearsonr(v1_common, v2_common)
        return correlation if not np.isnan(correlation) else 0.0
    
    def spearman_correlation(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Spearman rank correlation coefficient.
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            Spearman correlation coefficient
        """
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        if len(v1_common) < 2:
            return 0.0
        
        correlation, _ = stats.spearmanr(v1_common, v2_common)
        return correlation if not np.isnan(correlation) else 0.0
    
    def jaccard_similarity(self, vector1: np.ndarray, vector2: np.ndarray,
                          threshold: float = 0.0) -> float:
        """
        Compute Jaccard similarity coefficient.
        
        Args:
            vector1, vector2: Input vectors
            threshold: Threshold for converting to binary
            
        Returns:
            Jaccard similarity coefficient
        """
        # Convert to binary
        binary1 = (vector1 > threshold).astype(int)
        binary2 = (vector2 > threshold).astype(int)
        
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        
        return intersection / union if union > 0 else 0.0
    
    def dice_coefficient(self, vector1: np.ndarray, vector2: np.ndarray,
                        threshold: float = 0.0) -> float:
        """
        Compute Dice coefficient (Sørensen-Dice similarity).
        
        Args:
            vector1, vector2: Input vectors
            threshold: Threshold for converting to binary
            
        Returns:
            Dice coefficient
        """
        binary1 = (vector1 > threshold).astype(int)
        binary2 = (vector2 > threshold).astype(int)
        
        intersection = np.sum(binary1 & binary2)
        total = np.sum(binary1) + np.sum(binary2)
        
        return 2 * intersection / total if total > 0 else 0.0
    
    def hamming_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Hamming similarity (1 - Hamming distance).
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            Hamming similarity
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length")
        
        hamming_distance = np.sum(vector1 != vector2) / len(vector1)
        return 1 - hamming_distance
    
    def manhattan_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Convert Manhattan distance to similarity.
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            Manhattan-based similarity
        """
        # Only consider common non-zero elements
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        manhattan_distance = np.sum(np.abs(v1_common - v2_common))
        return 1 / (1 + manhattan_distance)
    
    def euclidean_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Convert Euclidean distance to similarity.
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            Euclidean-based similarity
        """
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        euclidean_distance = np.sqrt(np.sum((v1_common - v2_common) ** 2))
        return 1 / (1 + euclidean_distance)
    
    def mean_squared_difference_similarity(self, vector1: np.ndarray, 
                                         vector2: np.ndarray) -> float:
        """
        Compute similarity based on mean squared differences.
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            MSD-based similarity
        """
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        msd = np.mean((v1_common - v2_common) ** 2)
        return 1 / (1 + msd)

class AdvancedSimilarityMeasures(SimilarityMeasures):
    """
    Advanced similarity measures for specific use cases.
    """
    
    def constrained_pearson_correlation(self, vector1: np.ndarray, vector2: np.ndarray,
                                      median_rating: float = 3.0) -> float:
        """
        Constrained Pearson Correlation Coefficient (CPCC).
        Uses median rating as baseline instead of mean.
        
        Args:
            vector1, vector2: Input vectors
            median_rating: Median rating to use as baseline
            
        Returns:
            CPCC score
        """
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        if np.sum(common_mask) < self.min_overlap:
            return 0.0
        
        v1_common = vector1[common_mask] - median_rating
        v2_common = vector2[common_mask] - median_rating
        
        numerator = np.sum(v1_common * v2_common)
        denominator = np.sqrt(np.sum(v1_common ** 2) * np.sum(v2_common ** 2))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def bhattacharyya_coefficient(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Bhattacharyya coefficient for probability distributions.
        
        Args:
            vector1, vector2: Input vectors (treated as probability distributions)
            
        Returns:
            Bhattacharyya coefficient
        """
        # Normalize to probability distributions
        sum1 = np.sum(vector1)
        sum2 = np.sum(vector2)
        
        if sum1 == 0 or sum2 == 0:
            return 0.0
        
        p1 = vector1 / sum1
        p2 = vector2 / sum2
        
        return np.sum(np.sqrt(p1 * p2))
    
    def kullback_leibler_similarity(self, vector1: np.ndarray, vector2: np.ndarray,
                                   epsilon: float = 1e-10) -> float:
        """
        Convert KL divergence to similarity measure.
        
        Args:
            vector1, vector2: Input vectors (treated as probability distributions)
            epsilon: Small value to avoid log(0)
            
        Returns:
            KL-based similarity
        """
        # Normalize to probability distributions
        sum1 = np.sum(vector1)
        sum2 = np.sum(vector2)
        
        if sum1 == 0 or sum2 == 0:
            return 0.0
        
        p1 = vector1 / sum1 + epsilon
        p2 = vector2 / sum2 + epsilon
        
        kl_div = np.sum(p1 * np.log(p1 / p2))
        return 1 / (1 + kl_div)
    
    def jensen_shannon_similarity(self, vector1: np.ndarray, vector2: np.ndarray,
                                 epsilon: float = 1e-10) -> float:
        """
        Compute Jensen-Shannon similarity.
        
        Args:
            vector1, vector2: Input vectors (treated as probability distributions)
            epsilon: Small value to avoid log(0)
            
        Returns:
            Jensen-Shannon similarity
        """
        # Normalize to probability distributions
        sum1 = np.sum(vector1)
        sum2 = np.sum(vector2)
        
        if sum1 == 0 or sum2 == 0:
            return 0.0
        
        p1 = vector1 / sum1 + epsilon
        p2 = vector2 / sum2 + epsilon
        m = (p1 + p2) / 2
        
        js_div = 0.5 * np.sum(p1 * np.log(p1 / m)) + 0.5 * np.sum(p2 * np.log(p2 / m))
        return 1 - np.sqrt(js_div)
    
    def tanimoto_coefficient(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Tanimoto coefficient (extended Jaccard for continuous values).
        
        Args:
            vector1, vector2: Input vectors
            
        Returns:
            Tanimoto coefficient
        """
        dot_product = np.dot(vector1, vector2)
        norm1_sq = np.dot(vector1, vector1)
        norm2_sq = np.dot(vector2, vector2)
        
        denominator = norm1_sq + norm2_sq - dot_product
        
        return dot_product / denominator if denominator > 0 else 0.0
    
    def cosine_similarity_with_confidence(self, vector1: np.ndarray, vector2: np.ndarray,
                                        confidence_threshold: int = 50) -> Tuple[float, float]:
        """
        Compute cosine similarity with confidence measure.
        
        Args:
            vector1, vector2: Input vectors
            confidence_threshold: Minimum overlap for full confidence
            
        Returns:
            Tuple of (similarity, confidence)
        """
        mask1 = vector1 != 0
        mask2 = vector2 != 0
        common_mask = mask1 & mask2
        
        overlap = np.sum(common_mask)
        
        if overlap < self.min_overlap:
            return 0.0, 0.0
        
        v1_common = vector1[common_mask]
        v2_common = vector2[common_mask]
        
        similarity = self.cosine_similarity(v1_common, v2_common, sparse_aware=False)
        confidence = min(overlap / confidence_threshold, 1.0)
        
        return similarity, confidence

class SimilarityMatrixBuilder:
    """
    Efficient builder for similarity matrices with various optimizations.
    """
    
    def __init__(self, similarity_measure: str = 'cosine', 
                 chunk_size: int = 1000,
                 n_jobs: int = 1,
                 cache_size: int = 1000):
        """
        Initialize similarity matrix builder.
        
        Args:
            similarity_measure: Type of similarity measure to use
            chunk_size: Size of chunks for batch processing
            n_jobs: Number of parallel jobs (future implementation)
            cache_size: Size of similarity cache
        """
        self.similarity_measure = similarity_measure
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.cache_size = cache_size
        
        self.sim_measures = AdvancedSimilarityMeasures()
        self.similarity_cache = {}
        
        # Map similarity measure names to functions
        self.similarity_functions = {
            'cosine': self.sim_measures.cosine_similarity,
            'adjusted_cosine': self.sim_measures.adjusted_cosine_similarity,
            'pearson': self.sim_measures.pearson_correlation,
            'spearman': self.sim_measures.spearman_correlation,
            'jaccard': self.sim_measures.jaccard_similarity,
            'dice': self.sim_measures.dice_coefficient,
            'euclidean': self.sim_measures.euclidean_similarity,
            'manhattan': self.sim_measures.manhattan_similarity,
            'msd': self.sim_measures.mean_squared_difference_similarity,
            'cpcc': self.sim_measures.constrained_pearson_correlation,
            'bhattacharyya': self.sim_measures.bhattacharyya_coefficient,
            'tanimoto': self.sim_measures.tanimoto_coefficient
        }
    
    def compute_similarity_matrix(self, data_matrix: np.ndarray, 
                                 axis: int = 0,
                                 symmetric: bool = True,
                                 store_only_top_k: int = None) -> np.ndarray:
        """
        Compute similarity matrix efficiently.
        
        Args:
            data_matrix: Input data matrix
            axis: Axis along which to compute similarities (0=rows, 1=columns)
            symmetric: Whether the similarity matrix is symmetric
            store_only_top_k: Store only top-k similarities per entity
            
        Returns:
            Similarity matrix
        """
        if axis == 1:
            data_matrix = data_matrix.T
        
        n_entities = data_matrix.shape[0]
        similarity_matrix = np.zeros((n_entities, n_entities))
        
        print(f"Computing {self.similarity_measure} similarity matrix for {n_entities} entities...")
        
        # Get similarity function
        if self.similarity_measure not in self.similarity_functions:
            raise ValueError(f"Unknown similarity measure: {self.similarity_measure}")
        
        sim_func = self.similarity_functions[self.similarity_measure]
        
        # Compute similarities
        for i in range(n_entities):
            if i % 100 == 0:
                print(f"Processing entity {i}/{n_entities}")
            
            start_j = i + 1 if symmetric else 0
            
            for j in range(start_j, n_entities):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                # Check cache
                cache_key = (i, j) if i < j else (j, i)
                if cache_key in self.similarity_cache:
                    similarity = self.similarity_cache[cache_key]
                else:
                    # Compute similarity
                    try:
                        if self.similarity_measure == 'adjusted_cosine':
                            # Need user means for adjusted cosine
                            user_means = np.mean(data_matrix, axis=1)
                            similarity = sim_func(data_matrix[i], data_matrix[j], user_means)
                        else:
                            similarity = sim_func(data_matrix[i], data_matrix[j])
                    except Exception as e:
                        print(f"Error computing similarity between {i} and {j}: {e}")
                        similarity = 0.0
                    
                    # Cache similarity
                    if len(self.similarity_cache) < self.cache_size:
                        self.similarity_cache[cache_key] = similarity
                
                similarity_matrix[i, j] = similarity
                
                if symmetric:
                    similarity_matrix[j, i] = similarity
        
        # Store only top-k if requested
        if store_only_top_k is not None:
            similarity_matrix = self._keep_top_k_similarities(similarity_matrix, store_only_top_k)
        
        return similarity_matrix
    
    def _keep_top_k_similarities(self, similarity_matrix: np.ndarray, k: int) -> np.ndarray:
        """Keep only top-k similarities for each entity."""
        n_entities = similarity_matrix.shape[0]
        filtered_matrix = np.zeros_like(similarity_matrix)
        
        for i in range(n_entities):
            similarities = similarity_matrix[i, :]
            
            # Get top-k indices (excluding self)
            top_indices = np.argsort(similarities)[::-1]
            top_indices = top_indices[top_indices != i][:k]
            
            # Keep only top-k similarities
            for j in top_indices:
                filtered_matrix[i, j] = similarities[j]
        
        return filtered_matrix
    
    def compute_similarity_distribution_analysis(self, similarity_matrix: np.ndarray) -> Dict:
        """
        Analyze the distribution of similarity values.
        
        Args:
            similarity_matrix: Input similarity matrix
            
        Returns:
            Dictionary with distribution statistics
        """
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarities = similarity_matrix[triu_indices]
        
        # Remove zeros (no similarity computed)
        non_zero_similarities = similarities[similarities != 0]
        positive_similarities = similarities[similarities > 0]
        negative_similarities = similarities[similarities < 0]
        
        analysis = {
            'total_pairs': len(similarities),
            'non_zero_pairs': len(non_zero_similarities),
            'positive_pairs': len(positive_similarities),
            'negative_pairs': len(negative_similarities),
            'sparsity': 1 - (len(non_zero_similarities) / len(similarities)),
            
            'mean_similarity': np.mean(non_zero_similarities) if len(non_zero_similarities) > 0 else 0,
            'std_similarity': np.std(non_zero_similarities) if len(non_zero_similarities) > 0 else 0,
            'min_similarity': np.min(non_zero_similarities) if len(non_zero_similarities) > 0 else 0,
            'max_similarity': np.max(non_zero_similarities) if len(non_zero_similarities) > 0 else 0,
            
            'percentiles': {
                '25th': np.percentile(non_zero_similarities, 25) if len(non_zero_similarities) > 0 else 0,
                '50th': np.percentile(non_zero_similarities, 50) if len(non_zero_similarities) > 0 else 0,
                '75th': np.percentile(non_zero_similarities, 75) if len(non_zero_similarities) > 0 else 0,
                '90th': np.percentile(non_zero_similarities, 90) if len(non_zero_similarities) > 0 else 0,
                '95th': np.percentile(non_zero_similarities, 95) if len(non_zero_similarities) > 0 else 0,
                '99th': np.percentile(non_zero_similarities, 99) if len(non_zero_similarities) > 0 else 0
            }
        }
        
        return analysis
    
    def visualize_similarity_distribution(self, similarity_matrix: np.ndarray, 
                                        title: str = "Similarity Distribution"):
        """
        Visualize similarity distribution.
        
        Args:
            similarity_matrix: Input similarity matrix
            title: Plot title
        """
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarities = similarity_matrix[triu_indices]
        non_zero_similarities = similarities[similarities != 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram of all similarities
        axes[0, 0].hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('All Similarities (including zeros)')
        axes[0, 0].set_xlabel('Similarity')
        axes[0, 0].set_ylabel('Frequency')
        
        # Histogram of non-zero similarities
        if len(non_zero_similarities) > 0:
            axes[0, 1].hist(non_zero_similarities, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Non-zero Similarities')
            axes[0, 1].set_xlabel('Similarity')
            axes[0, 1].set_ylabel('Frequency')
        
        # Box plot
        if len(non_zero_similarities) > 0:
            axes[1, 0].boxplot(non_zero_similarities)
            axes[1, 0].set_title('Similarity Box Plot')
            axes[1, 0].set_ylabel('Similarity')
        
        # Heatmap of similarity matrix (sample)
        sample_size = min(50, similarity_matrix.shape[0])
        sample_indices = np.random.choice(similarity_matrix.shape[0], sample_size, replace=False)
        sample_matrix = similarity_matrix[np.ix_(sample_indices, sample_indices)]
        
        im = axes[1, 1].imshow(sample_matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_title(f'Similarity Matrix Sample ({sample_size}x{sample_size})')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Utility functions for similarity measure selection
class SimilarityMeasureSelector:
    """
    Helper class to select appropriate similarity measures based on data characteristics.
    """
    
    @staticmethod
    def analyze_data_characteristics(data_matrix: np.ndarray) -> Dict:
        """
        Analyze data characteristics to help select similarity measure.
        
        Args:
            data_matrix: Input data matrix
            
        Returns:
            Dictionary with data characteristics
        """
        non_zero_data = data_matrix[data_matrix != 0]
        
        characteristics = {
            'n_entities': data_matrix.shape[0],
            'n_features': data_matrix.shape[1],
            'sparsity': 1 - (len(non_zero_data) / data_matrix.size),
            'mean_value': np.mean(non_zero_data) if len(non_zero_data) > 0 else 0,
            'std_value': np.std(non_zero_data) if len(non_zero_data) > 0 else 0,
            'min_value': np.min(non_zero_data) if len(non_zero_data) > 0 else 0,
            'max_value': np.max(non_zero_data) if len(non_zero_data) > 0 else 0,
            'value_range': np.max(non_zero_data) - np.min(non_zero_data) if len(non_zero_data) > 0 else 0,
            'n_unique_values': len(np.unique(non_zero_data)) if len(non_zero_data) > 0 else 0,
            'is_binary': len(np.unique(non_zero_data)) <= 2 if len(non_zero_data) > 0 else False,
            'has_negative_values': np.any(data_matrix < 0),
        }
        
        # Analyze rating scale patterns
        if len(non_zero_data) > 0:
            unique_values = np.unique(non_zero_data)
            if len(unique_values) <= 10:  # Likely discrete rating scale
                characteristics['likely_rating_scale'] = True
                characteristics['rating_scale_values'] = unique_values.tolist()
            else:
                characteristics['likely_rating_scale'] = False
        
        return characteristics
    
    @staticmethod
    def recommend_similarity_measures(data_characteristics: Dict) -> List[str]:
        """
        Recommend similarity measures based on data characteristics.
        
        Args:
            data_characteristics: Data characteristics from analyze_data_characteristics
            
        Returns:
            List of recommended similarity measures
        """
        recommendations = []
        
        sparsity = data_characteristics['sparsity']
        is_binary = data_characteristics['is_binary']
        has_negative = data_characteristics['has_negative_values']
        value_range = data_characteristics['value_range']
        
        # Binary data
        if is_binary:
            recommendations.extend(['jaccard', 'dice', 'cosine'])
        
        # High sparsity
        elif sparsity > 0.95:
            recommendations.extend(['cosine', 'adjusted_cosine', 'jaccard'])
        
        # Moderate sparsity with rating scale
        elif sparsity > 0.8 and data_characteristics.get('likely_rating_scale', False):
            recommendations.extend(['pearson', 'adjusted_cosine', 'cpcc'])
        
        # Dense data with wide value range
        elif sparsity < 0.5 and value_range > 3:
            recommendations.extend(['pearson', 'spearman', 'euclidean'])
        
        # Default recommendations
        else:
            recommendations.extend(['cosine', 'pearson', 'adjusted_cosine'])
        
        # Add distance-based measures for continuous data
        if not is_binary and value_range > 1:
            recommendations.extend(['euclidean', 'manhattan'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for item in recommendations:
            if item not in seen:
                seen.add(item)
                unique_recommendations.append(item)
        
        return unique_recommendations[:5]  # Return top 5 recommendations

# Example usage and testing
def create_similarity_test_data():
    """Create test data for similarity measure evaluation."""
    np.random.seed(42)
    
    # Create different types of data
    datasets = {}
    
    # 1. Sparse rating data (like MovieLens)
    n_users, n_items = 100, 50
    sparse_ratings = np.zeros((n_users, n_items))
    
    # Fill with sparse ratings
    for i in range(n_users):
        n_ratings = np.random.randint(3, 15)
        rated_items = np.random.choice(n_items, n_ratings, replace=False)
        for item in rated_items:
            sparse_ratings[i, item] = np.random.randint(1, 6)
    
    datasets['sparse_ratings'] = sparse_ratings
    
    # 2. Binary implicit feedback
    binary_feedback = (np.random.random((100, 50)) > 0.9).astype(int)
    datasets['binary_feedback'] = binary_feedback
    
    # 3. Dense continuous data
    dense_continuous = np.random.normal(0, 1, (50, 20))
    datasets['dense_continuous'] = dense_continuous
    
    # 4. Mixed positive data (e.g., purchase amounts)
    positive_data = np.random.exponential(2, (80, 30))
    positive_data[np.random.random((80, 30)) > 0.3] = 0  # Make sparse
    datasets['positive_data'] = positive_data
    
    return datasets

if __name__ == "__main__":
    # Create test datasets
    print("Creating test datasets...")
    datasets = create_similarity_test_data()
    
    # Analyze each dataset
    selector = SimilarityMeasureSelector()
    
    for name, data in datasets.items():
        print(f"\n=== Analyzing {name} ===")
        
        # Analyze characteristics
        characteristics = selector.analyze_data_characteristics(data)
        print("Data characteristics:")
        for key, value in characteristics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Get recommendations
        recommendations = selector.recommend_similarity_measures(characteristics)
        print(f"\nRecommended similarity measures: {recommendations}")
        
        # Test different similarity measures
        print("\nTesting similarity measures:")
        builder = SimilarityMatrixBuilder()
        
        for sim_measure in recommendations[:3]:  # Test top 3
            try:
                builder.similarity_measure = sim_measure
                start_time = time.time()
                
                # Compute similarity matrix for a subset
                subset_size = min(20, data.shape[0])
                subset_data = data[:subset_size, :]
                
                sim_matrix = builder.compute_similarity_matrix(subset_data, store_only_top_k=5)
                
                # Analyze results
                analysis = builder.compute_similarity_distribution_analysis(sim_matrix)
                
                compute_time = time.time() - start_time
                
                print(f"  {sim_measure}:")
                print(f"    Compute time: {compute_time:.3f}s")
                print(f"    Mean similarity: {analysis['mean_similarity']:.4f}")
                print(f"    Sparsity: {analysis['sparsity']:.4f}")
                print(f"    Positive pairs: {analysis['positive_pairs']}/{analysis['total_pairs']}")
                
            except Exception as e:
                print(f"  {sim_measure}: Error - {e}")
        
        # Visualize similarity distribution for the first recommended measure
        if recommendations:
            try:
                builder.similarity_measure = recommendations[0]
                subset_size = min(30, data.shape[0])
                subset_data = data[:subset_size, :]
                sim_matrix = builder.compute_similarity_matrix(subset_data)
                
                builder.visualize_similarity_distribution(
                    sim_matrix, 
                    title=f"{name.title()} - {recommendations[0].title()} Similarity"
                )
            except Exception as e:
                print(f"Visualization failed: {e}")
```

## 4. Choosing the Right Similarity Measure

### 4.1 Decision Framework

| Data Type | Sparsity | Recommended Measures | Why |
|-----------|----------|---------------------|-----|
| Binary/Implicit | High | Jaccard, Dice, Cosine | Handle binary nature well |
| Rating Scale | High | Adjusted Cosine, Pearson | Account for user bias |
| Continuous | Low | Pearson, Euclidean | Handle continuous values |
| Mixed Scale | Any | Spearman, Cosine | Robust to scale differences |

### 4.2 Performance Considerations

1. **Computational Complexity**:
   - Cosine: O(d) where d is dimensionality
   - Pearson: O(d) but requires mean computation
   - Euclidean: O(d) simple computation

2. **Memory Requirements**:
   - Sparse-aware implementations save memory
   - Distance measures often require less storage

3. **Numerical Stability**:
   - Avoid division by zero
   - Handle edge cases (all zeros, single values)

## 5. Study Questions

### Basic Level
1. What is the difference between similarity and distance measures?
2. When would you use Jaccard similarity over cosine similarity?
3. How does adjusted cosine similarity differ from regular cosine similarity?
4. Why might Pearson correlation be better than cosine for rating data?

### Intermediate Level
5. Implement a similarity measure that combines multiple metrics using weighted averaging.
6. Design a similarity measure specifically for temporal rating data.
7. How would you handle similarity computation when users have very different rating scales?
8. Compare the computational efficiency of different similarity measures on sparse data.

### Advanced Level
9. Implement a learned similarity measure using neural networks.
10. Design a similarity measure that incorporates user demographics and item features.
11. How would you adapt similarity measures for multi-criteria rating systems?
12. Implement a dynamic similarity measure that adapts based on data distribution.

### Tricky Questions
13. Two users rate completely different sets of items. How would different similarity measures handle this scenario?
14. A user always rates items either 1 or 5 (binary behavior on continuous scale). How would this affect different similarity measures?
15. Design a similarity measure that is robust to adversarial attacks (fake ratings designed to manipulate similarities).
16. How would you create a similarity measure that works well for both explicit ratings and implicit feedback simultaneously?

## 6. Key Takeaways

1. **No universal best similarity measure** - choice depends on data characteristics
2. **Cosine similarity works well** for sparse high-dimensional data
3. **Pearson correlation handles** different rating scales effectively
4. **Binary measures like Jaccard** are perfect for implicit feedback
5. **Computational efficiency** becomes crucial at scale
6. **Data preprocessing** significantly impacts similarity computation quality

## Next Session Preview
In the next session, we'll explore neighborhood selection strategies - how to choose the right neighbors once we have computed similarities, including techniques for dynamic neighborhood sizing and quality-based selection.