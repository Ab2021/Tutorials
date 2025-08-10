# Day 3.7: Neighborhood Selection Strategies

## Learning Objectives
By the end of this session, you will:
- Master various neighborhood selection strategies for collaborative filtering
- Understand the trade-offs between neighborhood size and prediction quality
- Implement dynamic and adaptive neighborhood selection algorithms
- Learn quality-based neighbor filtering techniques
- Optimize neighborhood selection for different scenarios and constraints

## 1. Fundamentals of Neighborhood Selection

### Definition
Neighborhood selection is the process of choosing which similar users/items to use for making predictions in collaborative filtering systems. The quality of neighbors directly impacts recommendation accuracy and diversity.

### Key Challenges
1. **Optimal neighborhood size**: Too few neighbors → poor coverage; Too many → noise inclusion
2. **Quality vs. quantity**: High similarity vs. sufficient coverage
3. **Computational efficiency**: Balance accuracy with speed
4. **Dynamic adaptation**: Adjust to different users/items and contexts

### Neighborhood Selection Pipeline
```
Similarity Computation → Neighbor Filtering → Neighborhood Sizing → Quality Assessment → Final Selection
```

## 2. Classical Neighborhood Selection Strategies

### 2.1 Fixed-Size Neighborhoods (Top-K)
Select the k most similar neighbors regardless of similarity values.

**Advantages:**
- Predictable computational cost
- Simple to implement
- Consistent neighborhood sizes

**Disadvantages:**
- May include low-quality neighbors
- Ignores similarity distribution
- Not adaptive to data characteristics

### 2.2 Threshold-Based Selection
Select all neighbors above a similarity threshold.

**Advantages:**
- Quality guarantee (minimum similarity)
- Adaptive to similarity distribution
- Natural filtering of poor neighbors

**Disadvantages:**
- Variable neighborhood sizes
- May result in empty neighborhoods
- Threshold selection is critical

### 2.3 Hybrid Approaches
Combine fixed-size and threshold-based methods.

## 3. Implementation: Advanced Neighborhood Selection Framework

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import time
from abc import ABC, abstractmethod

class NeighborhoodSelector(ABC):
    """
    Abstract base class for neighborhood selection strategies.
    """
    
    def __init__(self, min_neighbors: int = 1, max_neighbors: int = 100):
        """
        Initialize neighborhood selector.
        
        Args:
            min_neighbors: Minimum number of neighbors required
            max_neighbors: Maximum number of neighbors to consider
        """
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
    
    @abstractmethod
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """
        Select neighbors for a target entity.
        
        Args:
            target_idx: Index of target entity
            similarity_vector: Similarity scores to all other entities
            context: Additional context for selection
            
        Returns:
            List of (neighbor_idx, similarity) tuples
        """
        pass

class TopKSelector(NeighborhoodSelector):
    """
    Select top-k most similar neighbors.
    """
    
    def __init__(self, k: int = 50, min_neighbors: int = 1, max_neighbors: int = 100):
        super().__init__(min_neighbors, max_neighbors)
        self.k = k
    
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """Select top-k neighbors."""
        # Remove self-similarity
        similarities = similarity_vector.copy()
        similarities[target_idx] = -np.inf
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:self.k]
        
        # Filter positive similarities and respect bounds
        neighbors = []
        for idx in top_indices:
            if similarities[idx] > 0 and len(neighbors) < self.max_neighbors:
                neighbors.append((idx, similarities[idx]))
        
        # Ensure minimum neighbors if possible
        if len(neighbors) < self.min_neighbors:
            # Add more neighbors even if similarity is low/zero
            remaining_indices = np.argsort(similarities)[::-1][len(neighbors):]
            for idx in remaining_indices:
                if len(neighbors) >= self.min_neighbors:
                    break
                if idx != target_idx:
                    neighbors.append((idx, similarities[idx]))
        
        return neighbors

class ThresholdSelector(NeighborhoodSelector):
    """
    Select neighbors above a similarity threshold.
    """
    
    def __init__(self, threshold: float = 0.1, min_neighbors: int = 1, 
                 max_neighbors: int = 100):
        super().__init__(min_neighbors, max_neighbors)
        self.threshold = threshold
    
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """Select neighbors above threshold."""
        similarities = similarity_vector.copy()
        similarities[target_idx] = -np.inf
        
        # Find neighbors above threshold
        above_threshold = np.where(similarities >= self.threshold)[0]
        
        # Sort by similarity (descending)
        sorted_indices = above_threshold[np.argsort(similarities[above_threshold])[::-1]]
        
        neighbors = []
        for idx in sorted_indices[:self.max_neighbors]:
            neighbors.append((idx, similarities[idx]))
        
        # Ensure minimum neighbors if needed
        if len(neighbors) < self.min_neighbors:
            all_indices = np.argsort(similarities)[::-1]
            for idx in all_indices:
                if len(neighbors) >= self.min_neighbors:
                    break
                if idx != target_idx and idx not in [n[0] for n in neighbors]:
                    neighbors.append((idx, similarities[idx]))
        
        return neighbors

class AdaptiveSelector(NeighborhoodSelector):
    """
    Adaptive neighborhood selection based on similarity distribution.
    """
    
    def __init__(self, base_size: int = 30, adaptation_factor: float = 0.5,
                 min_neighbors: int = 5, max_neighbors: int = 100):
        super().__init__(min_neighbors, max_neighbors)
        self.base_size = base_size
        self.adaptation_factor = adaptation_factor
    
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """Select neighbors adaptively based on similarity distribution."""
        similarities = similarity_vector.copy()
        similarities[target_idx] = -np.inf
        
        # Calculate adaptive neighborhood size
        positive_similarities = similarities[similarities > 0]
        
        if len(positive_similarities) == 0:
            return []
        
        # Adapt size based on similarity distribution
        mean_sim = np.mean(positive_similarities)
        std_sim = np.std(positive_similarities)
        
        # Higher std -> more diverse similarities -> can afford larger neighborhood
        # Higher mean -> generally high similarities -> can be more selective
        size_multiplier = 1 + self.adaptation_factor * (std_sim - mean_sim)
        adaptive_size = max(self.min_neighbors, 
                           min(self.max_neighbors, 
                               int(self.base_size * size_multiplier)))
        
        # Select top neighbors
        top_indices = np.argsort(similarities)[::-1][:adaptive_size]
        
        neighbors = []
        for idx in top_indices:
            if similarities[idx] > 0:
                neighbors.append((idx, similarities[idx]))
        
        return neighbors

class QualityBasedSelector(NeighborhoodSelector):
    """
    Select neighbors based on multiple quality criteria.
    """
    
    def __init__(self, similarity_weight: float = 0.6, 
                 coverage_weight: float = 0.2,
                 confidence_weight: float = 0.2,
                 min_neighbors: int = 5, max_neighbors: int = 50):
        super().__init__(min_neighbors, max_neighbors)
        self.similarity_weight = similarity_weight
        self.coverage_weight = coverage_weight
        self.confidence_weight = confidence_weight
    
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """Select neighbors based on multiple quality criteria."""
        similarities = similarity_vector.copy()
        similarities[target_idx] = -np.inf
        
        # Get context information
        if context is None:
            context = {}
        
        coverage_scores = context.get('coverage_scores', np.ones_like(similarities))
        confidence_scores = context.get('confidence_scores', np.ones_like(similarities))
        
        # Calculate composite quality scores
        quality_scores = (self.similarity_weight * similarities +
                         self.coverage_weight * coverage_scores +
                         self.confidence_weight * confidence_scores)
        
        # Select based on quality scores
        quality_indices = np.argsort(quality_scores)[::-1]
        
        neighbors = []
        for idx in quality_indices:
            if (idx != target_idx and 
                similarities[idx] > 0 and 
                len(neighbors) < self.max_neighbors):
                neighbors.append((idx, similarities[idx]))
        
        # Ensure minimum neighbors
        if len(neighbors) < self.min_neighbors:
            remaining_indices = np.argsort(similarities)[::-1]
            for idx in remaining_indices:
                if len(neighbors) >= self.min_neighbors:
                    break
                if idx != target_idx and idx not in [n[0] for n in neighbors]:
                    neighbors.append((idx, similarities[idx]))
        
        return neighbors

class DiversityAwareSelector(NeighborhoodSelector):
    """
    Select neighbors with diversity considerations.
    """
    
    def __init__(self, diversity_factor: float = 0.3, base_size: int = 40,
                 min_neighbors: int = 5, max_neighbors: int = 60):
        super().__init__(min_neighbors, max_neighbors)
        self.diversity_factor = diversity_factor
        self.base_size = base_size
    
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """Select diverse neighbors using greedy diversification."""
        similarities = similarity_vector.copy()
        similarities[target_idx] = -np.inf
        
        if context is None or 'entity_features' not in context:
            # Fall back to top-k if no diversity information
            return TopKSelector(self.base_size).select_neighbors(target_idx, similarity_vector)
        
        entity_features = context['entity_features']
        
        # Start with highest similarity neighbor
        candidates = np.where(similarities > 0)[0]
        if len(candidates) == 0:
            return []
        
        candidates = candidates[np.argsort(similarities[candidates])[::-1]]
        
        selected = []
        selected_features = []
        
        for candidate in candidates:
            if len(selected) >= self.max_neighbors:
                break
            
            candidate_sim = similarities[candidate]
            candidate_features = entity_features[candidate]
            
            if len(selected) == 0:
                # Always select first (highest similarity)
                selected.append((candidate, candidate_sim))
                selected_features.append(candidate_features)
            else:
                # Calculate diversity score
                diversity_score = self._calculate_diversity(candidate_features, selected_features)
                
                # Combined score: similarity + diversity
                combined_score = ((1 - self.diversity_factor) * candidate_sim + 
                                self.diversity_factor * diversity_score)
                
                # Accept if adds sufficient value
                if (combined_score > 0.1 * similarities[candidates[0]] or 
                    len(selected) < self.min_neighbors):
                    selected.append((candidate, candidate_sim))
                    selected_features.append(candidate_features)
        
        return selected
    
    def _calculate_diversity(self, candidate_features: np.ndarray, 
                           selected_features: List[np.ndarray]) -> float:
        """Calculate diversity score for a candidate."""
        if len(selected_features) == 0:
            return 1.0
        
        # Calculate minimum distance to selected neighbors
        distances = []
        for selected_feature in selected_features:
            distance = np.linalg.norm(candidate_features - selected_feature)
            distances.append(distance)
        
        # Return normalized minimum distance (higher = more diverse)
        min_distance = min(distances)
        max_possible_distance = np.sqrt(len(candidate_features))  # Rough normalization
        
        return min_distance / max_possible_distance

class DynamicSelector(NeighborhoodSelector):
    """
    Dynamic neighborhood selection that adapts based on prediction context.
    """
    
    def __init__(self, min_neighbors: int = 3, max_neighbors: int = 80,
                 quality_threshold: float = 0.05):
        super().__init__(min_neighbors, max_neighbors)
        self.quality_threshold = quality_threshold
        self.selection_history = defaultdict(list)
    
    def select_neighbors(self, target_idx: int, similarity_vector: np.ndarray,
                        context: Dict = None) -> List[Tuple[int, float]]:
        """Dynamically select neighbors based on context and history."""
        similarities = similarity_vector.copy()
        similarities[target_idx] = -np.inf
        
        if context is None:
            context = {}
        
        # Get context factors
        prediction_difficulty = context.get('prediction_difficulty', 0.5)
        target_popularity = context.get('target_popularity', 0.5)
        user_specificity = context.get('user_specificity', 0.5)
        
        # Adapt neighborhood size based on context
        base_size = self._calculate_adaptive_size(prediction_difficulty, 
                                                 target_popularity, 
                                                 user_specificity)
        
        # Select high-quality neighbors
        quality_threshold = self._calculate_adaptive_threshold(similarities)
        
        candidates = np.where(similarities >= quality_threshold)[0]
        if len(candidates) == 0:
            candidates = np.where(similarities > 0)[0]
        
        # Sort by similarity
        candidates = candidates[np.argsort(similarities[candidates])[::-1]]
        
        # Select up to base_size neighbors
        selected = []
        for candidate in candidates[:min(base_size, len(candidates))]:
            selected.append((candidate, similarities[candidate]))
        
        # Ensure minimum neighbors
        if len(selected) < self.min_neighbors:
            all_candidates = np.argsort(similarities)[::-1]
            for candidate in all_candidates:
                if len(selected) >= self.min_neighbors:
                    break
                if (candidate != target_idx and 
                    candidate not in [s[0] for s in selected]):
                    selected.append((candidate, similarities[candidate]))
        
        # Store selection for learning
        self.selection_history[target_idx].append({
            'neighborhood_size': len(selected),
            'context': context.copy(),
            'avg_similarity': np.mean([s[1] for s in selected]) if selected else 0
        })
        
        return selected
    
    def _calculate_adaptive_size(self, prediction_difficulty: float,
                               target_popularity: float,
                               user_specificity: float) -> int:
        """Calculate adaptive neighborhood size based on context."""
        # More difficult predictions need more neighbors
        # Popular targets can use fewer neighbors
        # Specific users might need more neighbors
        
        difficulty_factor = 1 + prediction_difficulty
        popularity_factor = 1 - 0.5 * target_popularity
        specificity_factor = 1 + 0.3 * user_specificity
        
        adaptive_factor = difficulty_factor * popularity_factor * specificity_factor
        
        base_size = int(30 * adaptive_factor)
        return max(self.min_neighbors, min(self.max_neighbors, base_size))
    
    def _calculate_adaptive_threshold(self, similarities: np.ndarray) -> float:
        """Calculate adaptive similarity threshold."""
        positive_sims = similarities[similarities > 0]
        
        if len(positive_sims) == 0:
            return 0.0
        
        # Use percentile-based threshold
        threshold = np.percentile(positive_sims, 70)  # Top 30% similarities
        return max(self.quality_threshold, threshold * 0.5)

class NeighborhoodOptimizer:
    """
    Optimizer for neighborhood selection strategies.
    """
    
    def __init__(self, rating_matrix: np.ndarray, similarity_matrix: np.ndarray):
        """
        Initialize neighborhood optimizer.
        
        Args:
            rating_matrix: User-item rating matrix
            similarity_matrix: Precomputed similarity matrix
        """
        self.rating_matrix = rating_matrix
        self.similarity_matrix = similarity_matrix
        self.n_entities = rating_matrix.shape[0]
        
    def evaluate_selector(self, selector: NeighborhoodSelector,
                         test_cases: List[Tuple[int, int]] = None,
                         n_test_cases: int = 100) -> Dict:
        """
        Evaluate a neighborhood selector.
        
        Args:
            selector: Neighborhood selector to evaluate
            test_cases: Specific test cases (entity_idx, target_item)
            n_test_cases: Number of random test cases if test_cases not provided
            
        Returns:
            Evaluation metrics
        """
        if test_cases is None:
            # Generate random test cases
            test_cases = []
            for _ in range(n_test_cases):
                entity_idx = np.random.randint(0, self.n_entities)
                # Find items this entity hasn't rated
                unrated_items = np.where(self.rating_matrix[entity_idx, :] == 0)[0]
                if len(unrated_items) > 0:
                    target_item = np.random.choice(unrated_items)
                    test_cases.append((entity_idx, target_item))
        
        metrics = {
            'neighborhood_sizes': [],
            'avg_similarities': [],
            'coverage_rates': [],
            'prediction_errors': [],
            'computation_times': []
        }
        
        for entity_idx, target_item in test_cases:
            start_time = time.time()
            
            # Select neighbors
            neighbors = selector.select_neighbors(
                entity_idx, 
                self.similarity_matrix[entity_idx, :],
                context={'target_item': target_item}
            )
            
            computation_time = time.time() - start_time
            
            # Calculate metrics
            neighborhood_size = len(neighbors)
            avg_similarity = np.mean([sim for _, sim in neighbors]) if neighbors else 0
            
            # Coverage: fraction of neighbors that have rated the target item
            coverage_count = 0
            for neighbor_idx, _ in neighbors:
                if self.rating_matrix[neighbor_idx, target_item] > 0:
                    coverage_count += 1
            
            coverage_rate = coverage_count / neighborhood_size if neighborhood_size > 0 else 0
            
            # Simple prediction error (if we can make a prediction)
            prediction_error = np.nan
            if coverage_count > 0:
                # Simple weighted average prediction
                numerator = 0
                denominator = 0
                for neighbor_idx, similarity in neighbors:
                    neighbor_rating = self.rating_matrix[neighbor_idx, target_item]
                    if neighbor_rating > 0:
                        numerator += similarity * neighbor_rating
                        denominator += abs(similarity)
                
                if denominator > 0:
                    prediction = numerator / denominator
                    # Use global mean as "true" rating for synthetic evaluation
                    true_rating = np.mean(self.rating_matrix[self.rating_matrix > 0])
                    prediction_error = abs(prediction - true_rating)
            
            # Store metrics
            metrics['neighborhood_sizes'].append(neighborhood_size)
            metrics['avg_similarities'].append(avg_similarity)
            metrics['coverage_rates'].append(coverage_rate)
            metrics['computation_times'].append(computation_time)
            
            if not np.isnan(prediction_error):
                metrics['prediction_errors'].append(prediction_error)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics.items():
            if values:  # Only if we have values
                summary[f'avg_{metric_name}'] = np.mean(values)
                summary[f'std_{metric_name}'] = np.std(values)
                summary[f'min_{metric_name}'] = np.min(values)
                summary[f'max_{metric_name}'] = np.max(values)
        
        summary['n_test_cases'] = len(test_cases)
        summary['raw_metrics'] = metrics
        
        return summary
    
    def compare_selectors(self, selectors: Dict[str, NeighborhoodSelector],
                         n_test_cases: int = 200) -> Dict:
        """
        Compare multiple neighborhood selectors.
        
        Args:
            selectors: Dictionary of selector_name -> selector
            n_test_cases: Number of test cases
            
        Returns:
            Comparison results
        """
        print(f"Comparing {len(selectors)} neighborhood selectors on {n_test_cases} test cases...")
        
        results = {}
        
        for selector_name, selector in selectors.items():
            print(f"Evaluating {selector_name}...")
            start_time = time.time()
            
            evaluation = self.evaluate_selector(selector, n_test_cases=n_test_cases)
            evaluation['total_evaluation_time'] = time.time() - start_time
            
            results[selector_name] = evaluation
            
            print(f"  Avg neighborhood size: {evaluation.get('avg_neighborhood_sizes', 0):.1f}")
            print(f"  Avg similarity: {evaluation.get('avg_avg_similarities', 0):.4f}")
            print(f"  Avg coverage: {evaluation.get('avg_coverage_rates', 0):.4f}")
            print(f"  Evaluation time: {evaluation['total_evaluation_time']:.2f}s")
        
        return results
    
    def visualize_comparison(self, comparison_results: Dict, 
                           metrics_to_plot: List[str] = None):
        """
        Visualize comparison results.
        
        Args:
            comparison_results: Results from compare_selectors
            metrics_to_plot: List of metrics to plot
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['avg_neighborhood_sizes', 'avg_avg_similarities', 
                             'avg_coverage_rates', 'avg_computation_times']
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Extract values for this metric
            selector_names = list(comparison_results.keys())
            values = [comparison_results[name].get(metric, 0) for name in selector_names]
            
            # Bar plot
            bars = ax.bar(selector_names, values)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# Utility functions for neighborhood analysis
class NeighborhoodAnalyzer:
    """
    Analyzer for neighborhood characteristics and quality.
    """
    
    def __init__(self, rating_matrix: np.ndarray, similarity_matrix: np.ndarray):
        self.rating_matrix = rating_matrix
        self.similarity_matrix = similarity_matrix
        self.n_users, self.n_items = rating_matrix.shape
    
    def analyze_neighborhood_quality(self, neighborhoods: Dict[int, List[Tuple[int, float]]]) -> Dict:
        """
        Analyze quality characteristics of neighborhoods.
        
        Args:
            neighborhoods: Dictionary mapping entity_idx to list of (neighbor_idx, similarity)
            
        Returns:
            Quality analysis results
        """
        analysis = {
            'size_distribution': [],
            'similarity_distributions': [],
            'coverage_analysis': {},
            'diversity_metrics': {}
        }
        
        for entity_idx, neighbors in neighborhoods.items():
            if not neighbors:
                continue
            
            # Size analysis
            neighborhood_size = len(neighbors)
            analysis['size_distribution'].append(neighborhood_size)
            
            # Similarity analysis
            similarities = [sim for _, sim in neighbors]
            analysis['similarity_distributions'].extend(similarities)
            
            # Coverage analysis - how many items do neighbors cover
            neighbor_indices = [idx for idx, _ in neighbors]
            neighbor_ratings = self.rating_matrix[neighbor_indices, :]
            items_covered = np.sum(np.any(neighbor_ratings > 0, axis=0))
            coverage_ratio = items_covered / self.n_items
            
            if entity_idx not in analysis['coverage_analysis']:
                analysis['coverage_analysis'][entity_idx] = coverage_ratio
        
        # Summary statistics
        analysis['summary'] = {
            'avg_neighborhood_size': np.mean(analysis['size_distribution']),
            'std_neighborhood_size': np.std(analysis['size_distribution']),
            'avg_similarity': np.mean(analysis['similarity_distributions']),
            'std_similarity': np.std(analysis['similarity_distributions']),
            'avg_coverage_ratio': np.mean(list(analysis['coverage_analysis'].values()))
        }
        
        return analysis
    
    def find_optimal_neighborhood_size(self, entity_idx: int, 
                                     size_range: Tuple[int, int] = (5, 100),
                                     validation_items: List[int] = None) -> Dict:
        """
        Find optimal neighborhood size for a specific entity.
        
        Args:
            entity_idx: Target entity index
            size_range: Range of neighborhood sizes to test
            validation_items: Items to use for validation
            
        Returns:
            Optimization results
        """
        if validation_items is None:
            # Use random sample of items this entity has rated
            rated_items = np.where(self.rating_matrix[entity_idx, :] > 0)[0]
            if len(rated_items) < 10:
                return {'error': 'Not enough rated items for validation'}
            validation_items = np.random.choice(rated_items, min(10, len(rated_items)), replace=False)
        
        results = {
            'sizes': [],
            'prediction_errors': [],
            'coverage_rates': []
        }
        
        similarities = self.similarity_matrix[entity_idx, :]
        similarities[entity_idx] = -np.inf  # Remove self
        
        # Sort potential neighbors by similarity
        sorted_neighbors = np.argsort(similarities)[::-1]
        
        for size in range(size_range[0], min(size_range[1] + 1, len(sorted_neighbors))):
            # Select top-size neighbors
            neighbor_indices = sorted_neighbors[:size]
            neighbor_similarities = similarities[neighbor_indices]
            
            # Only keep positive similarities
            positive_mask = neighbor_similarities > 0
            if np.sum(positive_mask) < size_range[0]:
                continue
            
            valid_neighbors = neighbor_indices[positive_mask]
            valid_similarities = neighbor_similarities[positive_mask]
            
            # Evaluate on validation items
            errors = []
            coverage_count = 0
            
            for item_idx in validation_items:
                # Hide this rating and try to predict
                true_rating = self.rating_matrix[entity_idx, item_idx]
                
                # Count neighbors who rated this item
                neighbor_ratings = self.rating_matrix[valid_neighbors, item_idx]
                covering_neighbors = neighbor_ratings > 0
                
                if np.any(covering_neighbors):
                    coverage_count += 1
                    
                    # Make prediction
                    covering_sims = valid_similarities[covering_neighbors]
                    covering_ratings = neighbor_ratings[covering_neighbors]
                    
                    if np.sum(np.abs(covering_sims)) > 0:
                        prediction = np.sum(covering_sims * covering_ratings) / np.sum(np.abs(covering_sims))
                        error = abs(prediction - true_rating)
                        errors.append(error)
            
            if errors:
                results['sizes'].append(len(valid_neighbors))
                results['prediction_errors'].append(np.mean(errors))
                results['coverage_rates'].append(coverage_count / len(validation_items))
        
        # Find optimal size
        if results['prediction_errors']:
            best_idx = np.argmin(results['prediction_errors'])
            optimal_size = results['sizes'][best_idx]
            optimal_error = results['prediction_errors'][best_idx]
            optimal_coverage = results['coverage_rates'][best_idx]
            
            results['optimal'] = {
                'size': optimal_size,
                'error': optimal_error,
                'coverage': optimal_coverage
            }
        
        return results

# Example usage and testing
def create_neighborhood_test_scenario():
    """Create test scenario for neighborhood selection evaluation."""
    np.random.seed(42)
    
    # Create synthetic rating matrix with known patterns
    n_users, n_items = 200, 100
    rating_matrix = np.zeros((n_users, n_items))
    
    # Create user clusters with different rating behaviors
    cluster_sizes = [50, 50, 50, 50]  # 4 clusters
    cluster_preferences = [
        {'preferred_items': list(range(0, 25)), 'avg_rating': 4.2},
        {'preferred_items': list(range(25, 50)), 'avg_rating': 3.8},
        {'preferred_items': list(range(50, 75)), 'avg_rating': 4.5},
        {'preferred_items': list(range(75, 100)), 'avg_rating': 3.5}
    ]
    
    user_idx = 0
    for cluster_idx, cluster_size in enumerate(cluster_sizes):
        prefs = cluster_preferences[cluster_idx]
        
        for _ in range(cluster_size):
            # Rate preferred items highly
            for item in prefs['preferred_items']:
                if np.random.random() > 0.3:  # 70% chance to rate
                    noise = np.random.normal(0, 0.8)
                    rating = np.clip(prefs['avg_rating'] + noise, 1, 5)
                    rating_matrix[user_idx, item] = rating
            
            # Rate some other items (lower ratings)
            other_items = [i for i in range(n_items) if i not in prefs['preferred_items']]
            n_other = np.random.randint(5, 15)
            rated_other = np.random.choice(other_items, n_other, replace=False)
            
            for item in rated_other:
                noise = np.random.normal(0, 1.0)
                rating = np.clip(2.5 + noise, 1, 5)
                rating_matrix[user_idx, item] = rating
            
            user_idx += 1
    
    # Compute similarity matrix (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Normalize by user mean for better similarity computation
    user_means = np.mean(rating_matrix, axis=1, keepdims=True)
    user_means[user_means == 0] = 0  # Handle users with no ratings
    
    normalized_matrix = rating_matrix - user_means
    normalized_matrix[rating_matrix == 0] = 0  # Keep zeros as zeros
    
    similarity_matrix = cosine_similarity(normalized_matrix)
    
    return rating_matrix, similarity_matrix, cluster_preferences

if __name__ == "__main__":
    # Create test scenario
    print("Creating neighborhood selection test scenario...")
    rating_matrix, similarity_matrix, cluster_info = create_neighborhood_test_scenario()
    
    print(f"Rating matrix shape: {rating_matrix.shape}")
    print(f"Matrix density: {np.count_nonzero(rating_matrix) / rating_matrix.size:.4f}")
    
    # Initialize different selectors
    selectors = {
        'TopK_20': TopKSelector(k=20),
        'TopK_50': TopKSelector(k=50),
        'Threshold_0.1': ThresholdSelector(threshold=0.1),
        'Threshold_0.3': ThresholdSelector(threshold=0.3),
        'Adaptive': AdaptiveSelector(base_size=30, adaptation_factor=0.5),
        'QualityBased': QualityBasedSelector(),
        'Dynamic': DynamicSelector()
    }
    
    # Initialize optimizer and compare selectors
    optimizer = NeighborhoodOptimizer(rating_matrix, similarity_matrix)
    
    print("\nComparing neighborhood selectors...")
    comparison_results = optimizer.compare_selectors(selectors, n_test_cases=100)
    
    # Print comparison summary
    print("\n=== Selector Comparison Summary ===")
    metrics = ['avg_neighborhood_sizes', 'avg_avg_similarities', 'avg_coverage_rates']
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for name, results in comparison_results.items():
            value = results.get(metric, 0)
            print(f"  {name:15}: {value:.4f}")
    
    # Visualize comparison
    optimizer.visualize_comparison(comparison_results)
    
    # Analyze neighborhood quality for best performer
    best_selector_name = min(comparison_results.keys(), 
                           key=lambda x: comparison_results[x].get('avg_prediction_errors', float('inf')))
    
    print(f"\nAnalyzing neighborhoods from best selector: {best_selector_name}")
    best_selector = selectors[best_selector_name]
    
    # Generate neighborhoods for analysis
    sample_entities = np.random.choice(rating_matrix.shape[0], 20, replace=False)
    neighborhoods = {}
    
    for entity_idx in sample_entities:
        neighbors = best_selector.select_neighbors(entity_idx, similarity_matrix[entity_idx, :])
        neighborhoods[entity_idx] = neighbors
    
    # Analyze neighborhood quality
    analyzer = NeighborhoodAnalyzer(rating_matrix, similarity_matrix)
    quality_analysis = analyzer.analyze_neighborhood_quality(neighborhoods)
    
    print("Neighborhood Quality Analysis:")
    for key, value in quality_analysis['summary'].items():
        print(f"  {key}: {value:.4f}")
    
    # Find optimal neighborhood size for a sample user
    sample_user = sample_entities[0]
    print(f"\nFinding optimal neighborhood size for user {sample_user}...")
    
    size_optimization = analyzer.find_optimal_neighborhood_size(sample_user, size_range=(5, 80))
    
    if 'optimal' in size_optimization:
        opt = size_optimization['optimal']
        print(f"Optimal neighborhood size: {opt['size']}")
        print(f"Prediction error: {opt['error']:.4f}")
        print(f"Coverage rate: {opt['coverage']:.4f}")
        
        # Plot size vs error
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(size_optimization['sizes'], size_optimization['prediction_errors'], 'b-o')
        plt.axvline(x=opt['size'], color='r', linestyle='--', label=f'Optimal size: {opt["size"]}')
        plt.xlabel('Neighborhood Size')
        plt.ylabel('Prediction Error')
        plt.title('Neighborhood Size vs Prediction Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(size_optimization['sizes'], size_optimization['coverage_rates'], 'g-o')
        plt.axvline(x=opt['size'], color='r', linestyle='--', label=f'Optimal size: {opt["size"]}')
        plt.xlabel('Neighborhood Size')
        plt.ylabel('Coverage Rate')
        plt.title('Neighborhood Size vs Coverage Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## 4. Advanced Neighborhood Selection Techniques

### 4.1 Multi-Criteria Selection
Combine multiple factors:
- Similarity strength
- Coverage capability
- Prediction confidence
- Diversity measures

### 4.2 Context-Aware Selection
Adapt neighborhoods based on:
- Item popularity
- User activity level
- Temporal factors
- Domain-specific constraints

### 4.3 Learning-Based Selection
Use machine learning to optimize:
- Reinforcement learning for adaptive sizing
- Classification for neighbor quality prediction
- Regression for optimal threshold learning

## 5. Study Questions

### Basic Level
1. What are the trade-offs between fixed-size and threshold-based neighborhood selection?
2. How does neighborhood size affect prediction accuracy and computational cost?
3. Why might you want different neighborhood sizes for different users or items?
4. What problems can arise from including too many or too few neighbors?

### Intermediate Level
5. Implement a neighborhood selector that balances similarity and diversity.
6. Design a method to detect when a neighborhood is of poor quality.
7. How would you adapt neighborhood selection for real-time recommendations?
8. Compare the effectiveness of different selection strategies on sparse vs dense data.

### Advanced Level
9. Implement a multi-armed bandit approach for adaptive neighborhood sizing.
10. Design a neighborhood selection strategy that considers temporal dynamics.
11. How would you incorporate user feedback to improve neighborhood selection?
12. Implement a distributed neighborhood selection algorithm for large-scale systems.

### Tricky Questions
13. A user has very unique preferences unlike anyone else. How would different neighborhood selection strategies handle this case?
14. Design a neighborhood selection method that works well when similarity measures are unreliable due to data sparsity.
15. How would you detect and handle cases where the most similar neighbors are actually adversarial (fake accounts, manipulation)?
16. Create a neighborhood selection strategy that automatically adapts its criteria based on the success of previous predictions.

## 6. Key Takeaways

1. **Neighborhood quality matters more than quantity** - fewer high-quality neighbors often beat many low-quality ones
2. **Adaptive sizing improves performance** across different users and contexts
3. **Coverage and diversity** should be balanced with similarity
4. **Context-aware selection** can significantly improve recommendation quality
5. **Dynamic adaptation** helps handle changing user preferences and data characteristics
6. **Evaluation and optimization** are crucial for finding the right strategy for your specific use case

## Next Session Preview
In the next session, we'll dive into matrix factorization techniques, starting with the mathematical foundations and moving to practical implementations of SVD, NMF, and other factorization methods that form the backbone of modern collaborative filtering systems.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create day2_09_ranking_evaluation.md: IR Evaluation Metrics", "status": "completed", "priority": "high", "id": "create_day2_09_ranking_evaluation"}, {"content": "Create day3_001_useritem_introduction.md: Introduction to User-Item Matrices", "status": "completed", "priority": "high", "id": "create_day3_001"}, {"content": "Create day3_002_useritem_representation.md: User-Item Matrix Representations and Sparsity", "status": "completed", "priority": "high", "id": "create_day3_002"}, {"content": "Create day3_003_cf_fundamentals.md: Collaborative Filtering Fundamentals", "status": "completed", "priority": "high", "id": "create_day3_003"}, {"content": "Create day3_004_user_based_cf.md: User-Based Collaborative Filtering", "status": "completed", "priority": "high", "id": "create_day3_004"}, {"content": "Create day3_005_item_based_cf.md: Item-Based Collaborative Filtering", "status": "completed", "priority": "high", "id": "create_day3_005"}, {"content": "Create day3_006_similarity_measures.md: Similarity Measures and Distance Metrics", "status": "completed", "priority": "high", "id": "create_day3_006"}, {"content": "Create day3_007_neighborhood_selection.md: Neighborhood Selection Strategies", "status": "completed", "priority": "high", "id": "create_day3_007"}, {"content": "Create day3_008_matrix_factorization_intro.md: Introduction to Matrix Factorization", "status": "in_progress", "priority": "high", "id": "create_day3_008"}, {"content": "Create day3_009_svd_techniques.md: SVD and Advanced Factorization Techniques", "status": "pending", "priority": "high", "id": "create_day3_009"}, {"content": "Create day3_010_cold_start_problem.md: Cold Start Problem Analysis", "status": "pending", "priority": "high", "id": "create_day3_010"}, {"content": "Create day3_011_cold_start_solutions.md: Cold Start Solutions and Strategies", "status": "pending", "priority": "high", "id": "create_day3_011"}]