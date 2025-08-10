# Day 5.1: Hybrid Recommendation Systems - Fundamentals

## Learning Objectives
By the end of this session, you will:
- Understand the fundamental concepts and motivations for hybrid recommendation systems
- Learn different hybridization approaches and their trade-offs
- Implement basic hybrid architectures combining multiple recommendation strategies
- Master the theoretical foundations of recommendation fusion
- Apply hybrid systems to overcome individual algorithm limitations

## 1. Introduction to Hybrid Recommendation Systems

### Why Hybrid Systems?

Hybrid recommendation systems combine multiple recommendation techniques to leverage their individual strengths while mitigating their weaknesses. This approach addresses several critical limitations:

**Individual Algorithm Limitations:**
- **Collaborative Filtering**: Cold start, sparsity, popularity bias
- **Content-Based**: Over-specialization, limited diversity, feature engineering challenges
- **Knowledge-Based**: Manual rule creation, domain expertise requirements

**Benefits of Hybridization:**
- **Improved Accuracy**: Combining predictions from multiple algorithms
- **Enhanced Coverage**: Different algorithms handle different user/item scenarios
- **Reduced Cold Start**: Content-based methods help with new items, CF helps with new users
- **Increased Diversity**: Multiple perspectives prevent filter bubbles
- **Robustness**: System remains functional when individual components fail

### Hybrid System Architecture Overview

```python
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd

@dataclass
class RecommendationResult:
    """Structure for individual recommendation results"""
    user_id: str
    item_id: str
    score: float
    algorithm: str
    confidence: float = 1.0
    explanation: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseRecommender(ABC):
    """Abstract base class for recommendation algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.performance_metrics = {}
        
    @abstractmethod
    def fit(self, interactions: pd.DataFrame, **kwargs) -> None:
        """Train the recommendation model"""
        pass
    
    @abstractmethod
    def predict(self, user_id: str, item_ids: List[str]) -> List[RecommendationResult]:
        """Generate predictions for user-item pairs"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, k: int = 10, 
                 exclude_seen: bool = True) -> List[RecommendationResult]:
        """Generate top-k recommendations for user"""
        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Return algorithm metadata and performance info"""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'performance_metrics': self.performance_metrics
        }

class MockCollaborativeFilteringRecommender(BaseRecommender):
    """Mock collaborative filtering recommender for demonstration"""
    
    def __init__(self):
        super().__init__("CollaborativeFiltering")
        self.user_item_matrix = None
        self.user_similarities = {}
        self.item_means = {}
        
    def fit(self, interactions: pd.DataFrame, **kwargs) -> None:
        """Train CF model"""
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        # Compute item means for rating prediction
        self.item_means = interactions.groupby('item_id')['rating'].mean().to_dict()
        
        # Simple user similarity computation (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        user_vectors = self.user_item_matrix.values
        similarity_matrix = cosine_similarity(user_vectors)
        
        users = self.user_item_matrix.index.tolist()
        for i, user in enumerate(users):
            similarities = {}
            for j, other_user in enumerate(users):
                if i != j:
                    similarities[other_user] = similarity_matrix[i][j]
            self.user_similarities[user] = similarities
        
        self.is_trained = True
    
    def predict(self, user_id: str, item_ids: List[str]) -> List[RecommendationResult]:
        """Generate CF predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        results = []
        for item_id in item_ids:
            # Simple prediction based on item mean and user similarity
            base_score = self.item_means.get(item_id, 3.0)
            
            # Add some variation based on "user similarity"
            user_adjustment = hash(f"{user_id}_{item_id}") % 100 / 100.0 - 0.5
            score = max(0.0, min(5.0, base_score + user_adjustment))
            
            results.append(RecommendationResult(
                user_id=user_id,
                item_id=item_id,
                score=score,
                algorithm=self.name,
                confidence=0.8,
                explanation=f"Based on similar users' preferences"
            ))
        
        return results
    
    def recommend(self, user_id: str, k: int = 10, 
                 exclude_seen: bool = True) -> List[RecommendationResult]:
        """Generate CF recommendations"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get all items
        all_items = list(self.item_means.keys())
        
        # Generate predictions
        predictions = self.predict(user_id, all_items)
        
        # Sort by score and return top-k
        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions[:k]

class MockContentBasedRecommender(BaseRecommender):
    """Mock content-based recommender for demonstration"""
    
    def __init__(self):
        super().__init__("ContentBased")
        self.item_features = {}
        self.user_profiles = {}
        
    def fit(self, interactions: pd.DataFrame, item_features: pd.DataFrame = None, **kwargs) -> None:
        """Train content-based model"""
        # Store item features
        if item_features is not None:
            self.item_features = item_features.set_index('item_id').to_dict('index')
        
        # Build user profiles from interactions
        for user_id in interactions['user_id'].unique():
            user_interactions = interactions[interactions['user_id'] == user_id]
            
            # Simple profile: average of liked item features
            liked_items = user_interactions[user_interactions['rating'] >= 4.0]['item_id']
            
            if len(liked_items) > 0:
                # Mock user profile (would use actual feature vectors)
                profile = {
                    'preferred_genres': ['Action', 'Drama'],  # Mock
                    'avg_rating_given': user_interactions['rating'].mean()
                }
                self.user_profiles[user_id] = profile
        
        self.is_trained = True
    
    def predict(self, user_id: str, item_ids: List[str]) -> List[RecommendationResult]:
        """Generate content-based predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        results = []
        user_profile = self.user_profiles.get(user_id, {})
        
        for item_id in item_ids:
            # Simple content matching
            base_score = 3.0
            
            # Mock content similarity computation
            content_similarity = (hash(f"{user_id}_{item_id}") % 100) / 100.0
            score = base_score + content_similarity * 2.0
            score = max(0.0, min(5.0, score))
            
            results.append(RecommendationResult(
                user_id=user_id,
                item_id=item_id,
                score=score,
                algorithm=self.name,
                confidence=0.9,
                explanation=f"Matches your content preferences"
            ))
        
        return results
    
    def recommend(self, user_id: str, k: int = 10, 
                 exclude_seen: bool = True) -> List[RecommendationResult]:
        """Generate content-based recommendations"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Mock item catalog
        all_items = [f"item_{i}" for i in range(100)]
        
        # Generate predictions
        predictions = self.predict(user_id, all_items)
        
        # Sort by score and return top-k
        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions[:k]

class MockKnowledgeBasedRecommender(BaseRecommender):
    """Mock knowledge-based recommender for demonstration"""
    
    def __init__(self):
        super().__init__("KnowledgeBased")
        self.rules = []
        self.item_metadata = {}
        
    def fit(self, interactions: pd.DataFrame, **kwargs) -> None:
        """Setup knowledge-based rules"""
        # Define simple rules
        self.rules = [
            {"condition": "user_age > 25", "boost": 0.5, "description": "Adult content boost"},
            {"condition": "item_genre == 'Action'", "boost": 0.3, "description": "Popular genre boost"},
            {"condition": "item_year >= 2020", "boost": 0.2, "description": "Recent content boost"}
        ]
        
        self.is_trained = True
    
    def predict(self, user_id: str, item_ids: List[str]) -> List[RecommendationResult]:
        """Generate knowledge-based predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        results = []
        for item_id in item_ids:
            # Base score
            base_score = 3.0
            
            # Apply rules (simplified)
            rule_boost = (hash(f"{user_id}_{item_id}") % 50) / 100.0  # Mock rule application
            score = base_score + rule_boost
            score = max(0.0, min(5.0, score))
            
            results.append(RecommendationResult(
                user_id=user_id,
                item_id=item_id,
                score=score,
                algorithm=self.name,
                confidence=0.7,
                explanation=f"Based on domain knowledge and rules"
            ))
        
        return results
    
    def recommend(self, user_id: str, k: int = 10, 
                 exclude_seen: bool = True) -> List[RecommendationResult]:
        """Generate knowledge-based recommendations"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Mock item catalog
        all_items = [f"item_{i}" for i in range(100)]
        
        # Generate predictions
        predictions = self.predict(user_id, all_items)
        
        # Sort by score and return top-k
        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions[:k]
```

## 2. Hybridization Strategies

### Classification of Hybrid Approaches

There are several established approaches to creating hybrid recommendation systems:

1. **Weighted Hybrid**: Combine scores from multiple algorithms using weights
2. **Switching Hybrid**: Choose one algorithm based on situation
3. **Mixed Hybrid**: Present recommendations from multiple algorithms simultaneously
4. **Feature Combination**: Combine features from different sources into single algorithm
5. **Cascade Hybrid**: Refine recommendations using multiple algorithms in sequence
6. **Feature Augmentation**: Use output of one algorithm as input to another
7. **Meta-level Hybrid**: Use model from one algorithm to generate input for another

### Basic Hybrid Architecture Implementation

```python
from enum import Enum
from typing import Callable

class HybridizationType(Enum):
    WEIGHTED = "weighted"
    SWITCHING = "switching"
    MIXED = "mixed"
    CASCADE = "cascade"
    FEATURE_AUGMENTATION = "feature_augmentation"
    META_LEVEL = "meta_level"

class HybridRecommendationSystem:
    """
    Comprehensive hybrid recommendation system supporting multiple hybridization strategies
    """
    
    def __init__(self, hybridization_type: HybridizationType = HybridizationType.WEIGHTED):
        self.hybridization_type = hybridization_type
        self.recommenders = {}  # algorithm_name -> BaseRecommender
        self.weights = {}  # algorithm_name -> weight
        self.switching_conditions = {}  # condition_name -> (condition_func, algorithm_name)
        self.cascade_order = []  # ordered list of algorithm names
        
        # Performance tracking
        self.algorithm_performance = defaultdict(dict)
        self.combination_history = []
        
    def add_recommender(self, recommender: BaseRecommender, weight: float = 1.0):
        """Add a recommender to the hybrid system"""
        self.recommenders[recommender.name] = recommender
        self.weights[recommender.name] = weight
        
    def set_weights(self, weights: Dict[str, float]):
        """Set combination weights for algorithms"""
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {name: weight/total_weight for name, weight in weights.items()}
        
    def add_switching_condition(self, condition_name: str, 
                              condition_func: Callable[[str, Dict], bool], 
                              algorithm_name: str):
        """Add switching condition for switching hybrid"""
        self.switching_conditions[condition_name] = (condition_func, algorithm_name)
        
    def set_cascade_order(self, algorithm_names: List[str]):
        """Set cascade order for cascade hybrid"""
        self.cascade_order = algorithm_names
        
    def fit(self, interactions: pd.DataFrame, **kwargs):
        """Train all component recommenders"""
        for recommender in self.recommenders.values():
            recommender.fit(interactions, **kwargs)
            
    def recommend(self, user_id: str, k: int = 10, 
                 user_context: Dict[str, Any] = None) -> List[RecommendationResult]:
        """Generate hybrid recommendations"""
        
        if self.hybridization_type == HybridizationType.WEIGHTED:
            return self._weighted_hybrid_recommend(user_id, k)
        elif self.hybridization_type == HybridizationType.SWITCHING:
            return self._switching_hybrid_recommend(user_id, k, user_context)
        elif self.hybridization_type == HybridizationType.MIXED:
            return self._mixed_hybrid_recommend(user_id, k)
        elif self.hybridization_type == HybridizationType.CASCADE:
            return self._cascade_hybrid_recommend(user_id, k)
        else:
            # Default to weighted
            return self._weighted_hybrid_recommend(user_id, k)
    
    def _weighted_hybrid_recommend(self, user_id: str, k: int) -> List[RecommendationResult]:
        """Weighted combination of multiple algorithms"""
        
        # Collect recommendations from all algorithms
        all_recommendations = {}  # item_id -> {algorithm: RecommendationResult}
        
        for algo_name, recommender in self.recommenders.items():
            try:
                recs = recommender.recommend(user_id, k * 2)  # Get more to ensure diversity
                
                for rec in recs:
                    if rec.item_id not in all_recommendations:
                        all_recommendations[rec.item_id] = {}
                    all_recommendations[rec.item_id][algo_name] = rec
                    
            except Exception as e:
                print(f"Error in {algo_name}: {e}")
                continue
        
        # Compute weighted scores
        final_recommendations = []
        
        for item_id, algo_recs in all_recommendations.items():
            weighted_score = 0.0
            total_weight = 0.0
            combined_confidence = 0.0
            explanations = []
            
            for algo_name, rec in algo_recs.items():
                if algo_name in self.weights:
                    weight = self.weights[algo_name]
                    weighted_score += weight * rec.score
                    total_weight += weight
                    combined_confidence += weight * rec.confidence
                    explanations.append(f"{algo_name}: {rec.explanation}")
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                final_confidence = combined_confidence / total_weight
                
                final_recommendations.append(RecommendationResult(
                    user_id=user_id,
                    item_id=item_id,
                    score=final_score,
                    algorithm="WeightedHybrid",
                    confidence=final_confidence,
                    explanation=" | ".join(explanations),
                    metadata={
                        'component_algorithms': list(algo_recs.keys()),
                        'weights_used': {name: self.weights.get(name, 0) for name in algo_recs.keys()}
                    }
                ))
        
        # Sort and return top-k
        final_recommendations.sort(key=lambda x: x.score, reverse=True)
        return final_recommendations[:k]
    
    def _switching_hybrid_recommend(self, user_id: str, k: int, 
                                  user_context: Dict[str, Any] = None) -> List[RecommendationResult]:
        """Switch between algorithms based on conditions"""
        
        if user_context is None:
            user_context = {}
        
        # Evaluate switching conditions
        selected_algorithm = None
        
        for condition_name, (condition_func, algo_name) in self.switching_conditions.items():
            try:
                if condition_func(user_id, user_context):
                    selected_algorithm = algo_name
                    break
            except Exception as e:
                print(f"Error evaluating condition {condition_name}: {e}")
                continue
        
        # Default to first algorithm if no condition matches
        if selected_algorithm is None:
            selected_algorithm = list(self.recommenders.keys())[0]
        
        # Generate recommendations using selected algorithm
        if selected_algorithm in self.recommenders:
            recommendations = self.recommenders[selected_algorithm].recommend(user_id, k)
            
            # Mark as switching hybrid
            for rec in recommendations:
                rec.algorithm = f"SwitchingHybrid({selected_algorithm})"
                rec.metadata['selected_by'] = 'switching_condition'
                rec.metadata['original_algorithm'] = selected_algorithm
            
            return recommendations
        
        return []
    
    def _mixed_hybrid_recommend(self, user_id: str, k: int) -> List[RecommendationResult]:
        """Present recommendations from multiple algorithms"""
        
        mixed_recommendations = []
        algorithms_count = len(self.recommenders)
        
        if algorithms_count == 0:
            return []
        
        # Distribute k recommendations among algorithms
        recs_per_algo = max(1, k // algorithms_count)
        remaining_recs = k % algorithms_count
        
        for i, (algo_name, recommender) in enumerate(self.recommenders.items()):
            # Determine number of recommendations for this algorithm
            current_k = recs_per_algo
            if i < remaining_recs:
                current_k += 1
            
            try:
                recs = recommender.recommend(user_id, current_k)
                
                # Mark as mixed hybrid
                for rec in recs:
                    rec.algorithm = f"MixedHybrid({rec.algorithm})"
                    rec.metadata['mixing_strategy'] = 'proportional'
                
                mixed_recommendations.extend(recs)
                
            except Exception as e:
                print(f"Error in {algo_name}: {e}")
                continue
        
        # Shuffle to mix algorithms
        import random
        random.shuffle(mixed_recommendations)
        
        return mixed_recommendations[:k]
    
    def _cascade_hybrid_recommend(self, user_id: str, k: int) -> List[RecommendationResult]:
        """Cascade through algorithms to refine recommendations"""
        
        if not self.cascade_order:
            return self._weighted_hybrid_recommend(user_id, k)
        
        current_candidates = None
        
        for i, algo_name in enumerate(self.cascade_order):
            if algo_name not in self.recommenders:
                continue
            
            recommender = self.recommenders[algo_name]
            
            if i == 0:
                # First algorithm generates initial candidates
                current_candidates = recommender.recommend(user_id, k * 3)  # Get more candidates
            else:
                # Subsequent algorithms filter/re-rank candidates
                if current_candidates:
                    item_ids = [rec.item_id for rec in current_candidates]
                    refined_predictions = recommender.predict(user_id, item_ids)
                    
                    # Update scores (simple average with previous)
                    refined_dict = {pred.item_id: pred for pred in refined_predictions}
                    
                    for candidate in current_candidates:
                        if candidate.item_id in refined_dict:
                            refined_pred = refined_dict[candidate.item_id]
                            # Average the scores
                            candidate.score = (candidate.score + refined_pred.score) / 2
                            candidate.confidence = min(candidate.confidence, refined_pred.confidence)
                            candidate.explanation += f" | Refined by {algo_name}: {refined_pred.explanation}"
        
        # Sort and return top-k
        if current_candidates:
            current_candidates.sort(key=lambda x: x.score, reverse=True)
            
            # Mark as cascade hybrid
            for rec in current_candidates:
                rec.algorithm = f"CascadeHybrid({' -> '.join(self.cascade_order)})"
                rec.metadata['cascade_order'] = self.cascade_order
            
            return current_candidates[:k]
        
        return []
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict[str, Any]:
        """Provide detailed explanation for a recommendation"""
        
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'hybridization_type': self.hybridization_type.value,
            'component_explanations': {},
            'combination_strategy': ''
        }
        
        # Get explanations from individual algorithms
        for algo_name, recommender in self.recommenders.items():
            try:
                predictions = recommender.predict(user_id, [item_id])
                if predictions:
                    pred = predictions[0]
                    explanation['component_explanations'][algo_name] = {
                        'score': pred.score,
                        'confidence': pred.confidence,
                        'explanation': pred.explanation
                    }
            except Exception as e:
                explanation['component_explanations'][algo_name] = {
                    'error': str(e)
                }
        
        # Add combination strategy explanation
        if self.hybridization_type == HybridizationType.WEIGHTED:
            explanation['combination_strategy'] = f"Weighted combination with weights: {self.weights}"
        elif self.hybridization_type == HybridizationType.SWITCHING:
            explanation['combination_strategy'] = "Algorithm switching based on user context"
        elif self.hybridization_type == HybridizationType.MIXED:
            explanation['combination_strategy'] = "Mixed presentation from multiple algorithms"
        elif self.hybridization_type == HybridizationType.CASCADE:
            explanation['combination_strategy'] = f"Cascade refinement order: {' -> '.join(self.cascade_order)}"
        
        return explanation
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get performance metrics for the hybrid system"""
        
        performance = {
            'hybridization_type': self.hybridization_type.value,
            'num_algorithms': len(self.recommenders),
            'algorithm_weights': self.weights.copy(),
            'individual_performance': {},
            'combination_stats': {
                'total_combinations': len(self.combination_history)
            }
        }
        
        # Individual algorithm performance
        for algo_name, recommender in self.recommenders.items():
            info = recommender.get_algorithm_info()
            performance['individual_performance'][algo_name] = info
        
        return performance
```

## 3. Fundamental Combination Techniques

### Linear Combination Methods

```python
class LinearCombinationHybrid:
    """
    Linear combination hybrid system with advanced weighting strategies
    """
    
    def __init__(self):
        self.recommenders = {}
        self.static_weights = {}
        self.dynamic_weight_functions = {}
        self.performance_history = defaultdict(list)
        
    def add_recommender(self, recommender: BaseRecommender, 
                       static_weight: float = 1.0,
                       dynamic_weight_func: Callable = None):
        """Add recommender with static and optional dynamic weighting"""
        self.recommenders[recommender.name] = recommender
        self.static_weights[recommender.name] = static_weight
        
        if dynamic_weight_func:
            self.dynamic_weight_functions[recommender.name] = dynamic_weight_func
    
    def compute_dynamic_weights(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Compute dynamic weights based on context"""
        dynamic_weights = {}
        
        for algo_name, weight_func in self.dynamic_weight_functions.items():
            try:
                dynamic_weight = weight_func(user_id, context or {})
                dynamic_weights[algo_name] = dynamic_weight
            except Exception as e:
                print(f"Error computing dynamic weight for {algo_name}: {e}")
                dynamic_weights[algo_name] = 1.0
        
        return dynamic_weights
    
    def linear_combination(self, predictions_dict: Dict[str, List[RecommendationResult]], 
                          user_id: str, context: Dict[str, Any] = None) -> List[RecommendationResult]:
        """Perform linear combination of predictions"""
        
        # Compute weights
        dynamic_weights = self.compute_dynamic_weights(user_id, context)
        
        # Collect all unique items
        all_items = set()
        for predictions in predictions_dict.values():
            all_items.update(pred.item_id for pred in predictions)
        
        # Combine predictions
        combined_results = []
        
        for item_id in all_items:
            weighted_score = 0.0
            total_weight = 0.0
            combined_confidence = 0.0
            explanations = []
            participating_algorithms = []
            
            for algo_name, predictions in predictions_dict.items():
                # Find prediction for this item
                item_pred = next((pred for pred in predictions if pred.item_id == item_id), None)
                
                if item_pred:
                    # Compute final weight (static * dynamic)
                    static_w = self.static_weights.get(algo_name, 1.0)
                    dynamic_w = dynamic_weights.get(algo_name, 1.0)
                    final_weight = static_w * dynamic_w
                    
                    weighted_score += final_weight * item_pred.score
                    total_weight += final_weight
                    combined_confidence += final_weight * item_pred.confidence
                    
                    explanations.append(f"{algo_name}(w={final_weight:.2f}): {item_pred.explanation}")
                    participating_algorithms.append(algo_name)
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                final_confidence = combined_confidence / total_weight
                
                combined_results.append(RecommendationResult(
                    user_id=user_id,
                    item_id=item_id,
                    score=final_score,
                    algorithm="LinearCombination",
                    confidence=final_confidence,
                    explanation=" | ".join(explanations),
                    metadata={
                        'participating_algorithms': participating_algorithms,
                        'final_weights': {algo: self.static_weights.get(algo, 1.0) * dynamic_weights.get(algo, 1.0) 
                                        for algo in participating_algorithms},
                        'combination_method': 'linear'
                    }
                ))
        
        return combined_results

class RankBasedCombination:
    """
    Rank-based combination methods for hybrid systems
    """
    
    def __init__(self):
        self.rank_fusion_methods = {
            'borda_count': self._borda_count,
            'reciprocal_rank_fusion': self._reciprocal_rank_fusion,
            'condorcet_fusion': self._condorcet_fusion
        }
    
    def combine_rankings(self, rankings_dict: Dict[str, List[RecommendationResult]], 
                        method: str = 'reciprocal_rank_fusion',
                        k: int = 10) -> List[RecommendationResult]:
        """Combine multiple rankings using specified method"""
        
        if method not in self.rank_fusion_methods:
            raise ValueError(f"Unknown fusion method: {method}")
        
        return self.rank_fusion_methods[method](rankings_dict, k)
    
    def _borda_count(self, rankings_dict: Dict[str, List[RecommendationResult]], 
                    k: int) -> List[RecommendationResult]:
        """Borda count rank fusion"""
        
        item_scores = defaultdict(float)
        item_data = {}  # Store item metadata
        
        for algo_name, ranking in rankings_dict.items():
            max_rank = len(ranking)
            
            for rank, rec in enumerate(ranking):
                # Borda score: (max_rank - rank)
                borda_score = max_rank - rank
                item_scores[rec.item_id] += borda_score
                
                # Store item data (use first occurrence)
                if rec.item_id not in item_data:
                    item_data[rec.item_id] = rec
        
        # Create final ranking
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_ranking = []
        for item_id, score in sorted_items[:k]:
            original_rec = item_data[item_id]
            
            final_ranking.append(RecommendationResult(
                user_id=original_rec.user_id,
                item_id=item_id,
                score=score,
                algorithm="BordaCount",
                confidence=original_rec.confidence,
                explanation=f"Borda count fusion score: {score}",
                metadata={
                    'fusion_method': 'borda_count',
                    'original_algorithm': original_rec.algorithm
                }
            ))
        
        return final_ranking
    
    def _reciprocal_rank_fusion(self, rankings_dict: Dict[str, List[RecommendationResult]], 
                               k: int, rrf_constant: float = 60.0) -> List[RecommendationResult]:
        """Reciprocal Rank Fusion (RRF)"""
        
        item_scores = defaultdict(float)
        item_data = {}
        
        for algo_name, ranking in rankings_dict.items():
            for rank, rec in enumerate(ranking):
                # RRF score: 1 / (rank + constant)
                rrf_score = 1.0 / (rank + 1 + rrf_constant)
                item_scores[rec.item_id] += rrf_score
                
                if rec.item_id not in item_data:
                    item_data[rec.item_id] = rec
        
        # Create final ranking
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_ranking = []
        for item_id, score in sorted_items[:k]:
            original_rec = item_data[item_id]
            
            final_ranking.append(RecommendationResult(
                user_id=original_rec.user_id,
                item_id=item_id,
                score=score,
                algorithm="ReciprocalRankFusion",
                confidence=original_rec.confidence,
                explanation=f"RRF fusion score: {score:.4f}",
                metadata={
                    'fusion_method': 'reciprocal_rank_fusion',
                    'rrf_constant': rrf_constant,
                    'original_algorithm': original_rec.algorithm
                }
            ))
        
        return final_ranking
    
    def _condorcet_fusion(self, rankings_dict: Dict[str, List[RecommendationResult]], 
                         k: int) -> List[RecommendationResult]:
        """Condorcet-based rank fusion"""
        
        # Collect all items
        all_items = set()
        item_data = {}
        
        for algo_name, ranking in rankings_dict.items():
            for rec in ranking:
                all_items.add(rec.item_id)
                if rec.item_id not in item_data:
                    item_data[rec.item_id] = rec
        
        all_items = list(all_items)
        
        # Compute pairwise wins
        pairwise_wins = defaultdict(int)
        
        for algo_name, ranking in rankings_dict.items():
            # Create rank mapping
            rank_map = {rec.item_id: rank for rank, rec in enumerate(ranking)}
            
            # Compare all pairs
            for i, item1 in enumerate(all_items):
                for item2 in all_items[i+1:]:
                    rank1 = rank_map.get(item1, float('inf'))
                    rank2 = rank_map.get(item2, float('inf'))
                    
                    if rank1 < rank2:  # item1 ranked higher
                        pairwise_wins[item1] += 1
                    elif rank2 < rank1:  # item2 ranked higher
                        pairwise_wins[item2] += 1
        
        # Sort by pairwise wins
        sorted_items = sorted(pairwise_wins.items(), key=lambda x: x[1], reverse=True)
        
        final_ranking = []
        for item_id, wins in sorted_items[:k]:
            original_rec = item_data[item_id]
            
            final_ranking.append(RecommendationResult(
                user_id=original_rec.user_id,
                item_id=item_id,
                score=float(wins),
                algorithm="CondorcetFusion",
                confidence=original_rec.confidence,
                explanation=f"Condorcet wins: {wins}",
                metadata={
                    'fusion_method': 'condorcet_fusion',
                    'pairwise_wins': wins,
                    'original_algorithm': original_rec.algorithm
                }
            ))
        
        return final_ranking
```

## 4. Practical Implementation Example

### Complete Hybrid System Demo

```python
def create_sample_data():
    """Create sample interaction data for testing"""
    import random
    
    users = [f"user_{i}" for i in range(50)]
    items = [f"item_{i}" for i in range(200)]
    
    interactions = []
    for user in users:
        # Each user rates 10-30 items
        num_ratings = random.randint(10, 30)
        user_items = random.sample(items, num_ratings)
        
        for item in user_items:
            rating = random.uniform(1.0, 5.0)
            interactions.append({
                'user_id': user,
                'item_id': item,
                'rating': rating
            })
    
    return pd.DataFrame(interactions)

def demo_hybrid_system():
    """Demonstrate hybrid recommendation system"""
    
    print("=== Hybrid Recommendation System Demo ===")
    print()
    
    # Create sample data
    interactions_df = create_sample_data()
    print(f"Created sample data: {len(interactions_df)} interactions")
    print()
    
    # Create individual recommenders
    cf_recommender = MockCollaborativeFilteringRecommender()
    cb_recommender = MockContentBasedRecommender()
    kb_recommender = MockKnowledgeBasedRecommender()
    
    # Train individual recommenders
    print("Training individual recommenders...")
    cf_recommender.fit(interactions_df)
    cb_recommender.fit(interactions_df)
    kb_recommender.fit(interactions_df)
    print("Training completed.")
    print()
    
    # Test different hybrid approaches
    test_user = "user_1"
    
    # 1. Weighted Hybrid
    print("1. WEIGHTED HYBRID SYSTEM")
    print("-" * 40)
    
    weighted_hybrid = HybridRecommendationSystem(HybridizationType.WEIGHTED)
    weighted_hybrid.add_recommender(cf_recommender, weight=0.5)
    weighted_hybrid.add_recommender(cb_recommender, weight=0.3)
    weighted_hybrid.add_recommender(kb_recommender, weight=0.2)
    
    weighted_recs = weighted_hybrid.recommend(test_user, k=5)
    
    for i, rec in enumerate(weighted_recs, 1):
        print(f"{i}. Item: {rec.item_id}")
        print(f"   Score: {rec.score:.3f}")
        print(f"   Confidence: {rec.confidence:.3f}")
        print(f"   Explanation: {rec.explanation}")
        print()
    
    # 2. Switching Hybrid
    print("2. SWITCHING HYBRID SYSTEM")
    print("-" * 40)
    
    switching_hybrid = HybridRecommendationSystem(HybridizationType.SWITCHING)
    switching_hybrid.add_recommender(cf_recommender)
    switching_hybrid.add_recommender(cb_recommender)
    switching_hybrid.add_recommender(kb_recommender)
    
    # Define switching conditions
    def is_new_user(user_id, context):
        # Mock condition: users with ID ending in odd numbers are "new"
        return int(user_id.split('_')[1]) % 2 == 1
    
    def has_many_ratings(user_id, context):
        # Mock condition: users with even IDs have "many ratings"
        return int(user_id.split('_')[1]) % 2 == 0
    
    switching_hybrid.add_switching_condition("new_user", is_new_user, "ContentBased")
    switching_hybrid.add_switching_condition("experienced_user", has_many_ratings, "CollaborativeFiltering")
    
    switching_recs = switching_hybrid.recommend(test_user, k=5, user_context={})
    
    for i, rec in enumerate(switching_recs, 1):
        print(f"{i}. Item: {rec.item_id}")
        print(f"   Score: {rec.score:.3f}")
        print(f"   Algorithm: {rec.algorithm}")
        print(f"   Explanation: {rec.explanation}")
        print()
    
    # 3. Rank-based Fusion
    print("3. RANK-BASED FUSION")
    print("-" * 40)
    
    # Get rankings from individual algorithms
    cf_recs = cf_recommender.recommend(test_user, k=10)
    cb_recs = cb_recommender.recommend(test_user, k=10)
    kb_recs = kb_recommender.recommend(test_user, k=10)
    
    rankings = {
        'CollaborativeFiltering': cf_recs,
        'ContentBased': cb_recs,
        'KnowledgeBased': kb_recs
    }
    
    rank_combiner = RankBasedCombination()
    
    # Try different fusion methods
    for method in ['reciprocal_rank_fusion', 'borda_count']:
        print(f"\n{method.upper().replace('_', ' ')}:")
        fused_recs = rank_combiner.combine_rankings(rankings, method=method, k=5)
        
        for i, rec in enumerate(fused_recs, 1):
            print(f"  {i}. Item: {rec.item_id}, Score: {rec.score:.3f}")
    
    print()
    
    # 4. System Performance Analysis
    print("4. SYSTEM PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    performance = weighted_hybrid.get_system_performance()
    print(f"Hybridization Type: {performance['hybridization_type']}")
    print(f"Number of Algorithms: {performance['num_algorithms']}")
    print(f"Algorithm Weights: {performance['algorithm_weights']}")
    print()
    
    # 5. Explanation Example
    print("5. RECOMMENDATION EXPLANATION")
    print("-" * 40)
    
    if weighted_recs:
        explanation = weighted_hybrid.explain_recommendation(test_user, weighted_recs[0].item_id)
        print(f"Explaining recommendation for {explanation['item_id']}:")
        print(f"Hybridization: {explanation['hybridization_type']}")
        print(f"Strategy: {explanation['combination_strategy']}")
        print("Component explanations:")
        
        for algo, details in explanation['component_explanations'].items():
            if 'error' not in details:
                print(f"  {algo}: Score={details['score']:.3f}, "
                      f"Confidence={details['confidence']:.3f}")
                print(f"    {details['explanation']}")

if __name__ == "__main__":
    demo_hybrid_system()
```

## 5. Study Questions

### Beginner Level

1. What are the main motivations for creating hybrid recommendation systems?
2. List and briefly explain the seven types of hybridization strategies.
3. What is the difference between weighted and switching hybrid approaches?
4. How does a cascade hybrid system work?
5. What are the advantages of rank-based fusion methods?

### Intermediate Level

6. Implement a hybrid system that combines collaborative filtering and content-based recommendations for a movie recommendation scenario.
7. How would you design dynamic weights that adapt based on user context and algorithm performance?
8. What are the challenges in implementing a mixed hybrid system that presents results from multiple algorithms?
9. Compare linear combination and rank-based fusion approaches in terms of effectiveness and computational complexity.
10. How would you handle the situation where some component algorithms fail in a hybrid system?

### Advanced Level

11. Implement a meta-level hybrid system where one algorithm's model is used to generate features for another algorithm.
12. Design a hybrid system that can automatically learn optimal combination weights from user feedback.
13. How would you implement feature augmentation hybridization where the output of one recommender becomes input to another?
14. Create a hybrid system that can dynamically switch between different hybridization strategies based on the recommendation context.
15. Implement a robust hybrid system that can handle algorithm failures gracefully and maintain recommendation quality.

### Tricky Questions

16. How would you design a hybrid system that can adapt its combination strategy in real-time based on user interactions and algorithm performance?
17. What are the challenges in explaining recommendations from complex hybrid systems, and how would you address them?
18. How would you handle the bias amplification problem that can occur when combining multiple biased algorithms?
19. Design a hybrid system that can work with algorithms having different output formats (scores, rankings, binary classifications).
20. How would you implement a hybrid system that can learn from multi-armed bandit feedback to optimize the combination strategy over time?

## Key Takeaways

1. **Hybrid systems** combine strengths of multiple algorithms while mitigating individual weaknesses
2. **Multiple hybridization strategies** exist, each suitable for different scenarios and requirements
3. **Weighted combination** is simple but effective for many applications
4. **Switching strategies** can adapt to different user or item contexts
5. **Rank-based fusion** methods are robust and don't require score normalization
6. **Dynamic weighting** can improve performance by adapting to changing conditions
7. **Explanation capabilities** become more complex but are crucial for user trust

## Next Session Preview

In Day 5.2, we'll explore **Combination Strategies and Architectures**, covering:
- Advanced combination architectures
- Score normalization techniques
- Confidence-based weighting
- Ensemble learning approaches
- Performance optimization strategies