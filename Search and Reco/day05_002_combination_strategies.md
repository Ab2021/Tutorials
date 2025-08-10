# Day 5.2: Combination Strategies and Architectures

## Learning Objectives
By the end of this session, you will:
- Master advanced combination architectures for hybrid systems
- Implement sophisticated score normalization techniques
- Apply confidence-based weighting strategies
- Use ensemble learning approaches for recommendation fusion
- Optimize performance of complex hybrid architectures

## 1. Advanced Combination Architectures

### Multi-Layer Hybrid Architecture

```python
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CombinationLayer:
    """Represents a layer in multi-layer hybrid architecture"""
    name: str
    combination_method: str  # 'weighted', 'learned', 'voting', 'stacking'
    input_algorithms: List[str]
    output_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_trained: bool = False
    model: Any = None

class MultiLayerHybridSystem:
    """
    Multi-layer hybrid recommendation architecture
    Supports hierarchical combination of algorithms
    """
    
    def __init__(self):
        self.base_recommenders = {}  # algorithm_name -> BaseRecommender
        self.layers = []  # List of CombinationLayer
        self.score_normalizers = {}  # algorithm_name -> normalizer
        self.confidence_estimators = {}  # algorithm_name -> confidence_function
        
        # Performance tracking
        self.layer_performance = defaultdict(dict)
        self.training_history = []
        
    def add_base_recommender(self, recommender, normalizer_type: str = 'minmax'):
        """Add base recommender with score normalization"""
        self.base_recommenders[recommender.name] = recommender
        
        # Set up score normalizer
        if normalizer_type == 'minmax':
            self.score_normalizers[recommender.name] = MinMaxScaler()
        elif normalizer_type == 'standard':
            self.score_normalizers[recommender.name] = StandardScaler()
        else:
            self.score_normalizers[recommender.name] = None
    
    def add_combination_layer(self, layer: CombinationLayer):
        """Add combination layer to the architecture"""
        self.layers.append(layer)
    
    def set_confidence_estimator(self, algorithm_name: str, 
                                confidence_func: Callable[[List[float]], float]):
        """Set confidence estimation function for an algorithm"""
        self.confidence_estimators[algorithm_name] = confidence_func
    
    def fit(self, interactions: pd.DataFrame, validation_data: pd.DataFrame = None, **kwargs):
        """Train the multi-layer hybrid system"""
        
        print("Training base recommenders...")
        # Train base recommenders
        for name, recommender in self.base_recommenders.items():
            print(f"  Training {name}...")
            recommender.fit(interactions, **kwargs)
        
        print("Fitting score normalizers...")
        # Fit score normalizers using training data predictions
        self._fit_score_normalizers(interactions)
        
        print("Training combination layers...")
        # Train combination layers
        for layer in self.layers:
            print(f"  Training layer: {layer.name}")
            self._train_combination_layer(layer, interactions, validation_data)
    
    def _fit_score_normalizers(self, interactions: pd.DataFrame):
        """Fit score normalizers on training predictions"""
        
        # Sample users for normalization fitting
        sample_users = interactions['user_id'].unique()[:50]  # Use subset for efficiency
        
        for algo_name, normalizer in self.score_normalizers.items():
            if normalizer is None:
                continue
            
            recommender = self.base_recommenders[algo_name]
            all_scores = []
            
            for user_id in sample_users:
                try:
                    # Get user's interacted items
                    user_items = interactions[interactions['user_id'] == user_id]['item_id'].tolist()
                    if user_items:
                        predictions = recommender.predict(user_id, user_items[:20])  # Sample items
                        scores = [pred.score for pred in predictions]
                        all_scores.extend(scores)
                except Exception as e:
                    continue
            
            if all_scores:
                normalizer.fit(np.array(all_scores).reshape(-1, 1))
    
    def _train_combination_layer(self, layer: CombinationLayer, 
                                interactions: pd.DataFrame, 
                                validation_data: pd.DataFrame = None):
        """Train a specific combination layer"""
        
        if layer.combination_method == 'weighted':
            # Static weights - no training needed
            layer.is_trained = True
            
        elif layer.combination_method == 'learned':
            # Train machine learning model for combination
            self._train_learned_combination(layer, interactions, validation_data)
            
        elif layer.combination_method == 'voting':
            # Voting - no training needed
            layer.is_trained = True
            
        elif layer.combination_method == 'stacking':
            # Train stacking model
            self._train_stacking_combination(layer, interactions, validation_data)
    
    def _train_learned_combination(self, layer: CombinationLayer, 
                                  interactions: pd.DataFrame, 
                                  validation_data: pd.DataFrame = None):
        """Train learned combination model"""
        
        # Use validation data if available, otherwise use portion of training data
        if validation_data is not None:
            training_interactions = validation_data
        else:
            # Use 20% of training data for meta-learning
            sample_size = min(1000, len(interactions) // 5)
            training_interactions = interactions.sample(n=sample_size)
        
        # Prepare training data
        X_train = []  # Features: predictions from input algorithms
        y_train = []  # Target: actual ratings
        
        for _, row in training_interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # Get predictions from input algorithms
            algorithm_predictions = []
            
            for algo_name in layer.input_algorithms:
                if algo_name in self.base_recommenders:
                    try:
                        predictions = self.base_recommenders[algo_name].predict(user_id, [item_id])
                        if predictions:
                            pred_score = predictions[0].score
                            # Normalize score
                            if self.score_normalizers[algo_name] is not None:
                                pred_score = self.score_normalizers[algo_name].transform([[pred_score]])[0][0]
                            algorithm_predictions.append(pred_score)
                        else:
                            algorithm_predictions.append(0.0)
                    except Exception:
                        algorithm_predictions.append(0.0)
                else:
                    algorithm_predictions.append(0.0)
            
            if len(algorithm_predictions) == len(layer.input_algorithms):
                X_train.append(algorithm_predictions)
                y_train.append(actual_rating)
        
        if len(X_train) > 10:  # Minimum training samples
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train ensemble model
            model_type = layer.parameters.get('model_type', 'random_forest')
            
            if model_type == 'random_forest':
                layer.model = RandomForestRegressor(
                    n_estimators=layer.parameters.get('n_estimators', 100),
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                layer.model = GradientBoostingRegressor(
                    n_estimators=layer.parameters.get('n_estimators', 100),
                    random_state=42
                )
            else:
                layer.model = LinearRegression()
            
            layer.model.fit(X_train, y_train)
            layer.is_trained = True
        else:
            print(f"  Warning: Not enough training data for layer {layer.name}")
    
    def _train_stacking_combination(self, layer: CombinationLayer, 
                                   interactions: pd.DataFrame, 
                                   validation_data: pd.DataFrame = None):
        """Train stacking combination with cross-validation"""
        # Similar to learned combination but with cross-validation
        # For brevity, using same implementation as learned combination
        self._train_learned_combination(layer, interactions, validation_data)
    
    def predict(self, user_id: str, item_ids: List[str]) -> List[RecommendationResult]:
        """Generate predictions using multi-layer architecture"""
        
        # Get base predictions
        base_predictions = self._get_base_predictions(user_id, item_ids)
        
        # Process through combination layers
        current_predictions = base_predictions
        
        for layer in self.layers:
            current_predictions = self._apply_combination_layer(
                layer, user_id, item_ids, current_predictions
            )
        
        return current_predictions
    
    def _get_base_predictions(self, user_id: str, item_ids: List[str]) -> Dict[str, List[RecommendationResult]]:
        """Get predictions from all base recommenders"""
        
        base_predictions = {}
        
        for algo_name, recommender in self.base_recommenders.items():
            try:
                predictions = recommender.predict(user_id, item_ids)
                
                # Normalize scores
                if self.score_normalizers[algo_name] is not None:
                    for pred in predictions:
                        normalized_score = self.score_normalizers[algo_name].transform([[pred.score]])[0][0]
                        pred.score = normalized_score
                
                # Estimate confidence
                if algo_name in self.confidence_estimators:
                    scores = [pred.score for pred in predictions]
                    confidence = self.confidence_estimators[algo_name](scores)
                    for pred in predictions:
                        pred.confidence = confidence
                
                base_predictions[algo_name] = predictions
                
            except Exception as e:
                print(f"Error getting predictions from {algo_name}: {e}")
                # Create dummy predictions
                dummy_predictions = []
                for item_id in item_ids:
                    dummy_predictions.append(RecommendationResult(
                        user_id=user_id,
                        item_id=item_id,
                        score=0.0,
                        algorithm=algo_name,
                        confidence=0.0,
                        explanation="Error in prediction"
                    ))
                base_predictions[algo_name] = dummy_predictions
        
        return base_predictions
    
    def _apply_combination_layer(self, layer: CombinationLayer, user_id: str, 
                                item_ids: List[str], 
                                input_predictions: Dict[str, List[RecommendationResult]]) -> Dict[str, List[RecommendationResult]]:
        """Apply combination layer to input predictions"""
        
        if not layer.is_trained:
            print(f"Warning: Layer {layer.name} not trained, skipping")
            return input_predictions
        
        combined_predictions = []
        
        for i, item_id in enumerate(item_ids):
            # Collect predictions for this item from input algorithms
            item_predictions = {}
            
            for algo_name in layer.input_algorithms:
                if algo_name in input_predictions and i < len(input_predictions[algo_name]):
                    item_predictions[algo_name] = input_predictions[algo_name][i]
            
            # Apply combination method
            if layer.combination_method == 'weighted':
                combined_pred = self._weighted_combination(layer, user_id, item_id, item_predictions)
            elif layer.combination_method == 'learned':
                combined_pred = self._learned_combination(layer, user_id, item_id, item_predictions)
            elif layer.combination_method == 'voting':
                combined_pred = self._voting_combination(layer, user_id, item_id, item_predictions)
            elif layer.combination_method == 'stacking':
                combined_pred = self._stacking_combination(layer, user_id, item_id, item_predictions)
            else:
                # Default to simple average
                combined_pred = self._simple_average_combination(layer, user_id, item_id, item_predictions)
            
            combined_predictions.append(combined_pred)
        
        # Return in same format as input
        return {layer.output_name: combined_predictions}
    
    def _weighted_combination(self, layer: CombinationLayer, user_id: str, item_id: str, 
                             item_predictions: Dict[str, RecommendationResult]) -> RecommendationResult:
        """Weighted combination of predictions"""
        
        weights = layer.parameters.get('weights', {})
        total_score = 0.0
        total_weight = 0.0
        explanations = []
        
        for algo_name, pred in item_predictions.items():
            weight = weights.get(algo_name, 1.0)
            total_score += weight * pred.score
            total_weight += weight
            explanations.append(f"{algo_name}(w={weight:.2f}): {pred.score:.3f}")
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        return RecommendationResult(
            user_id=user_id,
            item_id=item_id,
            score=final_score,
            algorithm=f"WeightedLayer({layer.name})",
            confidence=0.8,
            explanation=" | ".join(explanations),
            metadata={'layer_name': layer.name, 'combination_method': 'weighted'}
        )
    
    def _learned_combination(self, layer: CombinationLayer, user_id: str, item_id: str,
                            item_predictions: Dict[str, RecommendationResult]) -> RecommendationResult:
        """Learned combination using trained model"""
        
        if layer.model is None:
            return self._simple_average_combination(layer, user_id, item_id, item_predictions)
        
        # Prepare features
        features = []
        for algo_name in layer.input_algorithms:
            if algo_name in item_predictions:
                features.append(item_predictions[algo_name].score)
            else:
                features.append(0.0)
        
        # Predict using trained model
        if len(features) == len(layer.input_algorithms):
            predicted_score = layer.model.predict([features])[0]
            predicted_score = max(0.0, min(5.0, predicted_score))  # Clamp to valid range
        else:
            predicted_score = 0.0
        
        # Get feature importance if available
        explanation = f"Learned combination: {predicted_score:.3f}"
        if hasattr(layer.model, 'feature_importances_'):
            importances = layer.model.feature_importances_
            imp_str = ", ".join([f"{layer.input_algorithms[i]}:{imp:.2f}" 
                               for i, imp in enumerate(importances)])
            explanation += f" (importance: {imp_str})"
        
        return RecommendationResult(
            user_id=user_id,
            item_id=item_id,
            score=predicted_score,
            algorithm=f"LearnedLayer({layer.name})",
            confidence=0.9,
            explanation=explanation,
            metadata={'layer_name': layer.name, 'combination_method': 'learned'}
        )
    
    def _voting_combination(self, layer: CombinationLayer, user_id: str, item_id: str,
                           item_predictions: Dict[str, RecommendationResult]) -> RecommendationResult:
        """Voting-based combination"""
        
        threshold = layer.parameters.get('threshold', 3.0)
        
        # Count votes above threshold
        votes = 0
        total_algorithms = len(item_predictions)
        
        for pred in item_predictions.values():
            if pred.score >= threshold:
                votes += 1
        
        # Score is the proportion of positive votes
        final_score = votes / total_algorithms if total_algorithms > 0 else 0.0
        
        return RecommendationResult(
            user_id=user_id,
            item_id=item_id,
            score=final_score,
            algorithm=f"VotingLayer({layer.name})",
            confidence=0.7,
            explanation=f"Voting: {votes}/{total_algorithms} algorithms voted positive",
            metadata={'layer_name': layer.name, 'combination_method': 'voting'}
        )
    
    def _stacking_combination(self, layer: CombinationLayer, user_id: str, item_id: str,
                             item_predictions: Dict[str, RecommendationResult]) -> RecommendationResult:
        """Stacking combination (similar to learned for this implementation)"""
        return self._learned_combination(layer, user_id, item_id, item_predictions)
    
    def _simple_average_combination(self, layer: CombinationLayer, user_id: str, item_id: str,
                                   item_predictions: Dict[str, RecommendationResult]) -> RecommendationResult:
        """Simple average combination (fallback)"""
        
        if not item_predictions:
            return RecommendationResult(
                user_id=user_id,
                item_id=item_id,
                score=0.0,
                algorithm=f"AverageLayer({layer.name})",
                confidence=0.1,
                explanation="No predictions available"
            )
        
        average_score = sum(pred.score for pred in item_predictions.values()) / len(item_predictions)
        
        return RecommendationResult(
            user_id=user_id,
            item_id=item_id,
            score=average_score,
            algorithm=f"AverageLayer({layer.name})",
            confidence=0.6,
            explanation=f"Simple average of {len(item_predictions)} algorithms",
            metadata={'layer_name': layer.name, 'combination_method': 'average'}
        )
    
    def recommend(self, user_id: str, k: int = 10) -> List[RecommendationResult]:
        """Generate recommendations using multi-layer architecture"""
        
        # Get candidate items (simplified - in practice would be more sophisticated)
        candidate_items = [f"item_{i}" for i in range(100)]  # Mock candidates
        
        # Get predictions
        predictions = self.predict(user_id, candidate_items)
        
        # Extract final layer predictions
        if self.layers:
            final_layer_name = self.layers[-1].output_name
            if final_layer_name in predictions:
                final_predictions = predictions[final_layer_name]
            else:
                # Fallback to base predictions
                final_predictions = []
                for preds_list in predictions.values():
                    final_predictions.extend(preds_list)
        else:
            # No layers, use base predictions
            final_predictions = []
            for preds_list in predictions.values():
                final_predictions.extend(preds_list)
        
        # Sort and return top-k
        final_predictions.sort(key=lambda x: x.score, reverse=True)
        return final_predictions[:k]
```

## 2. Score Normalization Techniques

### Advanced Normalization Strategies

```python
class AdvancedScoreNormalizer:
    """
    Advanced score normalization techniques for hybrid systems
    """
    
    def __init__(self):
        self.normalization_methods = {
            'minmax': self._minmax_normalize,
            'zscore': self._zscore_normalize,
            'robust': self._robust_normalize,
            'quantile': self._quantile_normalize,
            'sigmoid': self._sigmoid_normalize,
            'tanh': self._tanh_normalize
        }
        
        self.fitted_parameters = {}
    
    def fit(self, scores: List[float], method: str = 'minmax') -> None:
        """Fit normalization parameters"""
        
        if method not in self.normalization_methods:
            raise ValueError(f"Unknown normalization method: {method}")
        
        scores_array = np.array(scores)
        
        if method == 'minmax':
            self.fitted_parameters['min'] = np.min(scores_array)
            self.fitted_parameters['max'] = np.max(scores_array)
        
        elif method == 'zscore':
            self.fitted_parameters['mean'] = np.mean(scores_array)
            self.fitted_parameters['std'] = np.std(scores_array)
        
        elif method == 'robust':
            self.fitted_parameters['median'] = np.median(scores_array)
            self.fitted_parameters['mad'] = np.median(np.abs(scores_array - self.fitted_parameters['median']))
        
        elif method == 'quantile':
            self.fitted_parameters['q25'] = np.percentile(scores_array, 25)
            self.fitted_parameters['q75'] = np.percentile(scores_array, 75)
        
        elif method in ['sigmoid', 'tanh']:
            self.fitted_parameters['mean'] = np.mean(scores_array)
            self.fitted_parameters['std'] = np.std(scores_array)
        
        self.fitted_parameters['method'] = method
    
    def transform(self, scores: List[float]) -> List[float]:
        """Transform scores using fitted parameters"""
        
        if 'method' not in self.fitted_parameters:
            raise ValueError("Normalizer must be fitted before transform")
        
        method = self.fitted_parameters['method']
        return self.normalization_methods[method](scores)
    
    def fit_transform(self, scores: List[float], method: str = 'minmax') -> List[float]:
        """Fit and transform in one step"""
        self.fit(scores, method)
        return self.transform(scores)
    
    def _minmax_normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalization to [0, 1]"""
        min_val = self.fitted_parameters['min']
        max_val = self.fitted_parameters['max']
        
        if max_val == min_val:
            return [0.5] * len(scores)  # All scores are the same
        
        return [(score - min_val) / (max_val - min_val) for score in scores]
    
    def _zscore_normalize(self, scores: List[float]) -> List[float]:
        """Z-score normalization"""
        mean = self.fitted_parameters['mean']
        std = self.fitted_parameters['std']
        
        if std == 0:
            return [0.0] * len(scores)
        
        return [(score - mean) / std for score in scores]
    
    def _robust_normalize(self, scores: List[float]) -> List[float]:
        """Robust normalization using median and MAD"""
        median = self.fitted_parameters['median']
        mad = self.fitted_parameters['mad']
        
        if mad == 0:
            return [0.0] * len(scores)
        
        return [(score - median) / mad for score in scores]
    
    def _quantile_normalize(self, scores: List[float]) -> List[float]:
        """Quantile-based normalization"""
        q25 = self.fitted_parameters['q25']
        q75 = self.fitted_parameters['q75']
        
        if q75 == q25:
            return [0.5] * len(scores)
        
        normalized = []
        for score in scores:
            if score <= q25:
                norm_score = 0.0
            elif score >= q75:
                norm_score = 1.0
            else:
                norm_score = (score - q25) / (q75 - q25)
            normalized.append(norm_score)
        
        return normalized
    
    def _sigmoid_normalize(self, scores: List[float]) -> List[float]:
        """Sigmoid normalization"""
        mean = self.fitted_parameters['mean']
        std = self.fitted_parameters['std']
        
        if std == 0:
            return [0.5] * len(scores)
        
        # Sigmoid function: 1 / (1 + exp(-(x - mean) / std))
        normalized = []
        for score in scores:
            z = (score - mean) / std
            sigmoid_val = 1 / (1 + np.exp(-z))
            normalized.append(sigmoid_val)
        
        return normalized
    
    def _tanh_normalize(self, scores: List[float]) -> List[float]:
        """Tanh normalization to [-1, 1], then shifted to [0, 1]"""
        mean = self.fitted_parameters['mean']
        std = self.fitted_parameters['std']
        
        if std == 0:
            return [0.5] * len(scores)
        
        normalized = []
        for score in scores:
            z = (score - mean) / std
            tanh_val = np.tanh(z)
            # Shift from [-1, 1] to [0, 1]
            normalized_val = (tanh_val + 1) / 2
            normalized.append(normalized_val)
        
        return normalized

class DistributionAwareNormalizer:
    """
    Normalization that considers score distributions
    """
    
    def __init__(self):
        self.score_distributions = {}  # algorithm -> distribution parameters
        
    def fit_distributions(self, algorithm_scores: Dict[str, List[float]]):
        """Fit score distributions for each algorithm"""
        
        for algorithm, scores in algorithm_scores.items():
            scores_array = np.array(scores)
            
            # Estimate distribution parameters
            distribution_params = {
                'mean': np.mean(scores_array),
                'std': np.std(scores_array),
                'median': np.median(scores_array),
                'q25': np.percentile(scores_array, 25),
                'q75': np.percentile(scores_array, 75),
                'min': np.min(scores_array),
                'max': np.max(scores_array),
                'skewness': self._compute_skewness(scores_array),
                'kurtosis': self._compute_kurtosis(scores_array)
            }
            
            self.score_distributions[algorithm] = distribution_params
    
    def normalize_to_common_distribution(self, algorithm_scores: Dict[str, List[float]], 
                                       target_distribution: str = 'uniform') -> Dict[str, List[float]]:
        """Normalize all algorithms to common distribution"""
        
        normalized_scores = {}
        
        for algorithm, scores in algorithm_scores.items():
            if algorithm not in self.score_distributions:
                normalized_scores[algorithm] = scores
                continue
            
            params = self.score_distributions[algorithm]
            
            if target_distribution == 'uniform':
                # Convert to uniform [0, 1] distribution
                normalized = self._to_uniform_distribution(scores, params)
            elif target_distribution == 'normal':
                # Convert to standard normal distribution
                normalized = self._to_normal_distribution(scores, params)
            else:
                normalized = scores
            
            normalized_scores[algorithm] = normalized
        
        return normalized_scores
    
    def _to_uniform_distribution(self, scores: List[float], params: Dict) -> List[float]:
        """Convert scores to uniform distribution"""
        min_val = params['min']
        max_val = params['max']
        
        if max_val == min_val:
            return [0.5] * len(scores)
        
        return [(score - min_val) / (max_val - min_val) for score in scores]
    
    def _to_normal_distribution(self, scores: List[float], params: Dict) -> List[float]:
        """Convert scores to standard normal distribution"""
        mean = params['mean']
        std = params['std']
        
        if std == 0:
            return [0.0] * len(scores)
        
        return [(score - mean) / std for score in scores]
    
    def _compute_skewness(self, scores: np.ndarray) -> float:
        """Compute skewness of score distribution"""
        if len(scores) < 3:
            return 0.0
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((scores - mean) / std) ** 3)
        return skewness
    
    def _compute_kurtosis(self, scores: np.ndarray) -> float:
        """Compute kurtosis of score distribution"""
        if len(scores) < 4:
            return 0.0
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((scores - mean) / std) ** 4) - 3  # Subtract 3 for excess kurtosis
        return kurtosis
```

## 3. Confidence-Based Weighting

### Dynamic Confidence Estimation

```python
class ConfidenceBasedWeighting:
    """
    Dynamic weighting based on algorithm confidence
    """
    
    def __init__(self):
        self.confidence_models = {}  # algorithm -> confidence model
        self.confidence_history = defaultdict(list)
        self.performance_tracker = defaultdict(dict)
        
    def add_confidence_model(self, algorithm_name: str, confidence_model):
        """Add confidence estimation model for algorithm"""
        self.confidence_models[algorithm_name] = confidence_model
    
    def estimate_prediction_confidence(self, algorithm_name: str, 
                                     user_id: str, item_id: str,
                                     prediction_context: Dict[str, Any]) -> float:
        """Estimate confidence for a specific prediction"""
        
        if algorithm_name not in self.confidence_models:
            return 0.5  # Default confidence
        
        confidence_model = self.confidence_models[algorithm_name]
        
        try:
            confidence = confidence_model.predict_confidence(
                user_id, item_id, prediction_context
            )
            return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except Exception as e:
            print(f"Error estimating confidence for {algorithm_name}: {e}")
            return 0.1  # Low confidence on error
    
    def compute_dynamic_weights(self, algorithm_predictions: Dict[str, List[RecommendationResult]], 
                               user_id: str, weighting_strategy: str = 'confidence_based') -> Dict[str, float]:
        """Compute dynamic weights based on prediction confidence"""
        
        weights = {}
        
        if weighting_strategy == 'confidence_based':
            # Weight by average confidence
            for algorithm, predictions in algorithm_predictions.items():
                if predictions:
                    avg_confidence = sum(pred.confidence for pred in predictions) / len(predictions)
                    weights[algorithm] = avg_confidence
                else:
                    weights[algorithm] = 0.0
        
        elif weighting_strategy == 'performance_based':
            # Weight by historical performance
            for algorithm in algorithm_predictions.keys():
                if algorithm in self.performance_tracker:
                    recent_performance = self.performance_tracker[algorithm].get('recent_accuracy', 0.5)
                    weights[algorithm] = recent_performance
                else:
                    weights[algorithm] = 0.5
        
        elif weighting_strategy == 'entropy_based':
            # Weight by prediction entropy (diversity)
            for algorithm, predictions in algorithm_predictions.items():
                if predictions:
                    scores = [pred.score for pred in predictions]
                    entropy = self._compute_entropy(scores)
                    # Higher entropy = more diverse predictions = higher weight
                    weights[algorithm] = entropy
                else:
                    weights[algorithm] = 0.0
        
        elif weighting_strategy == 'agreement_based':
            # Weight by agreement with other algorithms
            weights = self._compute_agreement_weights(algorithm_predictions)
        
        else:
            # Uniform weights
            num_algorithms = len(algorithm_predictions)
            for algorithm in algorithm_predictions.keys():
                weights[algorithm] = 1.0 / num_algorithms
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {algo: weight / total_weight for algo, weight in weights.items()}
        
        return weights
    
    def _compute_entropy(self, scores: List[float]) -> float:
        """Compute entropy of score distribution"""
        if not scores or len(set(scores)) == 1:
            return 0.0
        
        # Discretize scores into bins
        bins = np.histogram(scores, bins=5)[0]
        bins = bins[bins > 0]  # Remove empty bins
        
        if len(bins) == 0:
            return 0.0
        
        # Compute probabilities
        total = sum(bins)
        probabilities = [count / total for count in bins]
        
        # Compute entropy
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        return entropy
    
    def _compute_agreement_weights(self, algorithm_predictions: Dict[str, List[RecommendationResult]]) -> Dict[str, float]:
        """Compute weights based on agreement with other algorithms"""
        
        algorithms = list(algorithm_predictions.keys())
        weights = {algo: 0.0 for algo in algorithms}
        
        if len(algorithms) < 2:
            return {algo: 1.0 for algo in algorithms}
        
        # Compute pairwise agreements
        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms[i+1:], i+1):
                agreement = self._compute_prediction_agreement(
                    algorithm_predictions[algo1],
                    algorithm_predictions[algo2]
                )
                
                weights[algo1] += agreement
                weights[algo2] += agreement
        
        return weights
    
    def _compute_prediction_agreement(self, predictions1: List[RecommendationResult], 
                                    predictions2: List[RecommendationResult]) -> float:
        """Compute agreement between two sets of predictions"""
        
        if not predictions1 or not predictions2:
            return 0.0
        
        # Create score mappings
        scores1 = {pred.item_id: pred.score for pred in predictions1}
        scores2 = {pred.item_id: pred.score for pred in predictions2}
        
        # Find common items
        common_items = set(scores1.keys()).intersection(set(scores2.keys()))
        
        if not common_items:
            return 0.0
        
        # Compute correlation
        common_scores1 = [scores1[item] for item in common_items]
        common_scores2 = [scores2[item] for item in common_items]
        
        correlation = np.corrcoef(common_scores1, common_scores2)[0, 1]
        
        # Handle NaN correlation (all scores are the same)
        if np.isnan(correlation):
            return 0.5
        
        # Convert correlation to agreement (0 to 1)
        agreement = (correlation + 1) / 2
        
        return agreement
    
    def update_performance_metrics(self, algorithm_name: str, 
                                  predictions: List[RecommendationResult],
                                  actual_ratings: Dict[str, float]):
        """Update performance metrics for algorithm"""
        
        # Compute accuracy metrics
        correct_predictions = 0
        total_predictions = 0
        mae_sum = 0.0
        
        for pred in predictions:
            if pred.item_id in actual_ratings:
                actual_rating = actual_ratings[pred.item_id]
                predicted_rating = pred.score
                
                # Binary accuracy (within 0.5 of actual)
                if abs(predicted_rating - actual_rating) <= 0.5:
                    correct_predictions += 1
                
                # MAE
                mae_sum += abs(predicted_rating - actual_rating)
                total_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            mae = mae_sum / total_predictions
            
            # Store metrics
            self.performance_tracker[algorithm_name]['recent_accuracy'] = accuracy
            self.performance_tracker[algorithm_name]['recent_mae'] = mae
            
            # Update history
            self.confidence_history[algorithm_name].append({
                'accuracy': accuracy,
                'mae': mae,
                'timestamp': pd.Timestamp.now()
            })
            
            # Keep only recent history (last 100 entries)
            if len(self.confidence_history[algorithm_name]) > 100:
                self.confidence_history[algorithm_name].pop(0)

class AdaptiveConfidenceModel:
    """
    Adaptive confidence model that learns from feedback
    """
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.confidence_features = {}
        self.confidence_model = None
        self.feedback_history = []
        
    def extract_confidence_features(self, user_id: str, item_id: str, 
                                   prediction_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for confidence prediction"""
        
        features = {}
        
        # User features
        features['user_activity'] = prediction_context.get('user_rating_count', 0)
        features['user_avg_rating'] = prediction_context.get('user_avg_rating', 3.0)
        features['user_rating_std'] = prediction_context.get('user_rating_std', 1.0)
        
        # Item features
        features['item_popularity'] = prediction_context.get('item_rating_count', 0)
        features['item_avg_rating'] = prediction_context.get('item_avg_rating', 3.0)
        features['item_rating_std'] = prediction_context.get('item_rating_std', 1.0)
        features['item_age_days'] = prediction_context.get('item_age_days', 365)
        
        # Algorithm-specific features
        features['prediction_score'] = prediction_context.get('prediction_score', 3.0)
        features['prediction_rank'] = prediction_context.get('prediction_rank', 50)
        
        # Interaction features
        features['user_item_similarity'] = prediction_context.get('content_similarity', 0.5)
        features['collaborative_strength'] = prediction_context.get('collaborative_signal', 0.5)
        
        return features
    
    def predict_confidence(self, user_id: str, item_id: str, 
                          prediction_context: Dict[str, Any]) -> float:
        """Predict confidence for a prediction"""
        
        if self.confidence_model is None:
            # Use heuristic confidence
            return self._heuristic_confidence(user_id, item_id, prediction_context)
        
        # Use trained model
        features = self.extract_confidence_features(user_id, item_id, prediction_context)
        feature_vector = [features.get(key, 0.0) for key in sorted(features.keys())]
        
        try:
            confidence = self.confidence_model.predict([feature_vector])[0]
            return max(0.0, min(1.0, confidence))
        except Exception:
            return self._heuristic_confidence(user_id, item_id, prediction_context)
    
    def _heuristic_confidence(self, user_id: str, item_id: str, 
                             prediction_context: Dict[str, Any]) -> float:
        """Heuristic confidence estimation"""
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on user activity
        user_activity = prediction_context.get('user_rating_count', 0)
        if user_activity > 50:
            confidence += 0.2
        elif user_activity < 5:
            confidence -= 0.2
        
        # Adjust based on item popularity
        item_popularity = prediction_context.get('item_rating_count', 0)
        if item_popularity > 100:
            confidence += 0.1
        elif item_popularity < 10:
            confidence -= 0.1
        
        # Adjust based on prediction score extremes
        prediction_score = prediction_context.get('prediction_score', 3.0)
        if prediction_score > 4.5 or prediction_score < 1.5:
            confidence += 0.1  # More confident about extreme predictions
        
        return max(0.1, min(0.9, confidence))
    
    def add_feedback(self, user_id: str, item_id: str, 
                    prediction_context: Dict[str, Any],
                    actual_outcome: float, prediction_error: float):
        """Add feedback to improve confidence model"""
        
        features = self.extract_confidence_features(user_id, item_id, prediction_context)
        
        # Confidence target: inverse of prediction error (normalized)
        max_error = 4.0  # Maximum possible error for 5-point scale
        confidence_target = 1.0 - (prediction_error / max_error)
        confidence_target = max(0.0, min(1.0, confidence_target))
        
        feedback_record = {
            'features': features,
            'confidence_target': confidence_target,
            'prediction_error': prediction_error,
            'timestamp': pd.Timestamp.now()
        }
        
        self.feedback_history.append(feedback_record)
        
        # Retrain model if enough feedback
        if len(self.feedback_history) >= 100 and len(self.feedback_history) % 50 == 0:
            self._retrain_confidence_model()
    
    def _retrain_confidence_model(self):
        """Retrain confidence model with accumulated feedback"""
        
        if len(self.feedback_history) < 20:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for record in self.feedback_history[-200:]:  # Use recent feedback
            features = record['features']
            feature_vector = [features.get(key, 0.0) for key in sorted(features.keys())]
            
            X.append(feature_vector)
            y.append(record['confidence_target'])
        
        # Train model
        from sklearn.ensemble import RandomForestRegressor
        
        self.confidence_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        self.confidence_model.fit(X, y)
        
        print(f"Retrained confidence model for {self.algorithm_name} with {len(X)} samples")
```

## 4. Study Questions

### Beginner Level

1. What are the benefits of using multi-layer hybrid architectures?
2. Why is score normalization important in hybrid recommendation systems?
3. How does confidence-based weighting improve recommendation quality?
4. What is the difference between static and dynamic weighting strategies?
5. How can entropy be used to assess prediction diversity?

### Intermediate Level

6. Implement a multi-layer hybrid system with learned combination layers for movie recommendations.
7. Compare different score normalization techniques and their impact on recommendation accuracy.
8. How would you design a confidence estimation model that adapts based on user feedback?
9. What are the computational trade-offs between different combination architectures?
10. How would you handle missing predictions from some algorithms in a multi-layer system?

### Advanced Level

11. Design a sophisticated confidence-based weighting system that considers prediction uncertainty, algorithm reliability, and user context.
12. Implement a multi-layer architecture that can automatically discover optimal layer configurations.
13. How would you create a score normalization technique that preserves the relative ranking while ensuring compatibility across algorithms?
14. Design a dynamic architecture that can add or remove layers based on performance feedback.
15. Implement a combination strategy that can handle algorithms with different output formats and confidence measures.

### Tricky Questions

16. How would you design a multi-layer system that can detect and adapt to concept drift in individual algorithms?
17. What are the challenges in maintaining explainability in complex multi-layer hybrid architectures?
18. How would you implement a confidence-based system that can distinguish between aleatoric and epistemic uncertainty?
19. Design a combination architecture that can work effectively with both high-latency batch algorithms and low-latency online algorithms.
20. How would you create a hybrid system that can automatically balance between accuracy and diversity across multiple layers?

## Key Takeaways

1. **Multi-layer architectures** enable sophisticated combination strategies beyond simple weighted averages
2. **Score normalization** is crucial for fair combination of algorithms with different output ranges
3. **Confidence-based weighting** can dynamically adapt to algorithm reliability and prediction uncertainty
4. **Advanced combination methods** like learned fusion can outperform simple weighted approaches
5. **Performance monitoring** and adaptation are essential for maintaining hybrid system effectiveness
6. **Feature engineering** for confidence estimation requires domain knowledge and algorithm understanding
7. **Architectural flexibility** allows systems to evolve and improve over time

## Next Session Preview

In Day 5.3, we'll explore **Weighted and Switching Hybrid Methods**, covering:
- Advanced weighted combination strategies
- Context-aware switching mechanisms
- Adaptive weight learning algorithms
- Performance-based algorithm selection
- Real-time switching optimization