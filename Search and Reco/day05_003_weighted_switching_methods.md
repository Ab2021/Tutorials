# Day 5.3: Weighted and Switching Hybrid Methods

## Learning Objectives
By the end of this session, you will:
- Master advanced weighted combination strategies with adaptive learning
- Implement context-aware switching mechanisms for hybrid systems
- Apply performance-based algorithm selection techniques
- Design real-time switching optimization algorithms
- Build sophisticated weight learning systems from user feedback

## 1. Advanced Weighted Combination Strategies

### Adaptive Weight Learning Systems

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class WeightUpdateEvent:
    """Event for weight updates"""
    timestamp: float
    user_id: str
    algorithm_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    context: Dict[str, Any]

class AdaptiveWeightLearner:
    """
    Advanced weight learning system that adapts based on performance feedback
    """
    
    def __init__(self, algorithms: List[str], learning_rate: float = 0.01):
        self.algorithms = algorithms
        self.learning_rate = learning_rate
        
        # Weight storage
        self.global_weights = {algo: 1.0 / len(algorithms) for algo in algorithms}
        self.user_specific_weights = {}  # user_id -> {algorithm: weight}
        self.context_specific_weights = {}  # context_key -> {algorithm: weight}
        
        # Performance tracking
        self.algorithm_performance = defaultdict(list)  # algorithm -> [performance_scores]
        self.weight_update_history = deque(maxlen=1000)
        
        # Learning parameters
        self.update_strategies = {
            'gradient_descent': self._gradient_descent_update,
            'exponential_moving_average': self._ema_update,
            'multi_armed_bandit': self._bandit_update,
            'bayesian_optimization': self._bayesian_update
        }
        
        self.current_strategy = 'gradient_descent'
        
        # Bandit parameters
        self.bandit_counts = defaultdict(int)
        self.bandit_rewards = defaultdict(list)
        self.exploration_rate = 0.1
        
    def get_weights(self, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Get appropriate weights based on user and context"""
        
        # Start with global weights
        weights = self.global_weights.copy()
        
        # Apply user-specific adjustments
        if user_id and user_id in self.user_specific_weights:
            user_weights = self.user_specific_weights[user_id]
            for algo in weights:
                if algo in user_weights:
                    weights[algo] = 0.7 * weights[algo] + 0.3 * user_weights[algo]
        
        # Apply context-specific adjustments
        if context:
            context_key = self._generate_context_key(context)
            if context_key in self.context_specific_weights:
                context_weights = self.context_specific_weights[context_key]
                for algo in weights:
                    if algo in context_weights:
                        weights[algo] = 0.8 * weights[algo] + 0.2 * context_weights[algo]
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {algo: weight / total_weight for algo, weight in weights.items()}
        
        return weights
    
    def update_weights(self, performance_feedback: Dict[str, float], 
                      user_id: str = None, context: Dict[str, Any] = None,
                      strategy: str = None):
        """Update weights based on performance feedback"""
        
        if strategy is None:
            strategy = self.current_strategy
        
        if strategy not in self.update_strategies:
            raise ValueError(f"Unknown update strategy: {strategy}")
        
        # Update global weights
        self.global_weights = self.update_strategies[strategy](
            self.global_weights, performance_feedback
        )
        
        # Update user-specific weights
        if user_id:
            if user_id not in self.user_specific_weights:
                self.user_specific_weights[user_id] = self.global_weights.copy()
            
            self.user_specific_weights[user_id] = self.update_strategies[strategy](
                self.user_specific_weights[user_id], performance_feedback
            )
        
        # Update context-specific weights
        if context:
            context_key = self._generate_context_key(context)
            if context_key not in self.context_specific_weights:
                self.context_specific_weights[context_key] = self.global_weights.copy()
            
            self.context_specific_weights[context_key] = self.update_strategies[strategy](
                self.context_specific_weights[context_key], performance_feedback
            )
        
        # Record update event
        update_event = WeightUpdateEvent(
            timestamp=time.time(),
            user_id=user_id or "global",
            algorithm_weights=self.global_weights.copy(),
            performance_metrics=performance_feedback.copy(),
            context=context or {}
        )
        self.weight_update_history.append(update_event)
        
        # Update algorithm performance history
        for algo, performance in performance_feedback.items():
            self.algorithm_performance[algo].append(performance)
            # Keep only recent performance (last 100 scores)
            if len(self.algorithm_performance[algo]) > 100:
                self.algorithm_performance[algo].pop(0)
    
    def _gradient_descent_update(self, current_weights: Dict[str, float], 
                                performance_feedback: Dict[str, float]) -> Dict[str, float]:
        """Gradient descent weight update"""
        
        new_weights = current_weights.copy()
        
        # Compute gradients based on performance differences
        avg_performance = sum(performance_feedback.values()) / len(performance_feedback)
        
        for algo in new_weights:
            if algo in performance_feedback:
                # Gradient: (performance - average_performance)
                gradient = performance_feedback[algo] - avg_performance
                new_weights[algo] += self.learning_rate * gradient
                new_weights[algo] = max(0.01, new_weights[algo])  # Prevent negative weights
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {algo: weight / total_weight for algo, weight in new_weights.items()}
        
        return new_weights
    
    def _ema_update(self, current_weights: Dict[str, float], 
                   performance_feedback: Dict[str, float]) -> Dict[str, float]:
        """Exponential moving average weight update"""
        
        new_weights = current_weights.copy()
        alpha = self.learning_rate  # EMA decay factor
        
        # Convert performance to normalized weights
        if performance_feedback:
            total_performance = sum(performance_feedback.values())
            if total_performance > 0:
                performance_weights = {
                    algo: perf / total_performance 
                    for algo, perf in performance_feedback.items()
                }
                
                # EMA update: new_weight = alpha * performance_weight + (1-alpha) * old_weight
                for algo in new_weights:
                    if algo in performance_weights:
                        new_weights[algo] = (alpha * performance_weights[algo] + 
                                           (1 - alpha) * current_weights[algo])
        
        return new_weights
    
    def _bandit_update(self, current_weights: Dict[str, float], 
                      performance_feedback: Dict[str, float]) -> Dict[str, float]:
        """Multi-armed bandit weight update using Upper Confidence Bound"""
        
        # Update bandit statistics
        total_plays = sum(self.bandit_counts.values())
        
        for algo, performance in performance_feedback.items():
            self.bandit_counts[algo] += 1
            self.bandit_rewards[algo].append(performance)
        
        # Compute UCB scores
        ucb_scores = {}
        for algo in self.algorithms:
            if self.bandit_counts[algo] == 0:
                ucb_scores[algo] = float('inf')  # Explore unplayed arms
            else:
                avg_reward = np.mean(self.bandit_rewards[algo])
                confidence_interval = np.sqrt(2 * np.log(total_plays + 1) / self.bandit_counts[algo])
                ucb_scores[algo] = avg_reward + confidence_interval
        
        # Convert UCB scores to weights
        if ucb_scores:
            # Apply softmax to UCB scores
            max_score = max(ucb_scores.values())
            exp_scores = {algo: np.exp(score - max_score) for algo, score in ucb_scores.items()}
            total_exp = sum(exp_scores.values())
            
            new_weights = {algo: exp_score / total_exp for algo, exp_score in exp_scores.items()}
        else:
            new_weights = current_weights.copy()
        
        return new_weights
    
    def _bayesian_update(self, current_weights: Dict[str, float], 
                        performance_feedback: Dict[str, float]) -> Dict[str, float]:
        """Bayesian weight update (simplified)"""
        
        # Simple Bayesian update using Beta distribution parameters
        new_weights = current_weights.copy()
        
        # Update based on performance (treating as success/failure)
        threshold = 0.5  # Performance threshold
        
        for algo, performance in performance_feedback.items():
            # Convert performance to success probability
            success_prob = max(0.01, min(0.99, performance))
            
            # Simple update: increase weight for successful algorithms
            if performance > threshold:
                new_weights[algo] *= (1 + self.learning_rate * success_prob)
            else:
                new_weights[algo] *= (1 - self.learning_rate * (1 - success_prob))
            
            new_weights[algo] = max(0.01, new_weights[algo])
        
        # Normalize
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {algo: weight / total_weight for algo, weight in new_weights.items()}
        
        return new_weights
    
    def _generate_context_key(self, context: Dict[str, Any]) -> str:
        """Generate key for context-specific weights"""
        # Create a hashable key from context
        sorted_items = sorted(context.items())
        return str(hash(tuple(sorted_items)))
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get statistics about weight learning"""
        
        stats = {
            'global_weights': self.global_weights.copy(),
            'num_users': len(self.user_specific_weights),
            'num_contexts': len(self.context_specific_weights),
            'total_updates': len(self.weight_update_history),
            'algorithm_performance': {},
            'weight_stability': {}
        }
        
        # Algorithm performance statistics
        for algo, performances in self.algorithm_performance.items():
            if performances:
                stats['algorithm_performance'][algo] = {
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'recent_trend': self._compute_trend(performances[-20:]) if len(performances) >= 20 else 0.0
                }
        
        # Weight stability (variance over recent updates)
        recent_updates = list(self.weight_update_history)[-50:]
        if len(recent_updates) > 10:
            for algo in self.algorithms:
                weights = [event.algorithm_weights.get(algo, 0.0) for event in recent_updates]
                stats['weight_stability'][algo] = np.std(weights)
        
        return stats
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend in values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope if not np.isnan(slope) else 0.0

class PerformanceBasedWeighting:
    """
    Performance-based weighting system with multiple evaluation metrics
    """
    
    def __init__(self, algorithms: List[str]):
        self.algorithms = algorithms
        self.performance_metrics = {
            'accuracy': self._accuracy_metric,
            'precision': self._precision_metric,
            'recall': self._recall_metric,
            'f1_score': self._f1_metric,
            'mae': self._mae_metric,
            'rmse': self._rmse_metric,
            'diversity': self._diversity_metric,
            'coverage': self._coverage_metric
        }
        
        # Performance history
        self.metric_history = defaultdict(lambda: defaultdict(list))  # metric -> algorithm -> [scores]
        self.composite_scores = defaultdict(list)  # algorithm -> [composite_scores]
        
        # Metric weights for composite scoring
        self.metric_weights = {
            'accuracy': 0.3,
            'precision': 0.2,
            'recall': 0.2,
            'diversity': 0.15,
            'coverage': 0.15
        }
    
    def evaluate_algorithms(self, algorithm_predictions: Dict[str, List], 
                           ground_truth: Dict[str, Any],
                           evaluation_context: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all algorithms on multiple metrics"""
        
        results = {}
        
        for algo_name, predictions in algorithm_predictions.items():
            algo_results = {}
            
            for metric_name, metric_func in self.performance_metrics.items():
                try:
                    score = metric_func(predictions, ground_truth, evaluation_context)
                    algo_results[metric_name] = score
                    
                    # Store in history
                    self.metric_history[metric_name][algo_name].append(score)
                    
                    # Keep only recent history
                    if len(self.metric_history[metric_name][algo_name]) > 100:
                        self.metric_history[metric_name][algo_name].pop(0)
                        
                except Exception as e:
                    print(f"Error computing {metric_name} for {algo_name}: {e}")
                    algo_results[metric_name] = 0.0
            
            results[algo_name] = algo_results
        
        # Compute composite scores
        self._update_composite_scores(results)
        
        return results
    
    def compute_performance_based_weights(self, weighting_strategy: str = 'composite') -> Dict[str, float]:
        """Compute weights based on performance metrics"""
        
        if weighting_strategy == 'composite':
            return self._composite_weighting()
        elif weighting_strategy == 'recent_performance':
            return self._recent_performance_weighting()
        elif weighting_strategy == 'stable_performance':
            return self._stable_performance_weighting()
        elif weighting_strategy == 'metric_specific':
            return self._metric_specific_weighting()
        else:
            # Uniform weights
            return {algo: 1.0 / len(self.algorithms) for algo in self.algorithms}
    
    def _composite_weighting(self) -> Dict[str, float]:
        """Weight based on composite performance scores"""
        
        weights = {}
        
        if not any(self.composite_scores.values()):
            return {algo: 1.0 / len(self.algorithms) for algo in self.algorithms}
        
        # Use recent composite scores
        recent_scores = {}
        for algo in self.algorithms:
            if self.composite_scores[algo]:
                recent_scores[algo] = np.mean(self.composite_scores[algo][-10:])  # Last 10 scores
            else:
                recent_scores[algo] = 0.5
        
        # Convert to weights using softmax
        max_score = max(recent_scores.values())
        exp_scores = {algo: np.exp(score - max_score) for algo, score in recent_scores.items()}
        total_exp = sum(exp_scores.values())
        
        weights = {algo: exp_score / total_exp for algo, exp_score in exp_scores.items()}
        
        return weights
    
    def _recent_performance_weighting(self) -> Dict[str, float]:
        """Weight based on recent performance trend"""
        
        weights = {}
        trends = {}
        
        for algo in self.algorithms:
            # Compute trend across all metrics
            algo_trends = []
            
            for metric_name in self.performance_metrics:
                if (metric_name in self.metric_history and 
                    algo in self.metric_history[metric_name] and
                    len(self.metric_history[metric_name][algo]) >= 10):
                    
                    recent_scores = self.metric_history[metric_name][algo][-10:]
                    trend = self._compute_trend(recent_scores)
                    algo_trends.append(trend * self.metric_weights.get(metric_name, 1.0))
            
            trends[algo] = np.mean(algo_trends) if algo_trends else 0.0
        
        # Convert trends to weights (positive trend = higher weight)
        min_trend = min(trends.values())
        shifted_trends = {algo: trend - min_trend + 0.1 for algo, trend in trends.items()}
        
        total_trend = sum(shifted_trends.values())
        weights = {algo: trend / total_trend for algo, trend in shifted_trends.items()}
        
        return weights
    
    def _stable_performance_weighting(self) -> Dict[str, float]:
        """Weight based on performance stability (low variance)"""
        
        weights = {}
        stability_scores = {}
        
        for algo in self.algorithms:
            stabilities = []
            
            for metric_name in self.performance_metrics:
                if (metric_name in self.metric_history and 
                    algo in self.metric_history[metric_name] and
                    len(self.metric_history[metric_name][algo]) >= 5):
                    
                    scores = self.metric_history[metric_name][algo]
                    # Stability = 1 / (1 + std)
                    stability = 1.0 / (1.0 + np.std(scores))
                    stabilities.append(stability * self.metric_weights.get(metric_name, 1.0))
            
            stability_scores[algo] = np.mean(stabilities) if stabilities else 0.5
        
        # Normalize to weights
        total_stability = sum(stability_scores.values())
        weights = {algo: stability / total_stability for algo, stability in stability_scores.items()}
        
        return weights
    
    def _metric_specific_weighting(self) -> Dict[str, float]:
        """Weight based on specific metric performance"""
        
        # Focus on most important metric
        primary_metric = 'accuracy'
        
        weights = {}
        metric_scores = {}
        
        for algo in self.algorithms:
            if (primary_metric in self.metric_history and 
                algo in self.metric_history[primary_metric] and
                self.metric_history[primary_metric][algo]):
                
                recent_scores = self.metric_history[primary_metric][algo][-5:]
                metric_scores[algo] = np.mean(recent_scores)
            else:
                metric_scores[algo] = 0.5
        
        # Convert to weights
        total_score = sum(metric_scores.values())
        weights = {algo: score / total_score for algo, score in metric_scores.items()}
        
        return weights
    
    def _update_composite_scores(self, evaluation_results: Dict[str, Dict[str, float]]):
        """Update composite performance scores"""
        
        for algo_name, metrics in evaluation_results.items():
            composite_score = 0.0
            total_weight = 0.0
            
            for metric_name, score in metrics.items():
                if metric_name in self.metric_weights:
                    weight = self.metric_weights[metric_name]
                    composite_score += weight * score
                    total_weight += weight
            
            if total_weight > 0:
                composite_score /= total_weight
            
            self.composite_scores[algo_name].append(composite_score)
            
            # Keep only recent composite scores
            if len(self.composite_scores[algo_name]) > 50:
                self.composite_scores[algo_name].pop(0)
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend in values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        correlation = np.corrcoef(x, y)[0, 1]
        if np.isnan(correlation):
            return 0.0
        
        # Convert correlation to trend strength
        return correlation
    
    # Metric computation methods
    def _accuracy_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                        context: Dict[str, Any] = None) -> float:
        """Compute accuracy metric"""
        # Simplified accuracy computation
        # In practice, would depend on prediction format
        return np.random.uniform(0.5, 0.9)  # Mock implementation
    
    def _precision_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> float:
        """Compute precision metric"""
        return np.random.uniform(0.4, 0.8)  # Mock implementation
    
    def _recall_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                      context: Dict[str, Any] = None) -> float:
        """Compute recall metric"""
        return np.random.uniform(0.5, 0.85)  # Mock implementation
    
    def _f1_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                   context: Dict[str, Any] = None) -> float:
        """Compute F1 score"""
        # Would compute based on precision and recall
        return np.random.uniform(0.45, 0.82)  # Mock implementation
    
    def _mae_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                    context: Dict[str, Any] = None) -> float:
        """Compute Mean Absolute Error (lower is better, so return 1-normalized_mae)"""
        mae = np.random.uniform(0.5, 2.0)  # Mock MAE
        normalized_mae = mae / 5.0  # Assuming 5-point scale
        return 1.0 - normalized_mae  # Convert to "higher is better"
    
    def _rmse_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> float:
        """Compute Root Mean Square Error (lower is better, so return 1-normalized_rmse)"""
        rmse = np.random.uniform(0.6, 2.5)  # Mock RMSE
        normalized_rmse = rmse / 5.0  # Assuming 5-point scale
        return 1.0 - normalized_rmse  # Convert to "higher is better"
    
    def _diversity_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> float:
        """Compute diversity metric"""
        return np.random.uniform(0.3, 0.7)  # Mock implementation
    
    def _coverage_metric(self, predictions: List, ground_truth: Dict[str, Any], 
                        context: Dict[str, Any] = None) -> float:
        """Compute coverage metric"""
        return np.random.uniform(0.4, 0.8)  # Mock implementation
```

## 2. Context-Aware Switching Mechanisms

### Advanced Switching Strategies

```python
class ContextAwareSwitchingSystem:
    """
    Advanced switching system that selects algorithms based on context
    """
    
    def __init__(self, algorithms: List[str]):
        self.algorithms = algorithms
        self.switching_rules = []
        self.learned_switching_model = None
        self.context_performance_history = defaultdict(lambda: defaultdict(list))
        
        # Switching strategies
        self.switching_strategies = {
            'rule_based': self._rule_based_switching,
            'performance_based': self._performance_based_switching,
            'learned': self._learned_switching,
            'ensemble_voting': self._ensemble_voting_switching,
            'context_similarity': self._context_similarity_switching
        }
        
        # Context feature extractors
        self.context_extractors = {
            'user_features': self._extract_user_features,
            'item_features': self._extract_item_features,
            'temporal_features': self._extract_temporal_features,
            'interaction_features': self._extract_interaction_features
        }
        
    def add_switching_rule(self, rule_name: str, condition_func: Callable, 
                          algorithm_name: str, priority: int = 0):
        """Add rule-based switching condition"""
        
        rule = {
            'name': rule_name,
            'condition': condition_func,
            'algorithm': algorithm_name,
            'priority': priority
        }
        
        self.switching_rules.append(rule)
        # Sort by priority (higher priority first)
        self.switching_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def select_algorithm(self, user_id: str, context: Dict[str, Any], 
                        strategy: str = 'learned') -> str:
        """Select best algorithm for given context"""
        
        if strategy not in self.switching_strategies:
            raise ValueError(f"Unknown switching strategy: {strategy}")
        
        return self.switching_strategies[strategy](user_id, context)
    
    def _rule_based_switching(self, user_id: str, context: Dict[str, Any]) -> str:
        """Rule-based algorithm selection"""
        
        for rule in self.switching_rules:
            try:
                if rule['condition'](user_id, context):
                    return rule['algorithm']
            except Exception as e:
                print(f"Error evaluating rule {rule['name']}: {e}")
                continue
        
        # Default to first algorithm if no rule matches
        return self.algorithms[0] if self.algorithms else None
    
    def _performance_based_switching(self, user_id: str, context: Dict[str, Any]) -> str:
        """Performance-based algorithm selection"""
        
        # Extract context features for similarity matching
        context_key = self._generate_context_key(context)
        
        # Find similar contexts and their best performing algorithms
        best_algorithm = None
        best_performance = -1.0
        
        for stored_context, algo_performances in self.context_performance_history.items():
            # Simple context similarity (in practice, would use more sophisticated matching)
            similarity = self._compute_context_similarity(context_key, stored_context)
            
            if similarity > 0.7:  # Similarity threshold
                for algo, performances in algo_performances.items():
                    if performances:
                        avg_performance = np.mean(performances[-5:])  # Recent performance
                        weighted_performance = avg_performance * similarity
                        
                        if weighted_performance > best_performance:
                            best_performance = weighted_performance
                            best_algorithm = algo
        
        return best_algorithm if best_algorithm else self.algorithms[0]
    
    def _learned_switching(self, user_id: str, context: Dict[str, Any]) -> str:
        """Learned switching using ML model"""
        
        if self.learned_switching_model is None:
            # Fallback to performance-based switching
            return self._performance_based_switching(user_id, context)
        
        # Extract features for ML model
        features = self._extract_context_features(user_id, context)
        
        try:
            # Predict best algorithm
            algorithm_probabilities = self.learned_switching_model.predict_proba([features])[0]
            best_algo_index = np.argmax(algorithm_probabilities)
            return self.algorithms[best_algo_index]
        except Exception as e:
            print(f"Error in learned switching: {e}")
            return self._performance_based_switching(user_id, context)
    
    def _ensemble_voting_switching(self, user_id: str, context: Dict[str, Any]) -> str:
        """Ensemble voting across multiple switching strategies"""
        
        votes = defaultdict(int)
        
        # Get votes from different strategies
        strategies = ['rule_based', 'performance_based']
        
        for strategy in strategies:
            try:
                selected_algo = self.switching_strategies[strategy](user_id, context)
                if selected_algo:
                    votes[selected_algo] += 1
            except Exception:
                continue
        
        if votes:
            # Return algorithm with most votes
            return max(votes.items(), key=lambda x: x[1])[0]
        else:
            return self.algorithms[0]
    
    def _context_similarity_switching(self, user_id: str, context: Dict[str, Any]) -> str:
        """Switch based on context similarity to historical best contexts"""
        
        context_features = self._extract_context_features(user_id, context)
        
        # Find most similar historical context
        most_similar_context = None
        highest_similarity = -1.0
        
        for stored_context in self.context_performance_history.keys():
            try:
                # Reconstruct features from stored context (simplified)
                stored_features = self._context_key_to_features(stored_context)
                similarity = self._compute_feature_similarity(context_features, stored_features)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_context = stored_context
            except Exception:
                continue
        
        if most_similar_context and highest_similarity > 0.6:
            # Find best algorithm for most similar context
            algo_performances = self.context_performance_history[most_similar_context]
            best_algo = max(algo_performances.items(), 
                          key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
            return best_algo
        
        return self.algorithms[0]
    
    def train_switching_model(self, training_data: List[Dict[str, Any]]):
        """Train ML model for learned switching"""
        
        if len(training_data) < 50:
            print("Not enough training data for switching model")
            return
        
        # Prepare training data
        X = []  # Features
        y = []  # Best algorithm labels
        
        for record in training_data:
            features = self._extract_context_features(
                record['user_id'], 
                record['context']
            )
            best_algorithm = record['best_algorithm']
            
            X.append(features)
            
            # Convert algorithm name to index
            if best_algorithm in self.algorithms:
                y.append(self.algorithms.index(best_algorithm))
            else:
                y.append(0)  # Default to first algorithm
        
        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        
        self.learned_switching_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.learned_switching_model.fit(X, y)
        
        print(f"Trained switching model with {len(X)} samples")
    
    def update_performance_history(self, user_id: str, context: Dict[str, Any],
                                  algorithm_performance: Dict[str, float]):
        """Update performance history for context-aware switching"""
        
        context_key = self._generate_context_key(context)
        
        for algorithm, performance in algorithm_performance.items():
            self.context_performance_history[context_key][algorithm].append(performance)
            
            # Keep only recent history
            if len(self.context_performance_history[context_key][algorithm]) > 20:
                self.context_performance_history[context_key][algorithm].pop(0)
    
    def _extract_context_features(self, user_id: str, context: Dict[str, Any]) -> List[float]:
        """Extract numerical features from context"""
        
        features = []
        
        # Extract features using registered extractors
        for extractor_name, extractor_func in self.context_extractors.items():
            try:
                extracted_features = extractor_func(user_id, context)
                features.extend(extracted_features)
            except Exception as e:
                print(f"Error in {extractor_name}: {e}")
                # Add default features
                features.extend([0.0] * 5)  # Default feature count
        
        return features
    
    def _extract_user_features(self, user_id: str, context: Dict[str, Any]) -> List[float]:
        """Extract user-specific features"""
        
        # Mock user features - in practice, would query user database
        features = [
            hash(user_id) % 100 / 100.0,  # User ID hash (normalized)
            context.get('user_age', 30) / 100.0,  # Age (normalized)
            context.get('user_rating_count', 50) / 1000.0,  # Activity level
            context.get('user_avg_rating', 3.5) / 5.0,  # Rating tendency
            len(context.get('user_favorite_genres', [])) / 10.0  # Genre diversity
        ]
        
        return features
    
    def _extract_item_features(self, user_id: str, context: Dict[str, Any]) -> List[float]:
        """Extract item-specific features"""
        
        features = [
            context.get('item_popularity', 50) / 1000.0,  # Item popularity
            context.get('item_age_days', 365) / 3650.0,  # Item age (normalized to 10 years)
            context.get('item_avg_rating', 3.5) / 5.0,  # Item quality
            context.get('item_rating_count', 100) / 10000.0,  # Item rating volume
            len(context.get('item_genres', [])) / 10.0  # Genre count
        ]
        
        return features
    
    def _extract_temporal_features(self, user_id: str, context: Dict[str, Any]) -> List[float]:
        """Extract temporal features"""
        
        import datetime
        
        now = datetime.datetime.now()
        
        features = [
            now.hour / 24.0,  # Hour of day
            now.weekday() / 7.0,  # Day of week
            now.month / 12.0,  # Month of year
            int(context.get('is_weekend', False)),  # Weekend flag
            int(context.get('is_holiday', False))  # Holiday flag
        ]
        
        return features
    
    def _extract_interaction_features(self, user_id: str, context: Dict[str, Any]) -> List[float]:
        """Extract interaction context features"""
        
        features = [
            int(context.get('is_mobile', False)),  # Mobile device
            context.get('session_length', 30) / 120.0,  # Session length (normalized to 2 hours)
            context.get('previous_interactions', 5) / 50.0,  # Recent interactions
            len(context.get('current_mood', [])) / 5.0,  # Mood indicators
            context.get('social_context', 0) / 3.0  # Social context (alone=0, friends=1, family=2, etc.)
        ]
        
        return features
    
    def _generate_context_key(self, context: Dict[str, Any]) -> str:
        """Generate string key for context"""
        # Simplified context key generation
        key_parts = []
        
        # Include important context dimensions
        important_keys = ['user_age_group', 'device_type', 'time_of_day', 'session_type']
        
        for key in important_keys:
            if key in context:
                key_parts.append(f"{key}:{context[key]}")
        
        return "|".join(key_parts) if key_parts else "default"
    
    def _compute_context_similarity(self, context1: str, context2: str) -> float:
        """Compute similarity between context keys"""
        
        # Simple Jaccard similarity
        parts1 = set(context1.split("|"))
        parts2 = set(context2.split("|"))
        
        if not parts1 and not parts2:
            return 1.0
        
        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_feature_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Compute similarity between feature vectors"""
        
        if len(features1) != len(features2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        norm1 = sum(a * a for a in features1) ** 0.5
        norm2 = sum(b * b for b in features2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _context_key_to_features(self, context_key: str) -> List[float]:
        """Convert context key back to feature vector (simplified)"""
        
        # This is a simplified conversion - in practice, would store actual features
        # For demo purposes, generate consistent features from context key
        hash_value = hash(context_key)
        
        # Generate consistent pseudo-features
        features = []
        for i in range(20):  # Match total feature count
            feature_hash = hash(f"{context_key}_{i}")
            feature_value = (feature_hash % 1000) / 1000.0
            features.append(feature_value)
        
        return features
    
    def get_switching_statistics(self) -> Dict[str, Any]:
        """Get statistics about switching behavior"""
        
        stats = {
            'num_switching_rules': len(self.switching_rules),
            'num_contexts_tracked': len(self.context_performance_history),
            'learned_model_trained': self.learned_switching_model is not None,
            'context_coverage': {},
            'algorithm_selection_frequency': defaultdict(int)
        }
        
        # Analyze context coverage
        for context_key, algo_perfs in self.context_performance_history.items():
            stats['context_coverage'][context_key] = {
                'algorithms_tested': len(algo_perfs),
                'total_evaluations': sum(len(perfs) for perfs in algo_perfs.values())
            }
        
        return stats
```

## 3. Real-Time Switching Optimization

### Online Learning for Switching

```python
class OnlineSwitchingOptimizer:
    """
    Online learning system for real-time switching optimization
    """
    
    def __init__(self, algorithms: List[str], window_size: int = 100):
        self.algorithms = algorithms
        self.window_size = window_size
        
        # Online learning components
        self.performance_window = deque(maxlen=window_size)
        self.algorithm_rewards = defaultdict(lambda: deque(maxlen=window_size))
        self.context_embeddings = {}
        
        # Bandit parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        
        # Thompson Sampling parameters
        self.alpha_params = defaultdict(lambda: 1.0)  # Success parameters
        self.beta_params = defaultdict(lambda: 1.0)   # Failure parameters
        
        # Contextual bandit model
        self.contextual_weights = defaultdict(lambda: np.zeros(20))  # Feature dimension
        self.learning_rate = 0.01
        
    def select_algorithm_online(self, user_id: str, context: Dict[str, Any], 
                               method: str = 'epsilon_greedy') -> str:
        """Select algorithm using online learning method"""
        
        if method == 'epsilon_greedy':
            return self._epsilon_greedy_selection(user_id, context)
        elif method == 'thompson_sampling':
            return self._thompson_sampling_selection(user_id, context)
        elif method == 'contextual_bandit':
            return self._contextual_bandit_selection(user_id, context)
        elif method == 'adaptive_epsilon':
            return self._adaptive_epsilon_selection(user_id, context)
        else:
            return self._epsilon_greedy_selection(user_id, context)
    
    def _epsilon_greedy_selection(self, user_id: str, context: Dict[str, Any]) -> str:
        """Epsilon-greedy algorithm selection"""
        
        if np.random.random() < self.epsilon:
            # Exploration: random selection
            return np.random.choice(self.algorithms)
        else:
            # Exploitation: select best performing algorithm
            best_algorithm = None
            best_avg_reward = -1.0
            
            for algorithm in self.algorithms:
                if self.algorithm_rewards[algorithm]:
                    avg_reward = np.mean(list(self.algorithm_rewards[algorithm]))
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_algorithm = algorithm
            
            return best_algorithm if best_algorithm else np.random.choice(self.algorithms)
    
    def _thompson_sampling_selection(self, user_id: str, context: Dict[str, Any]) -> str:
        """Thompson Sampling algorithm selection"""
        
        algorithm_samples = {}
        
        for algorithm in self.algorithms:
            # Sample from Beta distribution
            alpha = self.alpha_params[algorithm]
            beta = self.beta_params[algorithm]
            
            sampled_reward = np.random.beta(alpha, beta)
            algorithm_samples[algorithm] = sampled_reward
        
        # Select algorithm with highest sample
        best_algorithm = max(algorithm_samples.items(), key=lambda x: x[1])[0]
        return best_algorithm
    
    def _contextual_bandit_selection(self, user_id: str, context: Dict[str, Any]) -> str:
        """Contextual bandit algorithm selection"""
        
        # Extract context features
        context_features = self._extract_context_vector(user_id, context)
        
        # Compute expected rewards for each algorithm
        expected_rewards = {}
        
        for algorithm in self.algorithms:
            weights = self.contextual_weights[algorithm]
            expected_reward = np.dot(weights, context_features)
            expected_rewards[algorithm] = expected_reward
        
        # Add exploration bonus (Upper Confidence Bound)
        exploration_bonus = 0.1
        total_plays = sum(len(rewards) for rewards in self.algorithm_rewards.values())
        
        for algorithm in self.algorithms:
            plays = len(self.algorithm_rewards[algorithm])
            if plays > 0 and total_plays > 0:
                confidence_bonus = exploration_bonus * np.sqrt(np.log(total_plays) / plays)
                expected_rewards[algorithm] += confidence_bonus
        
        # Select algorithm with highest expected reward + confidence
        best_algorithm = max(expected_rewards.items(), key=lambda x: x[1])[0]
        return best_algorithm
    
    def _adaptive_epsilon_selection(self, user_id: str, context: Dict[str, Any]) -> str:
        """Adaptive epsilon-greedy with performance-based exploration"""
        
        # Adapt epsilon based on recent performance variance
        if len(self.performance_window) > 10:
            recent_performance = list(self.performance_window)[-10:]
            performance_std = np.std(recent_performance)
            
            # Higher variance = more exploration
            adaptive_epsilon = min(0.3, self.epsilon + 0.1 * performance_std)
        else:
            adaptive_epsilon = self.epsilon
        
        if np.random.random() < adaptive_epsilon:
            # Exploration with bias toward less-tested algorithms
            algorithm_play_counts = {algo: len(self.algorithm_rewards[algo]) 
                                   for algo in self.algorithms}
            min_plays = min(algorithm_play_counts.values())
            
            # Bias toward algorithms with minimum plays
            candidates = [algo for algo, plays in algorithm_play_counts.items() 
                         if plays <= min_plays + 2]
            
            return np.random.choice(candidates)
        else:
            # Exploitation
            return self._epsilon_greedy_selection(user_id, context)
    
    def update_algorithm_performance(self, algorithm: str, reward: float, 
                                   user_id: str, context: Dict[str, Any]):
        """Update algorithm performance with new reward"""
        
        # Update reward history
        self.algorithm_rewards[algorithm].append(reward)
        self.performance_window.append(reward)
        
        # Update Thompson Sampling parameters
        if reward > 0.5:  # Success threshold
            self.alpha_params[algorithm] += 1.0
        else:
            self.beta_params[algorithm] += 1.0
        
        # Update contextual bandit weights
        context_features = self._extract_context_vector(user_id, context)
        
        weights = self.contextual_weights[algorithm]
        prediction = np.dot(weights, context_features)
        error = reward - prediction
        
        # Gradient descent update
        gradient = error * context_features
        self.contextual_weights[algorithm] += self.learning_rate * gradient
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _extract_context_vector(self, user_id: str, context: Dict[str, Any]) -> np.ndarray:
        """Extract context features as vector"""
        
        # Create consistent feature vector from context
        features = []
        
        # User features
        features.append(hash(user_id) % 1000 / 1000.0)  # User hash
        features.append(context.get('user_age', 30) / 100.0)
        features.append(context.get('user_activity', 50) / 500.0)
        
        # Temporal features
        import datetime
        now = datetime.datetime.now()
        features.append(now.hour / 24.0)
        features.append(now.weekday() / 7.0)
        
        # Item features
        features.append(context.get('item_popularity', 100) / 1000.0)
        features.append(context.get('item_age', 365) / 3650.0)
        
        # Session features
        features.append(int(context.get('is_mobile', False)))
        features.append(context.get('session_length', 30) / 300.0)
        
        # Pad or truncate to fixed size
        target_size = 20
        while len(features) < target_size:
            features.append(0.0)
        features = features[:target_size]
        
        return np.array(features)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get online optimization statistics"""
        
        stats = {
            'current_epsilon': self.epsilon,
            'total_interactions': len(self.performance_window),
            'algorithm_play_counts': {algo: len(rewards) 
                                    for algo, rewards in self.algorithm_rewards.items()},
            'algorithm_avg_rewards': {algo: np.mean(list(rewards)) if rewards else 0.0
                                    for algo, rewards in self.algorithm_rewards.items()},
            'recent_performance_trend': 0.0,
            'exploration_vs_exploitation_ratio': 0.0
        }
        
        # Compute recent performance trend
        if len(self.performance_window) >= 20:
            recent_half = list(self.performance_window)[-10:]
            earlier_half = list(self.performance_window)[-20:-10]
            
            recent_avg = np.mean(recent_half)
            earlier_avg = np.mean(earlier_half)
            
            stats['recent_performance_trend'] = recent_avg - earlier_avg
        
        # Estimate exploration ratio (simplified)
        if len(self.performance_window) > 0:
            stats['exploration_vs_exploitation_ratio'] = self.epsilon
        
        return stats
    
    def reset_learning(self):
        """Reset learning state for new context"""
        
        self.performance_window.clear()
        self.algorithm_rewards.clear()
        self.alpha_params.clear()
        self.beta_params.clear()
        self.contextual_weights.clear()
        self.epsilon = 0.1
        
        print("Online learning state reset")
```

## 4. Study Questions

### Beginner Level

1. What are the key differences between weighted and switching hybrid approaches?
2. How does adaptive weight learning improve upon static weight assignment?
3. What is the role of exploration vs exploitation in online algorithm selection?
4. How can context information be used to improve switching decisions?
5. What are the advantages of performance-based weighting strategies?

### Intermediate Level

6. Implement an adaptive weight learning system that uses multi-armed bandit techniques for a music recommendation scenario.
7. How would you design a context-aware switching system that considers user mood, device type, and time of day?
8. Compare different online learning algorithms (epsilon-greedy, Thompson sampling, contextual bandits) for algorithm selection.
9. What are the computational challenges of real-time switching optimization in production systems?
10. How would you handle cold start problems in contextual switching systems?

### Advanced Level

11. Design a sophisticated switching system that can learn hierarchical context patterns and make switching decisions at multiple levels.
12. Implement a meta-learning approach that can quickly adapt switching strategies to new domains or user populations.
13. How would you create a switching system that can handle non-stationary environments where algorithm performance changes over time?
14. Design a system that can automatically discover relevant context features for switching decisions.
15. Implement a distributed switching system that can coordinate algorithm selection across multiple recommendation services.

### Tricky Questions

16. How would you design a switching system that can balance individual user optimization with global system performance?
17. What are the privacy implications of context-aware switching, and how would you address them?
18. How would you handle the situation where switching decisions become predictable and users start gaming the system?
19. Design a switching system that can work effectively with both batch and real-time algorithms while maintaining consistent user experience.
20. How would you implement fair switching that ensures all algorithms get adequate evaluation opportunities without compromising performance?

## Key Takeaways

1. **Adaptive weight learning** enables systems to automatically optimize combination strategies based on performance feedback
2. **Context-aware switching** can significantly improve recommendation quality by selecting appropriate algorithms for different scenarios
3. **Online learning methods** allow real-time optimization of switching decisions with exploration-exploitation balance
4. **Performance-based weighting** provides objective criteria for algorithm combination and selection
5. **Multi-armed bandit approaches** are effective for algorithm selection under uncertainty
6. **Contextual features** are crucial for intelligent switching but require careful engineering and privacy consideration
7. **Real-time optimization** requires efficient algorithms and careful balance between accuracy and computational cost

## Next Session Preview

In Day 5.4, we'll explore **Meta-Learning for Recommendation Fusion**, covering:
- Meta-learning architectures for recommendation systems
- Learning to combine algorithms automatically
- Transfer learning across recommendation domains
- Neural approaches to hybrid recommendation
- Automated hyperparameter optimization for hybrid systems