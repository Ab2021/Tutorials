# Day 5.5: Dynamic Hybridization Strategies

## Learning Objectives
By the end of this session, you will:
- Master real-time adaptation techniques for hybrid recommendation systems
- Implement context-driven dynamic fusion algorithms
- Apply online learning methods for continuous hybrid optimization
- Handle temporal dynamics and concept drift in recommendation fusion
- Build adaptive ensemble methods that evolve with user behavior

## 1. Real-Time Adaptive Hybridization

### Dynamic Strategy Selection System

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from enum import Enum
import json

class AdaptationTrigger(Enum):
    """Types of adaptation triggers"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONTEXT_CHANGE = "context_change"
    USER_FEEDBACK = "user_feedback"
    TEMPORAL_DRIFT = "temporal_drift"
    LOAD_BALANCING = "load_balancing"

@dataclass
class AdaptationEvent:
    """Event that triggers hybridization adaptation"""
    trigger_type: AdaptationTrigger
    timestamp: float
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    severity: float = 1.0  # 0.0 to 1.0

class DynamicHybridizationManager:
    """
    Manages real-time adaptation of hybridization strategies
    """
    
    def __init__(self, algorithms: List[str], adaptation_window: int = 100):
        self.algorithms = algorithms
        self.adaptation_window = adaptation_window
        
        # Current strategy configuration
        self.current_strategy = "weighted"
        self.current_weights = {algo: 1.0/len(algorithms) for algo in algorithms}
        self.current_switching_rules = []
        
        # Performance monitoring
        self.performance_history = deque(maxlen=adaptation_window)
        self.algorithm_performance = defaultdict(lambda: deque(maxlen=adaptation_window))
        
        # Adaptation configuration
        self.adaptation_thresholds = {
            AdaptationTrigger.PERFORMANCE_DEGRADATION: 0.1,  # 10% performance drop
            AdaptationTrigger.CONTEXT_CHANGE: 0.3,  # 30% context change
            AdaptationTrigger.USER_FEEDBACK: 0.2,  # 20% negative feedback
            AdaptationTrigger.TEMPORAL_DRIFT: 0.15,  # 15% temporal drift
            AdaptationTrigger.LOAD_BALANCING: 0.5   # 50% load imbalance
        }
        
        # Available strategies
        self.available_strategies = {
            'weighted': self._weighted_strategy,
            'switching': self._switching_strategy,
            'cascade': self._cascade_strategy,
            'voting': self._voting_strategy,
            'meta_learned': self._meta_learned_strategy
        }
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(list)
        self.strategy_contexts = defaultdict(list)
        
        # Real-time adaptation components
        self.adaptation_scheduler = AdaptationScheduler()
        self.context_monitor = ContextMonitor()
        self.performance_predictor = PerformancePredictor()
        
        # Thread safety
        self.lock = threading.RLock()
        
    def register_performance(self, user_id: str, algorithm_results: Dict[str, float], 
                           actual_outcome: float, context: Dict[str, Any]):
        """Register performance data for adaptation decisions"""
        
        with self.lock:
            timestamp = time.time()
            
            # Store overall performance
            overall_performance = sum(algorithm_results.values()) / len(algorithm_results)
            self.performance_history.append({
                'timestamp': timestamp,
                'user_id': user_id,
                'performance': overall_performance,
                'context': context.copy()
            })
            
            # Store algorithm-specific performance
            for algorithm, result in algorithm_results.items():
                accuracy = 1.0 - abs(result - actual_outcome) / 5.0  # Assuming 5-point scale
                self.algorithm_performance[algorithm].append({
                    'timestamp': timestamp,
                    'accuracy': max(0.0, accuracy),
                    'context': context.copy()
                })
            
            # Check for adaptation triggers
            self._check_adaptation_triggers(user_id, context, overall_performance)
    
    def _check_adaptation_triggers(self, user_id: str, context: Dict[str, Any], 
                                  current_performance: float):
        """Check if adaptation should be triggered"""
        
        adaptation_events = []
        
        # Performance degradation check
        if len(self.performance_history) >= 10:
            recent_performance = np.mean([p['performance'] for p in list(self.performance_history)[-10:]])
            historical_performance = np.mean([p['performance'] for p in list(self.performance_history)[:-10]])
            
            if historical_performance - recent_performance > self.adaptation_thresholds[AdaptationTrigger.PERFORMANCE_DEGRADATION]:
                adaptation_events.append(AdaptationEvent(
                    trigger_type=AdaptationTrigger.PERFORMANCE_DEGRADATION,
                    timestamp=time.time(),
                    user_id=user_id,
                    context=context,
                    performance_metrics={'performance_drop': historical_performance - recent_performance},
                    severity=(historical_performance - recent_performance) / historical_performance
                ))
        
        # Context change check
        context_change_severity = self.context_monitor.detect_context_change(context)
        if context_change_severity > self.adaptation_thresholds[AdaptationTrigger.CONTEXT_CHANGE]:
            adaptation_events.append(AdaptationEvent(
                trigger_type=AdaptationTrigger.CONTEXT_CHANGE,
                timestamp=time.time(),
                user_id=user_id,
                context=context,
                severity=context_change_severity
            ))
        
        # Temporal drift check
        temporal_drift = self._detect_temporal_drift()
        if temporal_drift > self.adaptation_thresholds[AdaptationTrigger.TEMPORAL_DRIFT]:
            adaptation_events.append(AdaptationEvent(
                trigger_type=AdaptationTrigger.TEMPORAL_DRIFT,
                timestamp=time.time(),
                severity=temporal_drift
            ))
        
        # Process adaptation events
        for event in adaptation_events:
            self._process_adaptation_event(event)
    
    def _process_adaptation_event(self, event: AdaptationEvent):
        """Process adaptation event and potentially trigger strategy change"""
        
        # Determine if adaptation is needed based on event severity
        if event.severity > 0.5:  # High severity threshold
            self._trigger_immediate_adaptation(event)
        elif event.severity > 0.3:  # Medium severity threshold
            self.adaptation_scheduler.schedule_adaptation(event, delay=60)  # 1 minute delay
        else:
            # Low severity - log for batch processing
            self.adaptation_scheduler.add_to_batch(event)
    
    def _trigger_immediate_adaptation(self, event: AdaptationEvent):
        """Trigger immediate strategy adaptation"""
        
        print(f"Triggering immediate adaptation for {event.trigger_type} (severity: {event.severity:.3f})")
        
        # Predict best strategy for current context
        predicted_strategy = self._predict_best_strategy(event.context, event.performance_metrics)
        
        if predicted_strategy != self.current_strategy:
            self._switch_strategy(predicted_strategy, event.context)
        else:
            # Optimize current strategy parameters
            self._optimize_current_strategy(event)
    
    def _predict_best_strategy(self, context: Dict[str, Any], 
                              performance_metrics: Dict[str, float]) -> str:
        """Predict best hybridization strategy for given context"""
        
        # Use performance predictor
        strategy_scores = {}
        
        for strategy_name in self.available_strategies.keys():
            # Get historical performance for this strategy in similar contexts
            similar_contexts = self._find_similar_contexts(context, strategy_name)
            
            if similar_contexts:
                avg_performance = np.mean([ctx['performance'] for ctx in similar_contexts])
                strategy_scores[strategy_name] = avg_performance
            else:
                # Default score for unexplored strategies
                strategy_scores[strategy_name] = 0.5
        
        # Add exploration bonus for less-used strategies
        total_usage = sum(len(self.strategy_contexts[s]) for s in strategy_scores.keys())
        
        for strategy in strategy_scores:
            usage_count = len(self.strategy_contexts[strategy])
            if total_usage > 0:
                exploration_bonus = 0.1 * (1.0 - usage_count / total_usage)
                strategy_scores[strategy] += exploration_bonus
        
        # Return strategy with highest score
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _switch_strategy(self, new_strategy: str, context: Dict[str, Any]):
        """Switch to new hybridization strategy"""
        
        with self.lock:
            old_strategy = self.current_strategy
            self.current_strategy = new_strategy
            
            # Configure new strategy
            if new_strategy == 'weighted':
                self._configure_weighted_strategy(context)
            elif new_strategy == 'switching':
                self._configure_switching_strategy(context)
            elif new_strategy == 'cascade':
                self._configure_cascade_strategy(context)
            
            print(f"Switched hybridization strategy: {old_strategy} -> {new_strategy}")
    
    def _optimize_current_strategy(self, event: AdaptationEvent):
        """Optimize parameters of current strategy"""
        
        if self.current_strategy == 'weighted':
            self._optimize_weights(event)
        elif self.current_strategy == 'switching':
            self._optimize_switching_rules(event)
        elif self.current_strategy == 'cascade':
            self._optimize_cascade_order(event)
    
    def _optimize_weights(self, event: AdaptationEvent):
        """Optimize weights for weighted strategy"""
        
        # Get recent algorithm performance
        recent_performance = {}
        for algorithm in self.algorithms:
            if self.algorithm_performance[algorithm]:
                recent_perfs = list(self.algorithm_performance[algorithm])[-10:]
                recent_performance[algorithm] = np.mean([p['accuracy'] for p in recent_perfs])
            else:
                recent_performance[algorithm] = 0.5
        
        # Update weights based on performance
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            new_weights = {
                algo: perf / total_performance 
                for algo, perf in recent_performance.items()
            }
            
            # Smooth transition to avoid abrupt changes
            smoothing_factor = 0.3
            for algo in self.current_weights:
                self.current_weights[algo] = (
                    smoothing_factor * new_weights.get(algo, 0) + 
                    (1 - smoothing_factor) * self.current_weights[algo]
                )
    
    def _detect_temporal_drift(self) -> float:
        """Detect temporal drift in recommendation patterns"""
        
        if len(self.performance_history) < 20:
            return 0.0
        
        # Split history into two halves
        history_list = list(self.performance_history)
        mid_point = len(history_list) // 2
        
        early_half = history_list[:mid_point]
        recent_half = history_list[mid_point:]
        
        # Compare performance distributions
        early_performance = [h['performance'] for h in early_half]
        recent_performance = [h['performance'] for h in recent_half]
        
        # Simple drift detection using mean difference
        early_mean = np.mean(early_performance)
        recent_mean = np.mean(recent_performance)
        
        # Normalize by historical standard deviation
        all_performance = early_performance + recent_performance
        overall_std = np.std(all_performance)
        
        if overall_std > 0:
            drift_magnitude = abs(early_mean - recent_mean) / overall_std
        else:
            drift_magnitude = 0.0
        
        return min(1.0, drift_magnitude)
    
    def _find_similar_contexts(self, query_context: Dict[str, Any], 
                              strategy_name: str) -> List[Dict[str, Any]]:
        """Find similar contexts where strategy was used"""
        
        if strategy_name not in self.strategy_contexts:
            return []
        
        similar_contexts = []
        
        for stored_context in self.strategy_contexts[strategy_name]:
            similarity = self._compute_context_similarity(query_context, stored_context['context'])
            
            if similarity > 0.7:  # Similarity threshold
                similar_contexts.append({
                    'context': stored_context['context'],
                    'performance': stored_context['performance'],
                    'similarity': similarity
                })
        
        return similar_contexts
    
    def _compute_context_similarity(self, context1: Dict[str, Any], 
                                   context2: Dict[str, Any]) -> float:
        """Compute similarity between two contexts"""
        
        # Simple context similarity based on key overlap and value similarity
        common_keys = set(context1.keys()).intersection(set(context2.keys()))
        
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_val
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple equality)
                similarity = 1.0 if val1 == val2 else 0.0
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                # List similarity (Jaccard)
                set1, set2 = set(val1), set(val2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarity_scores.append(similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    # Strategy implementations
    def _weighted_strategy(self, algorithm_predictions: Dict[str, float]) -> float:
        """Weighted combination strategy"""
        return sum(pred * self.current_weights.get(algo, 0) 
                  for algo, pred in algorithm_predictions.items())
    
    def _switching_strategy(self, algorithm_predictions: Dict[str, float]) -> float:
        """Switching strategy implementation"""
        # Use first available algorithm (simplified)
        return list(algorithm_predictions.values())[0] if algorithm_predictions else 0.0
    
    def _cascade_strategy(self, algorithm_predictions: Dict[str, float]) -> float:
        """Cascade strategy implementation"""
        # Simple cascade: average of predictions
        return np.mean(list(algorithm_predictions.values())) if algorithm_predictions else 0.0
    
    def _voting_strategy(self, algorithm_predictions: Dict[str, float]) -> float:
        """Voting strategy implementation"""
        # Majority vote based on threshold
        threshold = 3.0
        votes = sum(1 for pred in algorithm_predictions.values() if pred > threshold)
        return votes / len(algorithm_predictions) if algorithm_predictions else 0.0
    
    def _meta_learned_strategy(self, algorithm_predictions: Dict[str, float]) -> float:
        """Meta-learned strategy implementation"""
        # Placeholder for meta-learned combination
        return self._weighted_strategy(algorithm_predictions)
    
    def _configure_weighted_strategy(self, context: Dict[str, Any]):
        """Configure weighted strategy for context"""
        # Use current weights - could be optimized based on context
        pass
    
    def _configure_switching_strategy(self, context: Dict[str, Any]):
        """Configure switching strategy for context"""
        # Set up context-specific switching rules
        self.current_switching_rules = [
            {'condition': lambda ctx: ctx.get('time_of_day', 12) < 12, 'algorithm': self.algorithms[0]},
            {'condition': lambda ctx: ctx.get('is_mobile', False), 'algorithm': self.algorithms[1] if len(self.algorithms) > 1 else self.algorithms[0]}
        ]
    
    def _configure_cascade_strategy(self, context: Dict[str, Any]):
        """Configure cascade strategy for context"""
        # Set cascade order based on context
        # For now, use algorithm order
        pass
    
    def _optimize_switching_rules(self, event: AdaptationEvent):
        """Optimize switching rules"""
        # Placeholder for switching rule optimization
        pass
    
    def _optimize_cascade_order(self, event: AdaptationEvent):
        """Optimize cascade order"""
        # Placeholder for cascade order optimization
        pass
    
    def get_current_strategy_info(self) -> Dict[str, Any]:
        """Get information about current strategy"""
        
        return {
            'strategy': self.current_strategy,
            'weights': self.current_weights.copy(),
            'switching_rules': len(self.current_switching_rules),
            'performance_history_size': len(self.performance_history),
            'adaptation_triggers_active': len(self.adaptation_thresholds)
        }

class AdaptationScheduler:
    """Scheduler for managing adaptation timing"""
    
    def __init__(self):
        self.scheduled_adaptations = []
        self.batch_events = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def schedule_adaptation(self, event: AdaptationEvent, delay: float):
        """Schedule adaptation with delay"""
        
        def delayed_adaptation():
            time.sleep(delay)
            print(f"Executing scheduled adaptation for {event.trigger_type}")
            # Process the adaptation event
        
        future = self.executor.submit(delayed_adaptation)
        self.scheduled_adaptations.append((event, future))
    
    def add_to_batch(self, event: AdaptationEvent):
        """Add event to batch processing queue"""
        self.batch_events.append(event)
        
        # Process batch when it reaches certain size
        if len(self.batch_events) >= 10:
            self._process_batch()
    
    def _process_batch(self):
        """Process batch of low-severity events"""
        
        if not self.batch_events:
            return
        
        print(f"Processing batch of {len(self.batch_events)} adaptation events")
        
        # Analyze batch for patterns
        trigger_counts = defaultdict(int)
        for event in self.batch_events:
            trigger_counts[event.trigger_type] += 1
        
        # If many events of same type, consider adaptation
        for trigger_type, count in trigger_counts.items():
            if count >= 5:  # Threshold for batch adaptation
                print(f"Triggering batch adaptation for {trigger_type} (count: {count})")
        
        # Clear batch
        self.batch_events.clear()

class ContextMonitor:
    """Monitor context changes for adaptation triggers"""
    
    def __init__(self, context_history_size: int = 50):
        self.context_history = deque(maxlen=context_history_size)
        
    def detect_context_change(self, current_context: Dict[str, Any]) -> float:
        """Detect significant context changes"""
        
        if not self.context_history:
            self.context_history.append(current_context)
            return 0.0
        
        # Compare with recent context
        recent_context = self.context_history[-1]
        change_magnitude = self._compute_context_change(recent_context, current_context)
        
        # Store current context
        self.context_history.append(current_context)
        
        return change_magnitude
    
    def _compute_context_change(self, old_context: Dict[str, Any], 
                               new_context: Dict[str, Any]) -> float:
        """Compute magnitude of context change"""
        
        all_keys = set(old_context.keys()).union(set(new_context.keys()))
        
        if not all_keys:
            return 0.0
        
        changes = []
        
        for key in all_keys:
            old_val = old_context.get(key)
            new_val = new_context.get(key)
            
            if old_val is None or new_val is None:
                changes.append(1.0)  # New or removed key
            elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                # Numerical change
                max_val = max(abs(old_val), abs(new_val), 1.0)
                change = abs(old_val - new_val) / max_val
                changes.append(change)
            elif old_val != new_val:
                changes.append(1.0)  # Different values
            else:
                changes.append(0.0)  # Same values
        
        return np.mean(changes)

class PerformancePredictor:
    """Predict performance of hybridization strategies"""
    
    def __init__(self):
        self.prediction_models = {}
        self.feature_extractors = {}
        
    def predict_strategy_performance(self, strategy: str, context: Dict[str, Any]) -> float:
        """Predict performance of strategy in given context"""
        
        if strategy not in self.prediction_models:
            return 0.5  # Default prediction
        
        # Extract features from context
        features = self._extract_context_features(context)
        
        # Use prediction model
        try:
            predicted_performance = self.prediction_models[strategy].predict([features])[0]
            return max(0.0, min(1.0, predicted_performance))
        except Exception:
            return 0.5
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract numerical features from context"""
        
        features = []
        
        # Time features
        features.append(context.get('hour_of_day', 12) / 24.0)
        features.append(context.get('day_of_week', 3) / 7.0)
        
        # User features
        features.append(context.get('user_activity_level', 50) / 100.0)
        features.append(context.get('user_age', 30) / 100.0)
        
        # Device features
        features.append(float(context.get('is_mobile', False)))
        features.append(context.get('screen_size', 1000) / 2000.0)
        
        # Session features
        features.append(context.get('session_length', 30) / 120.0)
        features.append(context.get('pages_viewed', 5) / 20.0)
        
        # Pad or truncate to fixed size
        target_size = 10
        while len(features) < target_size:
            features.append(0.0)
        
        return features[:target_size]
```

## 2. Context-Driven Dynamic Fusion

### Contextual Adaptation Engine

```python
class ContextualAdaptationEngine:
    """
    Engine for context-driven dynamic fusion of recommendation algorithms
    """
    
    def __init__(self, algorithms: List[str]):
        self.algorithms = algorithms
        self.context_clusters = {}
        self.cluster_strategies = {}
        self.context_classifier = None
        
        # Context dimensions and their importance
        self.context_dimensions = {
            'temporal': ['hour_of_day', 'day_of_week', 'season', 'is_holiday'],
            'user': ['age_group', 'activity_level', 'preference_diversity', 'tenure'],
            'device': ['device_type', 'screen_size', 'connection_speed', 'battery_level'],
            'session': ['session_length', 'interaction_count', 'browsing_depth', 'search_queries'],
            'social': ['social_context', 'group_size', 'social_influence', 'trending_topics'],
            'environmental': ['location_type', 'weather', 'noise_level', 'lighting']
        }
        
        self.dimension_weights = {dim: 1.0 for dim in self.context_dimensions}
        
        # Fusion strategies for different contexts
        self.fusion_strategies = {
            'exploration_focused': ExplorationFusionStrategy(),
            'exploitation_focused': ExploitationFusionStrategy(),
            'diversity_focused': DiversityFusionStrategy(),
            'accuracy_focused': AccuracyFusionStrategy(),
            'speed_focused': SpeedFusionStrategy(),
            'balanced': BalancedFusionStrategy()
        }
        
        # Context-strategy mapping learning
        self.context_strategy_rewards = defaultdict(lambda: defaultdict(list))
        
    def adapt_fusion_strategy(self, context: Dict[str, Any]) -> str:
        """Adapt fusion strategy based on current context"""
        
        # Classify context into cluster
        context_cluster = self._classify_context(context)
        
        # Get best strategy for this cluster
        if context_cluster in self.cluster_strategies:
            strategy_name = self.cluster_strategies[context_cluster]
        else:
            # Learn strategy for new cluster
            strategy_name = self._learn_strategy_for_cluster(context_cluster, context)
        
        return strategy_name, context_cluster
    
    def apply_fusion(self, algorithm_predictions: Dict[str, float], 
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply context-driven fusion"""
        
        strategy_name, context_cluster = self.adapt_fusion_strategy(context)
        fusion_strategy = self.fusion_strategies[strategy_name]
        
        # Apply fusion strategy
        result = fusion_strategy.fuse(algorithm_predictions, context)
        
        # Add metadata
        result['metadata'] = {
            'strategy_used': strategy_name,
            'context_cluster': context_cluster,
            'context_dimensions': self._extract_context_features(context)
        }
        
        return result
    
    def update_strategy_performance(self, context: Dict[str, Any], 
                                  strategy_name: str, performance: float):
        """Update performance feedback for context-strategy combination"""
        
        context_cluster = self._classify_context(context)
        self.context_strategy_rewards[context_cluster][strategy_name].append(performance)
        
        # Update cluster-strategy mapping if necessary
        self._update_cluster_strategy_mapping(context_cluster)
    
    def _classify_context(self, context: Dict[str, Any]) -> str:
        """Classify context into predefined clusters"""
        
        if self.context_classifier is None:
            self._initialize_context_classifier()
        
        # Extract context features
        features = self._extract_context_features(context)
        
        # Simple clustering based on feature ranges
        cluster_key = self._compute_cluster_key(features)
        
        return cluster_key
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize context features"""
        
        features = {}
        
        # Temporal features
        features['hour_normalized'] = context.get('hour_of_day', 12) / 24.0
        features['day_normalized'] = context.get('day_of_week', 3) / 7.0
        features['is_weekend'] = float(context.get('day_of_week', 3) >= 5)
        
        # User features
        features['user_activity'] = min(1.0, context.get('user_activity_level', 50) / 100.0)
        features['user_tenure'] = min(1.0, context.get('user_tenure_days', 365) / 1000.0)
        features['user_diversity'] = context.get('preference_diversity_score', 0.5)
        
        # Device features
        features['is_mobile'] = float(context.get('device_type', 'desktop') == 'mobile')
        features['screen_size_norm'] = min(1.0, context.get('screen_width', 1000) / 2000.0)
        features['connection_quality'] = context.get('connection_speed_mbps', 10) / 100.0
        
        # Session features
        features['session_length_norm'] = min(1.0, context.get('session_length_minutes', 30) / 120.0)
        features['interaction_intensity'] = min(1.0, context.get('interactions_per_minute', 2) / 10.0)
        
        # Social features
        features['social_influence'] = context.get('social_influence_score', 0.5)
        features['group_context'] = float(context.get('group_size', 1) > 1)
        
        return features
    
    def _compute_cluster_key(self, features: Dict[str, float]) -> str:
        """Compute cluster key from features"""
        
        # Simple discretization-based clustering
        cluster_parts = []
        
        # Time-based clustering
        hour_norm = features.get('hour_normalized', 0.5)
        if hour_norm < 0.25:  # Night
            cluster_parts.append('night')
        elif hour_norm < 0.5:  # Morning
            cluster_parts.append('morning')
        elif hour_norm < 0.75:  # Afternoon
            cluster_parts.append('afternoon')
        else:  # Evening
            cluster_parts.append('evening')
        
        # Device-based clustering
        if features.get('is_mobile', 0) > 0.5:
            cluster_parts.append('mobile')
        else:
            cluster_parts.append('desktop')
        
        # Activity-based clustering
        activity = features.get('user_activity', 0.5)
        if activity < 0.3:
            cluster_parts.append('low_activity')
        elif activity < 0.7:
            cluster_parts.append('medium_activity')
        else:
            cluster_parts.append('high_activity')
        
        # Session-based clustering
        session_length = features.get('session_length_norm', 0.5)
        if session_length < 0.3:
            cluster_parts.append('short_session')
        elif session_length < 0.7:
            cluster_parts.append('medium_session')
        else:
            cluster_parts.append('long_session')
        
        return '_'.join(cluster_parts)
    
    def _learn_strategy_for_cluster(self, cluster: str, context: Dict[str, Any]) -> str:
        """Learn best strategy for new context cluster"""
        
        # Initialize with balanced strategy
        initial_strategy = 'balanced'
        self.cluster_strategies[cluster] = initial_strategy
        
        return initial_strategy
    
    def _update_cluster_strategy_mapping(self, cluster: str):
        """Update strategy mapping based on performance feedback"""
        
        if cluster not in self.context_strategy_rewards:
            return
        
        # Find best performing strategy for this cluster
        strategy_performances = {}
        
        for strategy, rewards in self.context_strategy_rewards[cluster].items():
            if rewards:
                strategy_performances[strategy] = np.mean(rewards[-10:])  # Recent performance
        
        if strategy_performances:
            best_strategy = max(strategy_performances.items(), key=lambda x: x[1])[0]
            
            # Update mapping if performance improvement is significant
            current_strategy = self.cluster_strategies.get(cluster, 'balanced')
            current_performance = strategy_performances.get(current_strategy, 0.0)
            best_performance = strategy_performances[best_strategy]
            
            if best_performance > current_performance + 0.05:  # 5% improvement threshold
                self.cluster_strategies[cluster] = best_strategy
                print(f"Updated strategy for cluster {cluster}: {current_strategy} -> {best_strategy}")
    
    def _initialize_context_classifier(self):
        """Initialize context classification system"""
        
        # Placeholder for more sophisticated classification
        self.context_classifier = "simple_discretization"

# Fusion Strategy Implementations
class FusionStrategy(ABC):
    """Abstract base class for fusion strategies"""
    
    @abstractmethod
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        pass

class ExplorationFusionStrategy(FusionStrategy):
    """Fusion strategy focused on exploration and diversity"""
    
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Boost less confident predictions to encourage exploration
        min_prediction = min(algorithm_predictions.values())
        max_prediction = max(algorithm_predictions.values())
        
        if max_prediction > min_prediction:
            # Normalize and boost lower predictions
            exploration_weights = {}
            for algo, pred in algorithm_predictions.items():
                # Inverse relationship: lower predictions get higher weights
                normalized_pred = (pred - min_prediction) / (max_prediction - min_prediction)
                exploration_weight = 1.0 - normalized_pred + 0.2  # Add base weight
                exploration_weights[algo] = exploration_weight
            
            # Normalize weights
            total_weight = sum(exploration_weights.values())
            normalized_weights = {algo: w/total_weight for algo, w in exploration_weights.items()}
            
            # Compute exploration-focused prediction
            fused_prediction = sum(pred * normalized_weights[algo] 
                                 for algo, pred in algorithm_predictions.items())
        else:
            # All predictions are similar, use uniform weighting
            fused_prediction = np.mean(list(algorithm_predictions.values()))
        
        return {
            'prediction': fused_prediction,
            'confidence': 0.7,  # Lower confidence due to exploration focus
            'strategy_type': 'exploration',
            'weights': normalized_weights if 'normalized_weights' in locals() else None
        }

class ExploitationFusionStrategy(FusionStrategy):
    """Fusion strategy focused on exploitation of best algorithms"""
    
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Weight algorithms by their confidence/performance
        algorithm_confidences = context.get('algorithm_confidences', {})
        
        if algorithm_confidences:
            # Use confidence-based weighting
            total_confidence = sum(algorithm_confidences.values())
            
            if total_confidence > 0:
                exploitation_weights = {
                    algo: conf / total_confidence 
                    for algo, conf in algorithm_confidences.items()
                }
            else:
                exploitation_weights = {algo: 1.0/len(algorithm_predictions) 
                                     for algo in algorithm_predictions}
        else:
            # Use prediction magnitude as proxy for confidence
            pred_magnitudes = {algo: abs(pred - 2.5) for algo, pred in algorithm_predictions.items()}
            total_magnitude = sum(pred_magnitudes.values())
            
            if total_magnitude > 0:
                exploitation_weights = {
                    algo: mag / total_magnitude 
                    for algo, mag in pred_magnitudes.items()
                }
            else:
                exploitation_weights = {algo: 1.0/len(algorithm_predictions) 
                                     for algo in algorithm_predictions}
        
        # Compute exploitation-focused prediction
        fused_prediction = sum(pred * exploitation_weights[algo] 
                             for algo, pred in algorithm_predictions.items())
        
        return {
            'prediction': fused_prediction,
            'confidence': 0.9,  # Higher confidence due to exploitation focus
            'strategy_type': 'exploitation',
            'weights': exploitation_weights
        }

class DiversityFusionStrategy(FusionStrategy):
    """Fusion strategy focused on maintaining diversity"""
    
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Calculate prediction diversity
        predictions = list(algorithm_predictions.values())
        prediction_std = np.std(predictions)
        
        if prediction_std > 0.5:  # High diversity
            # Use uniform weighting to maintain diversity
            uniform_weights = {algo: 1.0/len(algorithm_predictions) 
                             for algo in algorithm_predictions}
            fused_prediction = np.mean(predictions)
        else:  # Low diversity
            # Boost algorithms with different predictions
            mean_prediction = np.mean(predictions)
            diversity_weights = {}
            
            for algo, pred in algorithm_predictions.items():
                # Weight by distance from mean
                diversity_weight = abs(pred - mean_prediction) + 0.1  # Add base weight
                diversity_weights[algo] = diversity_weight
            
            # Normalize weights
            total_weight = sum(diversity_weights.values())
            normalized_weights = {algo: w/total_weight for algo, w in diversity_weights.items()}
            
            fused_prediction = sum(pred * normalized_weights[algo] 
                                 for algo, pred in algorithm_predictions.items())
            uniform_weights = normalized_weights
        
        return {
            'prediction': fused_prediction,
            'confidence': 0.8,
            'strategy_type': 'diversity',
            'weights': uniform_weights,
            'diversity_score': prediction_std
        }

class AccuracyFusionStrategy(FusionStrategy):
    """Fusion strategy focused on maximizing accuracy"""
    
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Use historical accuracy to weight algorithms
        algorithm_accuracies = context.get('algorithm_accuracies', {})
        
        if algorithm_accuracies:
            # Weight by accuracy squared to emphasize best performers
            accuracy_weights = {
                algo: acc ** 2 for algo, acc in algorithm_accuracies.items()
                if algo in algorithm_predictions
            }
            
            total_weight = sum(accuracy_weights.values())
            if total_weight > 0:
                normalized_weights = {algo: w/total_weight for algo, w in accuracy_weights.items()}
            else:
                normalized_weights = {algo: 1.0/len(algorithm_predictions) 
                                    for algo in algorithm_predictions}
        else:
            # Fallback to uniform weighting
            normalized_weights = {algo: 1.0/len(algorithm_predictions) 
                                for algo in algorithm_predictions}
        
        # Compute accuracy-focused prediction
        fused_prediction = sum(pred * normalized_weights.get(algo, 0) 
                             for algo, pred in algorithm_predictions.items())
        
        return {
            'prediction': fused_prediction,
            'confidence': 0.95,  # Highest confidence for accuracy focus
            'strategy_type': 'accuracy',
            'weights': normalized_weights
        }

class SpeedFusionStrategy(FusionStrategy):
    """Fusion strategy focused on fast response times"""
    
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Use only fastest algorithms or simple combination
        algorithm_speeds = context.get('algorithm_response_times', {})
        
        if algorithm_speeds:
            # Select fastest algorithms (lower response time is better)
            sorted_algos = sorted(algorithm_speeds.items(), key=lambda x: x[1])
            fast_algos = [algo for algo, _ in sorted_algos[:2]]  # Top 2 fastest
            
            # Use only fast algorithms
            fast_predictions = {algo: pred for algo, pred in algorithm_predictions.items() 
                              if algo in fast_algos}
            
            if fast_predictions:
                fused_prediction = np.mean(list(fast_predictions.values()))
                speed_weights = {algo: 1.0/len(fast_predictions) for algo in fast_predictions}
            else:
                fused_prediction = np.mean(list(algorithm_predictions.values()))
                speed_weights = {algo: 1.0/len(algorithm_predictions) 
                               for algo in algorithm_predictions}
        else:
            # Simple average for speed
            fused_prediction = np.mean(list(algorithm_predictions.values()))
            speed_weights = {algo: 1.0/len(algorithm_predictions) 
                           for algo in algorithm_predictions}
        
        return {
            'prediction': fused_prediction,
            'confidence': 0.75,
            'strategy_type': 'speed',
            'weights': speed_weights,
            'processing_time': 'minimized'
        }

class BalancedFusionStrategy(FusionStrategy):
    """Balanced fusion strategy combining multiple objectives"""
    
    def fuse(self, algorithm_predictions: Dict[str, float], 
            context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Combine multiple factors with balanced weights
        factors = {
            'accuracy': context.get('algorithm_accuracies', {}),
            'confidence': context.get('algorithm_confidences', {}),
            'speed': context.get('algorithm_response_times', {})  # Inverted for speed
        }
        
        # Compute combined weights
        combined_weights = {}
        
        for algo in algorithm_predictions:
            weight_components = []
            
            # Accuracy component
            if factors['accuracy'] and algo in factors['accuracy']:
                weight_components.append(factors['accuracy'][algo])
            else:
                weight_components.append(0.5)  # Default
            
            # Confidence component
            if factors['confidence'] and algo in factors['confidence']:
                weight_components.append(factors['confidence'][algo])
            else:
                weight_components.append(0.5)  # Default
            
            # Speed component (inverted - lower time is better)
            if factors['speed'] and algo in factors['speed']:
                max_time = max(factors['speed'].values())
                speed_weight = (max_time - factors['speed'][algo]) / max_time if max_time > 0 else 0.5
                weight_components.append(speed_weight)
            else:
                weight_components.append(0.5)  # Default
            
            # Balanced combination
            combined_weights[algo] = np.mean(weight_components)
        
        # Normalize weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            normalized_weights = {algo: w/total_weight for algo, w in combined_weights.items()}
        else:
            normalized_weights = {algo: 1.0/len(algorithm_predictions) 
                                for algo in algorithm_predictions}
        
        # Compute balanced prediction
        fused_prediction = sum(pred * normalized_weights[algo] 
                             for algo, pred in algorithm_predictions.items())
        
        return {
            'prediction': fused_prediction,
            'confidence': 0.85,
            'strategy_type': 'balanced',
            'weights': normalized_weights,
            'balance_factors': ['accuracy', 'confidence', 'speed']
        }
```

## 3. Study Questions

### Beginner Level

1. What are the key triggers for dynamic adaptation in hybrid recommendation systems?
2. How does context-driven fusion differ from static combination strategies?
3. What role does real-time performance monitoring play in dynamic hybridization?
4. How can temporal drift be detected and addressed in recommendation systems?
5. What are the trade-offs between different fusion strategies (exploration vs exploitation)?

### Intermediate Level

6. Implement a dynamic hybridization system that adapts to user behavior changes throughout the day.
7. How would you design a context classification system for different recommendation scenarios?
8. What are the computational challenges of real-time strategy adaptation in production systems?
9. How would you balance stability vs adaptability in dynamic hybrid systems?
10. Design a system that can detect and adapt to seasonal patterns in user preferences.

### Advanced Level

11. Implement a meta-learning system that can automatically discover new fusion strategies based on context patterns.
12. Design a dynamic hybridization system that can handle concept drift while maintaining recommendation quality.
13. How would you create a system that can adapt fusion strategies based on business objectives (revenue, engagement, etc.)?
14. Implement a distributed dynamic hybridization system that can coordinate across multiple recommendation services.
15. Design a system that can predict when strategy adaptation will be beneficial versus when it might harm performance.

### Tricky Questions

16. How would you design a dynamic hybridization system that can adapt to adversarial attacks or gaming attempts?
17. What are the privacy implications of context-driven adaptation, and how would you implement privacy-preserving dynamic fusion?
18. How would you handle the cold start problem for new contexts that haven't been seen before?
19. Design a system that can balance individual user optimization with global system performance in dynamic hybridization.
20. How would you implement fair dynamic hybridization that ensures equitable treatment across different user groups?

## Key Takeaways

1. **Dynamic adaptation** enables hybrid systems to respond to changing conditions in real-time
2. **Context-driven fusion** allows strategies to be optimized for specific user situations and environments
3. **Performance monitoring** is essential for detecting when adaptation is needed
4. **Multi-trigger adaptation** provides robust detection of various types of system changes
5. **Strategy diversity** allows systems to optimize for different objectives (accuracy, speed, diversity)
6. **Real-time optimization** requires careful balance between responsiveness and stability
7. **Context classification** enables systematic application of appropriate fusion strategies

## Next Session Preview

In Day 5.6, we'll explore **Evaluation of Hybrid Systems**, covering:
- Comprehensive evaluation frameworks for hybrid systems
- Multi-objective evaluation metrics
- A/B testing methodologies for hybrid recommendations
- Long-term impact assessment
- Fairness and bias evaluation in hybrid systems