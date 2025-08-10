# Day 4.3: Feature Store Architecture Deep Dive

## 🏪 Storage Layers & Feature Store Deep Dive - Part 3

**Focus**: Feast, Tecton, Built-in vs Custom Solutions, Feature Versioning & Consistency  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master feature store architecture patterns and trade-offs (Feast vs Tecton vs Custom)
- Understand feature versioning, lineage, and consistency guarantee models
- Learn online vs offline store synchronization and freshness SLA management
- Implement advanced feature serving patterns and performance optimization

---

## 🏗️ Feature Store Theoretical Foundation

### **Feature Store Mathematical Model**

#### **Feature Consistency and Freshness Framework**
```
Feature Freshness Function:
F(t) = max(0, 1 - (t_current - t_computed) / SLA_freshness)

Where:
- t_current = current time
- t_computed = feature computation time  
- SLA_freshness = maximum acceptable staleness

Feature Consistency Model:
Consistency(F_online, F_offline) = 1 - |F_online - F_offline| / max(|F_online|, |F_offline|)

Training-Serving Skew:
Skew(F_training, F_serving) = Σ|F_training_i - F_serving_i| / n
```

```python
import asyncio
import redis
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib

class FeatureStoreType(Enum):
    """Types of feature store implementations"""
    FEAST = "feast"
    TECTON = "tecton" 
    SAGEMAKER = "sagemaker"
    CUSTOM = "custom"
    DATABRICKS = "databricks"

class FeatureStoreComponent(Enum):
    """Core components of feature store architecture"""
    OFFLINE_STORE = "offline_store"
    ONLINE_STORE = "online_store"
    FEATURE_REGISTRY = "feature_registry"
    SERVING_LAYER = "serving_layer"
    COMPUTATION_ENGINE = "computation_engine"
    MONITORING_SYSTEM = "monitoring_system"

class ConsistencyLevel(Enum):
    """Feature consistency levels"""
    EVENTUAL = "eventual"
    STRONG = "strong"
    SESSION = "session"
    MONOTONIC_READ = "monotonic_read"

@dataclass
class FeatureDefinition:
    """Defines a feature with all its metadata"""
    feature_name: str
    feature_type: str  # categorical, numerical, embedding
    data_type: str     # int64, float64, string, array
    description: str
    owner: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Computation metadata
    source_query: Optional[str] = None
    transformation_logic: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # SLA requirements
    freshness_sla_minutes: int = 60
    availability_sla: float = 0.99
    
    # Versioning
    version: str = "1.0.0"
    schema_version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FeatureVector:
    """Represents a feature vector with metadata"""
    entity_id: str
    features: Dict[str, Any]
    feature_timestamp: datetime
    computation_timestamp: datetime
    version_info: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_freshness(self, current_time: datetime = None) -> float:
        """Calculate feature freshness score"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        staleness_seconds = (current_time - self.feature_timestamp).total_seconds()
        # Exponential decay model for freshness
        freshness = np.exp(-staleness_seconds / 3600)  # 1-hour half-life
        return max(0.0, min(1.0, freshness))

class FeatureStoreArchitectureComparator:
    """Compare different feature store architectures"""
    
    def __init__(self):
        self.architecture_profiles = self._initialize_architecture_profiles()
        self.performance_benchmarks = {}
        
    def _initialize_architecture_profiles(self) -> Dict[FeatureStoreType, Dict[str, Any]]:
        """Initialize architecture profiles for different feature stores"""
        
        return {
            FeatureStoreType.FEAST: {
                'architecture_pattern': 'decoupled_compute_storage',
                'online_store_options': ['Redis', 'DynamoDB', 'Datastore'],
                'offline_store_options': ['BigQuery', 'Snowflake', 'Redshift', 'Spark'],
                'serving_latency_p99_ms': 10,
                'throughput_qps': 10000,
                'feature_freshness_minutes': 5,
                'setup_complexity': 'medium',
                'operational_overhead': 'medium',
                'cost_model': 'pay_per_compute_storage',
                'strengths': [
                    'Open source with strong community',
                    'Flexible architecture with pluggable stores',
                    'Good integration with ML frameworks',
                    'Strong consistency guarantees'
                ],
                'limitations': [
                    'Complex setup and configuration',
                    'Limited built-in feature transformation',
                    'Requires separate orchestration system',
                    'Manual scaling and optimization'
                ],
                'ideal_use_cases': [
                    'Organizations wanting open-source solution',
                    'Teams with existing data infrastructure',
                    'Custom transformation logic requirements',
                    'Multi-cloud or hybrid deployments'
                ]
            },
            
            FeatureStoreType.TECTON: {
                'architecture_pattern': 'unified_platform',
                'online_store_options': ['Managed Redis', 'DynamoDB'],
                'offline_store_options': ['Managed Spark', 'Snowflake', 'BigQuery'],
                'serving_latency_p99_ms': 5,
                'throughput_qps': 50000,
                'feature_freshness_minutes': 1,
                'setup_complexity': 'low',
                'operational_overhead': 'low',
                'cost_model': 'managed_service_premium',
                'strengths': [
                    'Fully managed with minimal ops overhead',
                    'Advanced feature transformation engine',
                    'Built-in monitoring and alerting',
                    'Automatic scaling and optimization',
                    'Strong enterprise security features'
                ],
                'limitations': [
                    'Vendor lock-in concerns',
                    'Higher cost for managed services',
                    'Less flexibility in infrastructure choices',
                    'Proprietary feature definition language'
                ],
                'ideal_use_cases': [
                    'Enterprises wanting managed solution',
                    'Teams with limited ML infrastructure expertise',
                    'High-performance serving requirements',
                    'Complex feature transformation needs'
                ]
            },
            
            FeatureStoreType.SAGEMAKER: {
                'architecture_pattern': 'aws_native_integrated',
                'online_store_options': ['Managed In-Memory'],
                'offline_store_options': ['S3', 'Glue', 'Athena'],
                'serving_latency_p99_ms': 8,
                'throughput_qps': 20000,
                'feature_freshness_minutes': 15,
                'setup_complexity': 'low',
                'operational_overhead': 'low',
                'cost_model': 'aws_pay_per_use',
                'strengths': [
                    'Native AWS integration',
                    'Built-in with SageMaker ML workflows',
                    'Automatic scaling',
                    'IAM-based security model'
                ],
                'limitations': [
                    'AWS vendor lock-in',
                    'Limited customization options',
                    'Basic transformation capabilities',
                    'Higher latency compared to specialized solutions'
                ],
                'ideal_use_cases': [
                    'AWS-centric organizations',
                    'SageMaker-based ML pipelines',
                    'Simple feature serving needs',
                    'Rapid prototyping and development'
                ]
            },
            
            FeatureStoreType.CUSTOM: {
                'architecture_pattern': 'bespoke_implementation',
                'online_store_options': ['Any'],
                'offline_store_options': ['Any'],
                'serving_latency_p99_ms': 'variable',
                'throughput_qps': 'variable',
                'feature_freshness_minutes': 'configurable',
                'setup_complexity': 'high',
                'operational_overhead': 'high',
                'cost_model': 'infrastructure_only',
                'strengths': [
                    'Complete control and customization',
                    'Optimal performance for specific use cases',
                    'No vendor lock-in',
                    'Cost optimization potential'
                ],
                'limitations': [
                    'High development and maintenance effort',
                    'Requires deep expertise',
                    'Long time to production',
                    'Risk of reinventing the wheel'
                ],
                'ideal_use_cases': [
                    'Unique requirements not met by existing solutions',
                    'Performance-critical applications',
                    'Organizations with strong engineering teams',
                    'Long-term strategic investments'
                ]
            }
        }
    
    def compare_architectures(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Compare feature store architectures against requirements"""
        
        comparison_result = {
            'requirements': requirements,
            'architecture_scores': {},
            'detailed_analysis': {},
            'recommendation': None,
            'recommendation_reasoning': []
        }
        
        weights = requirements.get('decision_weights', {
            'performance': 0.25,
            'cost': 0.20,
            'operational_complexity': 0.20,
            'flexibility': 0.15,
            'time_to_market': 0.10,
            'vendor_lock_in_risk': 0.10
        })
        
        for arch_type, profile in self.architecture_profiles.items():
            score = self._calculate_architecture_score(profile, requirements, weights)
            comparison_result['architecture_scores'][arch_type.value] = score
            comparison_result['detailed_analysis'][arch_type.value] = {
                'profile': profile,
                'score_breakdown': score['detailed_scores'],
                'fit_analysis': self._analyze_requirements_fit(profile, requirements)
            }
        
        # Determine recommendation
        best_architecture = max(
            comparison_result['architecture_scores'].items(),
            key=lambda x: x[1]['total_score']
        )
        
        comparison_result['recommendation'] = best_architecture[0]
        comparison_result['recommendation_reasoning'] = best_architecture[1]['reasoning']
        
        return comparison_result
    
    def _calculate_architecture_score(self, profile: Dict[str, Any],
                                    requirements: Dict[str, Any],
                                    weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate weighted score for architecture against requirements"""
        
        scores = {}
        
        # Performance score
        req_latency_ms = requirements.get('max_latency_ms', 50)
        req_throughput_qps = requirements.get('min_throughput_qps', 1000)
        
        latency_score = 1.0 if profile['serving_latency_p99_ms'] <= req_latency_ms else 0.5
        throughput_score = 1.0 if profile['throughput_qps'] >= req_throughput_qps else 0.5
        scores['performance'] = (latency_score + throughput_score) / 2
        
        # Cost score (inverse - lower cost = higher score)
        cost_preference = requirements.get('cost_preference', 'balanced')  # low, balanced, premium
        cost_mapping = {'low': 0.8, 'balanced': 0.6, 'premium': 0.4}
        base_cost_score = cost_mapping.get(cost_preference, 0.6)
        
        if 'managed_service' in profile['cost_model']:
            cost_score = base_cost_score * 0.7  # Managed services are more expensive
        else:
            cost_score = base_cost_score
        scores['cost'] = cost_score
        
        # Operational complexity score (inverse - lower complexity = higher score)
        complexity_mapping = {'low': 1.0, 'medium': 0.6, 'high': 0.2}
        scores['operational_complexity'] = complexity_mapping.get(profile['setup_complexity'], 0.5)
        
        # Flexibility score
        flexibility_req = requirements.get('flexibility_importance', 'medium')
        if flexibility_req == 'high':
            flexibility_score = 1.0 if profile['architecture_pattern'] == 'bespoke_implementation' else 0.6
        else:
            flexibility_score = 0.8  # Most solutions provide reasonable flexibility
        scores['flexibility'] = flexibility_score
        
        # Time to market score
        ttm_mapping = {'low': 1.0, 'medium': 0.7, 'high': 0.3}
        scores['time_to_market'] = ttm_mapping.get(profile['setup_complexity'], 0.5)
        
        # Vendor lock-in risk score (inverse - higher risk = lower score)
        vendor_risk_tolerance = requirements.get('vendor_risk_tolerance', 'medium')
        if 'aws_native' in profile['architecture_pattern'] or 'managed_service' in profile['cost_model']:
            lock_in_risk = 'high'
        elif profile['architecture_pattern'] == 'bespoke_implementation':
            lock_in_risk = 'none'
        else:
            lock_in_risk = 'medium'
        
        risk_mapping = {'none': 1.0, 'low': 0.8, 'medium': 0.6, 'high': 0.3}
        scores['vendor_lock_in_risk'] = risk_mapping.get(lock_in_risk, 0.5)
        
        # Calculate weighted total score
        total_score = sum(scores[criterion] * weights[criterion] for criterion in scores)
        
        return {
            'total_score': total_score,
            'detailed_scores': scores,
            'reasoning': self._generate_scoring_reasoning(scores, profile)
        }
    
    def _generate_scoring_reasoning(self, scores: Dict[str, float],
                                  profile: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for architecture scoring"""
        
        reasoning = []
        
        if scores['performance'] >= 0.8:
            reasoning.append(f"Strong performance characteristics: {profile['serving_latency_p99_ms']}ms latency, {profile['throughput_qps']} QPS")
        
        if scores['operational_complexity'] >= 0.8:
            reasoning.append(f"Low operational overhead with {profile['setup_complexity']} setup complexity")
        
        if scores['cost'] >= 0.7:
            reasoning.append(f"Cost-effective solution with {profile['cost_model']} pricing model")
        
        if scores['flexibility'] >= 0.8:
            reasoning.append("High flexibility for customization and integration")
        
        if scores['vendor_lock_in_risk'] <= 0.4:
            reasoning.append("Potential vendor lock-in concerns should be evaluated")
        
        return reasoning

class AdvancedFeatureStore:
    """Advanced feature store implementation with comprehensive capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.online_store = self._initialize_online_store(config['online_store'])
        self.offline_store = self._initialize_offline_store(config['offline_store'])
        self.feature_registry = FeatureRegistry(config.get('registry', {}))
        self.version_manager = FeatureVersionManager()
        self.consistency_manager = FeatureConsistencyManager()
        self.monitoring_system = FeatureMonitoringSystem()
        
    def _initialize_online_store(self, config: Dict[str, Any]) -> 'OnlineFeatureStore':
        """Initialize online feature store based on configuration"""
        store_type = config.get('type', 'redis')
        
        if store_type == 'redis':
            return RedisOnlineStore(config)
        elif store_type == 'dynamodb':
            return DynamoDBOnlineStore(config)
        else:
            raise ValueError(f"Unsupported online store type: {store_type}")
    
    def _initialize_offline_store(self, config: Dict[str, Any]) -> 'OfflineFeatureStore':
        """Initialize offline feature store based on configuration"""
        store_type = config.get('type', 'bigquery')
        
        if store_type == 'bigquery':
            return BigQueryOfflineStore(config)
        elif store_type == 'snowflake':
            return SnowflakeOfflineStore(config)
        else:
            raise ValueError(f"Unsupported offline store type: {store_type}")
    
    async def get_online_features(self, entity_ids: List[str],
                                feature_names: List[str],
                                consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL) -> Dict[str, FeatureVector]:
        """Retrieve features from online store with consistency guarantees"""
        
        start_time = time.time()
        
        # Validate feature definitions
        validated_features = []
        for feature_name in feature_names:
            feature_def = await self.feature_registry.get_feature_definition(feature_name)
            if feature_def:
                validated_features.append(feature_def)
        
        # Retrieve features with consistency controls
        feature_vectors = {}
        
        if consistency_level == ConsistencyLevel.STRONG:
            # Strong consistency: ensure read-after-write consistency
            feature_vectors = await self._get_features_strong_consistency(
                entity_ids, validated_features
            )
        else:
            # Eventual consistency: allow stale reads for better performance
            feature_vectors = await self._get_features_eventual_consistency(
                entity_ids, validated_features
            )
        
        # Apply freshness filtering
        fresh_feature_vectors = {}
        for entity_id, vector in feature_vectors.items():
            if self._is_feature_vector_fresh(vector, validated_features):
                fresh_feature_vectors[entity_id] = vector
            else:
                # Feature vector is stale, trigger refresh if possible
                await self._trigger_feature_refresh(entity_id, validated_features)
        
        # Record metrics
        end_time = time.time()
        await self.monitoring_system.record_online_serving_metrics({
            'latency_ms': (end_time - start_time) * 1000,
            'entity_count': len(entity_ids),
            'feature_count': len(validated_features),
            'cache_hit_ratio': len(fresh_feature_vectors) / len(entity_ids) if entity_ids else 0,
            'consistency_level': consistency_level.value
        })
        
        return fresh_feature_vectors
    
    async def _get_features_strong_consistency(self, entity_ids: List[str],
                                             feature_definitions: List[FeatureDefinition]) -> Dict[str, FeatureVector]:
        """Retrieve features with strong consistency guarantees"""
        
        feature_vectors = {}
        
        # Use read-your-writes consistency
        for entity_id in entity_ids:
            # Check if there are recent writes for this entity
            recent_writes = await self.consistency_manager.get_recent_writes(entity_id)
            
            if recent_writes:
                # Wait for write propagation or read from primary
                await self.consistency_manager.ensure_write_propagation(entity_id, recent_writes)
            
            # Read features
            vector = await self.online_store.get_feature_vector(entity_id, feature_definitions)
            if vector:
                feature_vectors[entity_id] = vector
        
        return feature_vectors
    
    async def _get_features_eventual_consistency(self, entity_ids: List[str],
                                               feature_definitions: List[FeatureDefinition]) -> Dict[str, FeatureVector]:
        """Retrieve features with eventual consistency (better performance)"""
        
        # Batch retrieval for better performance
        return await self.online_store.batch_get_feature_vectors(entity_ids, feature_definitions)
    
    def _is_feature_vector_fresh(self, vector: FeatureVector,
                               feature_definitions: List[FeatureDefinition]) -> bool:
        """Check if feature vector meets freshness SLA"""
        
        current_time = datetime.utcnow()
        
        for feature_def in feature_definitions:
            feature_freshness_threshold = timedelta(minutes=feature_def.freshness_sla_minutes)
            feature_age = current_time - vector.feature_timestamp
            
            if feature_age > feature_freshness_threshold:
                return False
        
        return True
    
    async def _trigger_feature_refresh(self, entity_id: str,
                                     feature_definitions: List[FeatureDefinition]):
        """Trigger asynchronous feature refresh for stale features"""
        
        # This would trigger feature computation pipeline
        refresh_request = {
            'entity_id': entity_id,
            'features_to_refresh': [f.feature_name for f in feature_definitions],
            'priority': 'high',
            'requested_at': datetime.utcnow().isoformat()
        }
        
        # Send to feature computation service (mock implementation)
        await self._enqueue_refresh_request(refresh_request)
    
    async def get_historical_features(self, entity_df: pd.DataFrame,
                                    feature_names: List[str],
                                    timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """Retrieve historical features for training data"""
        
        # Validate feature definitions
        feature_definitions = []
        for feature_name in feature_names:
            feature_def = await self.feature_registry.get_feature_definition(feature_name)
            if feature_def:
                feature_definitions.append(feature_def)
        
        # Point-in-time correct feature retrieval
        historical_features = await self.offline_store.get_historical_features(
            entity_df, feature_definitions, timestamp_column
        )
        
        # Ensure training-serving consistency
        consistency_report = await self.consistency_manager.validate_training_serving_consistency(
            historical_features, feature_definitions
        )
        
        # Record metrics
        await self.monitoring_system.record_historical_serving_metrics({
            'entity_count': len(entity_df),
            'feature_count': len(feature_definitions),
            'time_range_days': self._calculate_time_range_days(entity_df, timestamp_column),
            'consistency_score': consistency_report['consistency_score']
        })
        
        return historical_features

class OnlineFeatureStore(ABC):
    """Abstract base class for online feature stores"""
    
    @abstractmethod
    async def get_feature_vector(self, entity_id: str,
                               feature_definitions: List[FeatureDefinition]) -> Optional[FeatureVector]:
        pass
    
    @abstractmethod
    async def batch_get_feature_vectors(self, entity_ids: List[str],
                                      feature_definitions: List[FeatureDefinition]) -> Dict[str, FeatureVector]:
        pass
    
    @abstractmethod
    async def put_feature_vector(self, entity_id: str, feature_vector: FeatureVector):
        pass

class RedisOnlineStore(OnlineFeatureStore):
    """Redis-based online feature store implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.redis_client = redis.Redis(
            host=config.get('host', 'localhost'),
            port=config.get('port', 6379),
            db=config.get('db', 0),
            decode_responses=True
        )
        self.key_prefix = config.get('key_prefix', 'features')
        self.ttl_seconds = config.get('ttl_seconds', 3600)
    
    async def get_feature_vector(self, entity_id: str,
                               feature_definitions: List[FeatureDefinition]) -> Optional[FeatureVector]:
        """Retrieve feature vector from Redis"""
        
        key = f"{self.key_prefix}:{entity_id}"
        
        try:
            # Get feature data from Redis
            feature_data = self.redis_client.hgetall(key)
            
            if not feature_data:
                return None
            
            # Parse feature data
            features = {}
            for feature_def in feature_definitions:
                if feature_def.feature_name in feature_data:
                    raw_value = feature_data[feature_def.feature_name]
                    features[feature_def.feature_name] = self._parse_feature_value(
                        raw_value, feature_def.data_type
                    )
            
            # Parse metadata
            feature_timestamp = datetime.fromisoformat(feature_data.get('_timestamp', datetime.utcnow().isoformat()))
            computation_timestamp = datetime.fromisoformat(feature_data.get('_computed_at', datetime.utcnow().isoformat()))
            
            return FeatureVector(
                entity_id=entity_id,
                features=features,
                feature_timestamp=feature_timestamp,
                computation_timestamp=computation_timestamp,
                version_info=json.loads(feature_data.get('_version_info', '{}')),
                metadata=json.loads(feature_data.get('_metadata', '{}'))
            )
            
        except Exception as e:
            print(f"Error retrieving features for entity {entity_id}: {e}")
            return None
    
    async def batch_get_feature_vectors(self, entity_ids: List[str],
                                      feature_definitions: List[FeatureDefinition]) -> Dict[str, FeatureVector]:
        """Batch retrieve feature vectors from Redis"""
        
        if not entity_ids:
            return {}
        
        # Use Redis pipeline for batch operations
        pipe = self.redis_client.pipeline()
        
        # Queue all HGETALL operations  
        keys = [f"{self.key_prefix}:{entity_id}" for entity_id in entity_ids]
        for key in keys:
            pipe.hgetall(key)
        
        # Execute pipeline
        results = pipe.execute()
        
        # Process results
        feature_vectors = {}
        for i, (entity_id, feature_data) in enumerate(zip(entity_ids, results)):
            if feature_data:
                try:
                    # Parse features
                    features = {}
                    for feature_def in feature_definitions:
                        if feature_def.feature_name in feature_data:
                            raw_value = feature_data[feature_def.feature_name]
                            features[feature_def.feature_name] = self._parse_feature_value(
                                raw_value, feature_def.data_type
                            )
                    
                    if features:  # Only create vector if we have features
                        feature_timestamp = datetime.fromisoformat(
                            feature_data.get('_timestamp', datetime.utcnow().isoformat())
                        )
                        computation_timestamp = datetime.fromisoformat(
                            feature_data.get('_computed_at', datetime.utcnow().isoformat())
                        )
                        
                        feature_vectors[entity_id] = FeatureVector(
                            entity_id=entity_id,
                            features=features,
                            feature_timestamp=feature_timestamp,
                            computation_timestamp=computation_timestamp,
                            version_info=json.loads(feature_data.get('_version_info', '{}')),
                            metadata=json.loads(feature_data.get('_metadata', '{}'))
                        )
                        
                except Exception as e:
                    print(f"Error parsing features for entity {entity_id}: {e}")
        
        return feature_vectors
    
    async def put_feature_vector(self, entity_id: str, feature_vector: FeatureVector):
        """Store feature vector in Redis"""
        
        key = f"{self.key_prefix}:{entity_id}"
        
        # Prepare feature data for storage
        feature_data = {}
        
        # Add feature values
        for feature_name, feature_value in feature_vector.features.items():
            feature_data[feature_name] = self._serialize_feature_value(feature_value)
        
        # Add metadata
        feature_data['_timestamp'] = feature_vector.feature_timestamp.isoformat()
        feature_data['_computed_at'] = feature_vector.computation_timestamp.isoformat()
        feature_data['_version_info'] = json.dumps(feature_vector.version_info)
        feature_data['_metadata'] = json.dumps(feature_vector.metadata)
        
        # Store in Redis with TTL
        pipe = self.redis_client.pipeline()
        pipe.hmset(key, feature_data)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()
    
    def _parse_feature_value(self, raw_value: str, data_type: str) -> Any:
        """Parse feature value based on data type"""
        
        if data_type == 'int64':
            return int(raw_value)
        elif data_type == 'float64':
            return float(raw_value)
        elif data_type == 'string':
            return raw_value
        elif data_type == 'array':
            return json.loads(raw_value)
        else:
            return raw_value
    
    def _serialize_feature_value(self, feature_value: Any) -> str:
        """Serialize feature value for storage"""
        
        if isinstance(feature_value, (list, dict)):
            return json.dumps(feature_value)
        else:
            return str(feature_value)

class FeatureRegistry:
    """Feature definition registry with versioning support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_definitions = {}  # feature_name -> FeatureDefinition
        self.feature_versions = {}     # feature_name -> List[version]
        
    async def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature definition"""
        
        try:
            # Validate feature definition
            validation_result = self._validate_feature_definition(feature_def)
            if not validation_result['valid']:
                raise ValueError(f"Invalid feature definition: {validation_result['errors']}")
            
            # Check for version conflicts
            existing_def = self.feature_definitions.get(feature_def.feature_name)
            if existing_def and existing_def.version == feature_def.version:
                raise ValueError(f"Feature {feature_def.feature_name} version {feature_def.version} already exists")
            
            # Register feature
            self.feature_definitions[feature_def.feature_name] = feature_def
            
            # Track versions
            if feature_def.feature_name not in self.feature_versions:
                self.feature_versions[feature_def.feature_name] = []
            self.feature_versions[feature_def.feature_name].append(feature_def.version)
            
            return True
            
        except Exception as e:
            print(f"Error registering feature {feature_def.feature_name}: {e}")
            return False
    
    async def get_feature_definition(self, feature_name: str,
                                   version: Optional[str] = None) -> Optional[FeatureDefinition]:
        """Get feature definition by name and optional version"""
        
        if version is None:
            # Return latest version
            return self.feature_definitions.get(feature_name)
        else:
            # Return specific version (simplified - would need version storage)
            feature_def = self.feature_definitions.get(feature_name)
            if feature_def and feature_def.version == version:
                return feature_def
            return None
    
    def _validate_feature_definition(self, feature_def: FeatureDefinition) -> Dict[str, Any]:
        """Validate feature definition"""
        
        errors = []
        
        # Required field validation
        if not feature_def.feature_name:
            errors.append("Feature name is required")
        
        if not feature_def.feature_type:
            errors.append("Feature type is required")
        
        if not feature_def.data_type:
            errors.append("Data type is required")
        
        # SLA validation
        if feature_def.freshness_sla_minutes <= 0:
            errors.append("Freshness SLA must be positive")
        
        if not (0 <= feature_def.availability_sla <= 1):
            errors.append("Availability SLA must be between 0 and 1")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

class FeatureConsistencyManager:
    """Manage feature consistency across online and offline stores"""
    
    def __init__(self):
        self.write_log = {}  # Track recent writes for consistency
        self.consistency_cache = {}
        
    async def get_recent_writes(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get recent writes for an entity"""
        return self.write_log.get(entity_id, [])
    
    async def ensure_write_propagation(self, entity_id: str, recent_writes: List[Dict[str, Any]]):
        """Ensure write propagation for strong consistency"""
        # Mock implementation - would check replication status
        await asyncio.sleep(0.01)  # Simulate propagation delay
    
    async def validate_training_serving_consistency(self, historical_features: pd.DataFrame,
                                                  feature_definitions: List[FeatureDefinition]) -> Dict[str, Any]:
        """Validate consistency between training and serving features"""
        
        # Mock consistency validation
        consistency_score = 0.95  # Would compute actual consistency metrics
        
        return {
            'consistency_score': consistency_score,
            'inconsistent_features': [],
            'recommendation': 'Features are consistent for training-serving'
        }
```

This completes Part 3 of Day 4, covering advanced feature store architecture, implementation patterns, and consistency management systems.