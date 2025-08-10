# Day 4.4: Consistency Models & Feature Versioning

## 🔄 Storage Layers & Feature Store Deep Dive - Part 4

**Focus**: Online/Offline Store Consistency, Feature Versioning Strategies, Freshness SLA Management  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master consistency models for distributed feature stores (eventual, strong, causal)
- Understand feature versioning strategies and backward compatibility management
- Learn freshness SLA enforcement and automatic feature refresh mechanisms
- Implement training-serving skew detection and mitigation strategies

---

## 🔄 Distributed Consistency Theory for Feature Stores

### **CAP Theorem Applied to Feature Stores**

#### **Consistency-Availability-Partition Tolerance Trade-offs**
```
Feature Store CAP Analysis:

Consistency (C): All nodes see the same feature values simultaneously
Availability (A): System remains operational for feature serving
Partition Tolerance (P): System continues despite network failures

Possible Combinations:
- CP System: Strong consistency + Partition tolerance → Reduced availability
- AP System: High availability + Partition tolerance → Eventual consistency  
- CA System: Consistency + Availability → No partition tolerance (impractical)

Feature Store Reality: Most choose AP (High availability + Eventual consistency)
```

```python
import asyncio
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid

class ConsistencyModel(Enum):
    """Consistency models for distributed feature stores"""
    STRONG = "strong"              # Linearizability - all operations appear instantaneous
    SEQUENTIAL = "sequential"      # Operations appear in some sequential order
    CAUSAL = "causal"             # Causally related operations are ordered
    EVENTUAL = "eventual"         # System will eventually converge
    WEAK = "weak"                 # Minimal consistency guarantees

class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts in feature values"""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    TIMESTAMP_ORDERING = "timestamp_ordering"
    VERSION_VECTOR = "version_vector"
    CUSTOM_RESOLVER = "custom_resolver"

@dataclass
class FeatureVersion:
    """Represents a versioned feature with full metadata"""
    feature_name: str
    version: str
    value: Any
    timestamp: datetime
    vector_clock: Dict[str, int] = field(default_factory=dict)
    causality_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.feature_name, self.version, self.timestamp.isoformat()))

@dataclass
class ConsistencyGuarantee:
    """Defines consistency guarantees for feature operations"""
    model: ConsistencyModel
    read_consistency: str           # strong, eventual, session
    write_consistency: str          # strong, eventual
    isolation_level: str           # read_uncommitted, read_committed, repeatable_read, serializable
    conflict_resolution: ConflictResolutionStrategy
    max_staleness_seconds: Optional[int] = None
    causal_dependencies: bool = False

class DistributedConsistencyManager:
    """Manage consistency across distributed feature store nodes"""
    
    def __init__(self, node_id: str, consistency_config: Dict[str, Any]):
        self.node_id = node_id
        self.config = consistency_config
        self.vector_clock = VectorClock(node_id)
        self.causal_context = CausalContext()
        self.conflict_resolver = ConflictResolver()
        self.consistency_monitor = ConsistencyMonitor()
        
        # Node state
        self.node_states = {}  # node_id -> last_known_state
        self.pending_operations = {}
        self.committed_operations = {}
        
    async def write_feature_with_consistency(self, feature_name: str, value: Any,
                                           consistency_guarantee: ConsistencyGuarantee,
                                           causal_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write feature value with specified consistency guarantees"""
        
        write_operation = {
            'operation_id': str(uuid.uuid4()),
            'feature_name': feature_name,
            'value': value,
            'timestamp': datetime.utcnow(),
            'node_id': self.node_id,
            'consistency_model': consistency_guarantee.model.value,
            'causal_context': causal_context or {}
        }
        
        if consistency_guarantee.model == ConsistencyModel.STRONG:
            return await self._write_with_strong_consistency(write_operation, consistency_guarantee)
        elif consistency_guarantee.model == ConsistencyModel.CAUSAL:
            return await self._write_with_causal_consistency(write_operation, consistency_guarantee)
        elif consistency_guarantee.model == ConsistencyModel.EVENTUAL:
            return await self._write_with_eventual_consistency(write_operation, consistency_guarantee)
        else:
            return await self._write_with_weak_consistency(write_operation, consistency_guarantee)
    
    async def _write_with_strong_consistency(self, operation: Dict[str, Any],
                                           guarantee: ConsistencyGuarantee) -> Dict[str, Any]:
        """Implement strong consistency using consensus protocol"""
        
        # Phase 1: Prepare phase (2PC-like protocol)
        prepare_responses = await self._broadcast_prepare(operation)
        
        if not self._all_nodes_prepared(prepare_responses):
            # Abort operation
            await self._broadcast_abort(operation['operation_id'])
            return {
                'success': False,
                'error': 'Not all nodes could prepare for operation',
                'consistency_level': 'strong'
            }
        
        # Phase 2: Commit phase
        commit_responses = await self._broadcast_commit(operation)
        
        if self._majority_committed(commit_responses):
            # Operation succeeded with strong consistency
            self._apply_local_operation(operation)
            
            return {
                'success': True,
                'operation_id': operation['operation_id'],
                'consistency_level': 'strong',
                'commit_timestamp': datetime.utcnow().isoformat(),
                'participating_nodes': len(commit_responses)
            }
        else:
            # Partial failure - attempt recovery
            await self._initiate_recovery(operation['operation_id'])
            return {
                'success': False,
                'error': 'Partial commit failure, recovery initiated',
                'consistency_level': 'strong'
            }
    
    async def _write_with_causal_consistency(self, operation: Dict[str, Any],
                                           guarantee: ConsistencyGuarantee) -> Dict[str, Any]:
        """Implement causal consistency using vector clocks"""
        
        # Update vector clock
        self.vector_clock.increment()
        operation['vector_clock'] = self.vector_clock.get_clock()
        
        # Check causal dependencies
        causal_dependencies = operation.get('causal_context', {}).get('depends_on', [])
        
        if causal_dependencies:
            # Wait for causal dependencies to be satisfied
            await self._wait_for_causal_dependencies(causal_dependencies)
        
        # Apply operation locally
        self._apply_local_operation(operation)
        
        # Propagate to other nodes asynchronously
        asyncio.create_task(self._propagate_causal_operation(operation))
        
        return {
            'success': True,
            'operation_id': operation['operation_id'],
            'consistency_level': 'causal',
            'vector_clock': operation['vector_clock'],
            'causal_context': self.causal_context.get_context()
        }
    
    async def _write_with_eventual_consistency(self, operation: Dict[str, Any],
                                             guarantee: ConsistencyGuarantee) -> Dict[str, Any]:
        """Implement eventual consistency with conflict resolution"""
        
        # Apply operation locally immediately
        self._apply_local_operation(operation)
        
        # Add conflict resolution metadata
        operation['conflict_resolution'] = guarantee.conflict_resolution.value
        operation['lamport_timestamp'] = self._get_lamport_timestamp()
        
        # Propagate asynchronously to other nodes
        asyncio.create_task(self._propagate_eventual_operation(operation))
        
        return {
            'success': True,
            'operation_id': operation['operation_id'],
            'consistency_level': 'eventual',
            'local_timestamp': operation['timestamp'].isoformat(),
            'propagation_initiated': True
        }
    
    async def read_feature_with_consistency(self, feature_name: str,
                                          consistency_guarantee: ConsistencyGuarantee,
                                          read_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Read feature value with specified consistency guarantees"""
        
        read_operation = {
            'operation_id': str(uuid.uuid4()),
            'feature_name': feature_name,
            'timestamp': datetime.utcnow(),
            'node_id': self.node_id,
            'consistency_model': consistency_guarantee.model.value,
            'read_context': read_context or {}
        }
        
        if consistency_guarantee.model == ConsistencyModel.STRONG:
            return await self._read_with_strong_consistency(read_operation, consistency_guarantee)
        elif consistency_guarantee.model == ConsistencyModel.CAUSAL:
            return await self._read_with_causal_consistency(read_operation, consistency_guarantee)
        elif consistency_guarantee.model == ConsistencyModel.EVENTUAL:
            return await self._read_with_eventual_consistency(read_operation, consistency_guarantee)
        else:
            return await self._read_with_weak_consistency(read_operation, consistency_guarantee)
    
    async def _read_with_strong_consistency(self, operation: Dict[str, Any],
                                          guarantee: ConsistencyGuarantee) -> Dict[str, Any]:
        """Read with strong consistency - may require coordination"""
        
        # For strong consistency reads, we need to ensure we're reading the latest committed value
        # This may require querying multiple nodes
        
        node_responses = await self._query_all_nodes_for_feature(operation['feature_name'])
        
        # Find the most recent committed value
        latest_value = self._find_latest_committed_value(node_responses)
        
        if latest_value is None:
            return {
                'success': False,
                'error': 'Feature not found or no committed value available',
                'consistency_level': 'strong'
            }
        
        return {
            'success': True,
            'feature_name': operation['feature_name'],
            'value': latest_value['value'],
            'timestamp': latest_value['timestamp'],
            'consistency_level': 'strong',
            'read_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _read_with_causal_consistency(self, operation: Dict[str, Any],
                                          guarantee: ConsistencyGuarantee) -> Dict[str, Any]:
        """Read with causal consistency - respect causal ordering"""
        
        # Check if local replica has all required causal dependencies
        required_context = operation['read_context'].get('causal_context', {})
        
        if not self._causal_dependencies_satisfied(required_context):
            # Wait for causal dependencies or read from another node
            await self._wait_for_causal_context(required_context)
        
        # Read local value
        local_value = self._get_local_feature_value(operation['feature_name'])
        
        if local_value is None:
            return {
                'success': False,
                'error': 'Feature not found',
                'consistency_level': 'causal'
            }
        
        return {
            'success': True,
            'feature_name': operation['feature_name'],
            'value': local_value['value'],
            'timestamp': local_value['timestamp'],
            'consistency_level': 'causal',
            'causal_context': self.causal_context.get_context()
        }

class VectorClock:
    """Vector clock implementation for causal consistency"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock = {node_id: 0}
        self.lock = threading.Lock()
    
    def increment(self):
        """Increment local clock"""
        with self.lock:
            self.clock[self.node_id] += 1
    
    def update(self, other_clock: Dict[str, int]):
        """Update clock with information from another node"""
        with self.lock:
            for node_id, timestamp in other_clock.items():
                if node_id == self.node_id:
                    continue
                self.clock[node_id] = max(self.clock.get(node_id, 0), timestamp)
            
            # Increment own clock
            self.clock[self.node_id] += 1
    
    def get_clock(self) -> Dict[str, int]:
        """Get current clock state"""
        with self.lock:
            return self.clock.copy()
    
    def happens_before(self, other_clock: Dict[str, int]) -> bool:
        """Check if this clock happens before another clock"""
        with self.lock:
            # self < other if self[i] <= other[i] for all i, and self[j] < other[j] for some j
            all_less_equal = True
            some_strictly_less = False
            
            all_nodes = set(self.clock.keys()) | set(other_clock.keys())
            
            for node in all_nodes:
                self_val = self.clock.get(node, 0)
                other_val = other_clock.get(node, 0)
                
                if self_val > other_val:
                    all_less_equal = False
                    break
                elif self_val < other_val:
                    some_strictly_less = True
            
            return all_less_equal and some_strictly_less
    
    def concurrent_with(self, other_clock: Dict[str, int]) -> bool:
        """Check if this clock is concurrent with another clock"""
        return not (self.happens_before(other_clock) or 
                   VectorClock.static_happens_before(other_clock, self.clock))
    
    @staticmethod
    def static_happens_before(clock1: Dict[str, int], clock2: Dict[str, int]) -> bool:
        """Static method to check happens-before relationship"""
        all_nodes = set(clock1.keys()) | set(clock2.keys())
        all_less_equal = True
        some_strictly_less = False
        
        for node in all_nodes:
            val1 = clock1.get(node, 0)
            val2 = clock2.get(node, 0)
            
            if val1 > val2:
                all_less_equal = False
                break
            elif val1 < val2:
                some_strictly_less = True
        
        return all_less_equal and some_strictly_less

class FeatureVersioningManager:
    """Manage feature versioning and schema evolution"""
    
    def __init__(self):
        self.version_history = {}  # feature_name -> List[FeatureVersion]
        self.active_versions = {}  # feature_name -> active_version
        self.schema_compatibility_checker = SchemaCompatibilityChecker()
        self.version_migration_manager = VersionMigrationManager()
        
    def create_feature_version(self, feature_name: str, value: Any,
                             schema_info: Dict[str, Any],
                             compatibility_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new version of a feature with compatibility validation"""
        
        current_version = self.active_versions.get(feature_name)
        new_version_number = self._generate_next_version(feature_name, compatibility_requirements)
        
        # Create new feature version
        new_version = FeatureVersion(
            feature_name=feature_name,
            version=new_version_number,
            value=value,
            timestamp=datetime.utcnow(),
            metadata={
                'schema_info': schema_info,
                'compatibility_requirements': compatibility_requirements,
                'created_by': 'system',
                'migration_strategy': compatibility_requirements.get('migration_strategy', 'backward_compatible')
            }
        )
        
        # Validate compatibility with existing versions
        if current_version:
            compatibility_result = self.schema_compatibility_checker.check_compatibility(
                current_version, new_version, compatibility_requirements
            )
            
            if not compatibility_result['compatible']:
                return {
                    'success': False,
                    'error': 'Schema compatibility validation failed',
                    'compatibility_issues': compatibility_result['issues'],
                    'suggested_actions': compatibility_result['suggestions']
                }
        
        # Add to version history
        if feature_name not in self.version_history:
            self.version_history[feature_name] = []
        self.version_history[feature_name].append(new_version)
        
        # Update active version
        self.active_versions[feature_name] = new_version
        
        # Plan migration if needed
        migration_plan = None
        if current_version and compatibility_requirements.get('migration_strategy') != 'immediate':
            migration_plan = self.version_migration_manager.create_migration_plan(
                current_version, new_version, compatibility_requirements
            )
        
        return {
            'success': True,
            'new_version': new_version_number,
            'previous_version': current_version.version if current_version else None,
            'compatibility_validated': True,
            'migration_plan': migration_plan,
            'activation_timestamp': new_version.timestamp.isoformat()
        }
    
    def _generate_next_version(self, feature_name: str, 
                             compatibility_requirements: Dict[str, Any]) -> str:
        """Generate next version number based on compatibility requirements"""
        
        current_version = self.active_versions.get(feature_name)
        
        if not current_version:
            return "1.0.0"
        
        # Parse current version (assuming semantic versioning)
        version_parts = current_version.version.split('.')
        major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
        
        # Determine version increment based on compatibility
        change_type = compatibility_requirements.get('change_type', 'patch')
        
        if change_type == 'breaking':
            major += 1
            minor = 0
            patch = 0
        elif change_type == 'feature':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def get_feature_version(self, feature_name: str, 
                           version: Optional[str] = None,
                           timestamp: Optional[datetime] = None) -> Optional[FeatureVersion]:
        """Retrieve specific version of a feature"""
        
        if feature_name not in self.version_history:
            return None
        
        versions = self.version_history[feature_name]
        
        if version:
            # Get specific version
            for v in versions:
                if v.version == version:
                    return v
            return None
        
        elif timestamp:
            # Get version active at specific timestamp (point-in-time query)
            valid_versions = [v for v in versions if v.timestamp <= timestamp]
            if valid_versions:
                return max(valid_versions, key=lambda x: x.timestamp)
            return None
        
        else:
            # Get latest version
            return self.active_versions.get(feature_name)
    
    def list_feature_versions(self, feature_name: str) -> List[Dict[str, Any]]:
        """List all versions of a feature with metadata"""
        
        if feature_name not in self.version_history:
            return []
        
        versions = self.version_history[feature_name]
        
        return [
            {
                'version': v.version,
                'timestamp': v.timestamp.isoformat(),
                'schema_info': v.metadata.get('schema_info', {}),
                'compatibility_info': v.metadata.get('compatibility_requirements', {}),
                'is_active': v == self.active_versions.get(feature_name)
            }
            for v in sorted(versions, key=lambda x: x.timestamp, reverse=True)
        ]

class SchemaCompatibilityChecker:
    """Check schema compatibility between feature versions"""
    
    def check_compatibility(self, old_version: FeatureVersion, 
                          new_version: FeatureVersion,
                          requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Check compatibility between two feature versions"""
        
        compatibility_result = {
            'compatible': True,
            'compatibility_level': 'full',
            'issues': [],
            'suggestions': []
        }
        
        old_schema = old_version.metadata.get('schema_info', {})
        new_schema = new_version.metadata.get('schema_info', {})
        
        # Check data type compatibility
        type_compatibility = self._check_type_compatibility(old_schema, new_schema)
        if not type_compatibility['compatible']:
            compatibility_result['compatible'] = False
            compatibility_result['issues'].extend(type_compatibility['issues'])
            compatibility_result['suggestions'].extend(type_compatibility['suggestions'])
        
        # Check value range compatibility
        range_compatibility = self._check_value_range_compatibility(old_schema, new_schema)
        if not range_compatibility['compatible']:
            compatibility_result['compatible'] = False
            compatibility_result['issues'].extend(range_compatibility['issues'])
            compatibility_result['suggestions'].extend(range_compatibility['suggestions'])
        
        # Check semantic compatibility
        semantic_compatibility = self._check_semantic_compatibility(old_version, new_version)
        if not semantic_compatibility['compatible']:
            compatibility_result['compatibility_level'] = 'partial'
            compatibility_result['issues'].extend(semantic_compatibility['issues'])
            compatibility_result['suggestions'].extend(semantic_compatibility['suggestions'])
        
        return compatibility_result
    
    def _check_type_compatibility(self, old_schema: Dict[str, Any], 
                                new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data types are compatible"""
        
        old_type = old_schema.get('data_type', 'unknown')
        new_type = new_schema.get('data_type', 'unknown')
        
        # Define type compatibility matrix
        compatible_types = {
            'int32': ['int32', 'int64', 'float32', 'float64'],
            'int64': ['int64', 'float64'],
            'float32': ['float32', 'float64'],
            'float64': ['float64'],
            'string': ['string'],
            'array': ['array'],
            'object': ['object']
        }
        
        if new_type in compatible_types.get(old_type, []):
            return {'compatible': True, 'issues': [], 'suggestions': []}
        else:
            return {
                'compatible': False,
                'issues': [f'Incompatible type change: {old_type} -> {new_type}'],
                'suggestions': [f'Consider using a type converter or creating a new feature']
            }
    
    def _check_value_range_compatibility(self, old_schema: Dict[str, Any],
                                       new_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check if value ranges are compatible"""
        
        old_range = old_schema.get('value_range', {})
        new_range = new_schema.get('value_range', {})
        
        if not old_range or not new_range:
            return {'compatible': True, 'issues': [], 'suggestions': []}
        
        # Check if new range is within old range (backward compatible)
        old_min = old_range.get('min', float('-inf'))
        old_max = old_range.get('max', float('inf'))
        new_min = new_range.get('min', float('-inf'))
        new_max = new_range.get('max', float('inf'))
        
        if new_min >= old_min and new_max <= old_max:
            return {'compatible': True, 'issues': [], 'suggestions': []}
        else:
            return {
                'compatible': False,
                'issues': [f'Value range expansion: [{old_min}, {old_max}] -> [{new_min}, {new_max}]'],
                'suggestions': ['Consider gradual range expansion or separate feature for extended range']
            }

class FreshnessSLAManager:
    """Manage feature freshness SLAs and automatic refresh"""
    
    def __init__(self):
        self.sla_definitions = {}  # feature_name -> SLA config
        self.freshness_monitor = FreshnessMonitor()
        self.refresh_scheduler = RefreshScheduler()
        self.violation_handler = SLAViolationHandler()
        
    def define_freshness_sla(self, feature_name: str, sla_config: Dict[str, Any]) -> bool:
        """Define freshness SLA for a feature"""
        
        try:
            # Validate SLA configuration
            required_fields = ['max_staleness_minutes', 'target_refresh_interval_minutes']
            for field in required_fields:
                if field not in sla_config:
                    raise ValueError(f"Missing required SLA field: {field}")
            
            # Additional validation
            if sla_config['max_staleness_minutes'] <= 0:
                raise ValueError("Max staleness must be positive")
            
            if sla_config['target_refresh_interval_minutes'] <= 0:
                raise ValueError("Target refresh interval must be positive")
            
            # Store SLA definition
            self.sla_definitions[feature_name] = {
                **sla_config,
                'created_at': datetime.utcnow(),
                'status': 'active'
            }
            
            # Schedule monitoring
            self.freshness_monitor.start_monitoring(feature_name, sla_config)
            
            return True
            
        except Exception as e:
            print(f"Error defining SLA for feature {feature_name}: {e}")
            return False
    
    async def check_feature_freshness(self, feature_name: str, 
                                    current_timestamp: datetime) -> Dict[str, Any]:
        """Check if feature meets freshness SLA"""
        
        if feature_name not in self.sla_definitions:
            return {
                'has_sla': False,
                'error': 'No SLA defined for feature'
            }
        
        sla_config = self.sla_definitions[feature_name]
        
        # Get latest feature timestamp
        latest_feature_timestamp = await self._get_latest_feature_timestamp(feature_name)
        
        if not latest_feature_timestamp:
            return {
                'has_sla': True,
                'fresh': False,
                'violation_type': 'missing_feature',
                'staleness_minutes': float('inf')
            }
        
        # Calculate staleness
        staleness = current_timestamp - latest_feature_timestamp
        staleness_minutes = staleness.total_seconds() / 60
        
        max_staleness_minutes = sla_config['max_staleness_minutes']
        
        freshness_result = {
            'has_sla': True,
            'fresh': staleness_minutes <= max_staleness_minutes,
            'staleness_minutes': staleness_minutes,
            'max_allowed_staleness_minutes': max_staleness_minutes,
            'sla_compliance_ratio': max(0, 1 - (staleness_minutes / max_staleness_minutes)),
            'latest_feature_timestamp': latest_feature_timestamp.isoformat()
        }
        
        if not freshness_result['fresh']:
            freshness_result['violation_type'] = 'staleness_exceeded'
            freshness_result['violation_severity'] = self._calculate_violation_severity(
                staleness_minutes, max_staleness_minutes
            )
            
            # Trigger refresh if configured
            if sla_config.get('auto_refresh', True):
                refresh_request = await self._trigger_feature_refresh(feature_name, 'sla_violation')
                freshness_result['refresh_triggered'] = refresh_request['success']
        
        return freshness_result
    
    def _calculate_violation_severity(self, actual_staleness: float, 
                                    max_allowed_staleness: float) -> str:
        """Calculate severity of SLA violation"""
        
        violation_ratio = actual_staleness / max_allowed_staleness
        
        if violation_ratio <= 1.2:  # Up to 20% over SLA
            return 'minor'
        elif violation_ratio <= 2.0:  # Up to 100% over SLA
            return 'major'
        else:  # More than 100% over SLA
            return 'critical'
    
    async def _get_latest_feature_timestamp(self, feature_name: str) -> Optional[datetime]:
        """Get timestamp of latest feature value"""
        # Mock implementation - would query actual feature store
        return datetime.utcnow() - timedelta(minutes=30)  # Simulate 30-minute old feature
    
    async def _trigger_feature_refresh(self, feature_name: str, 
                                     trigger_reason: str) -> Dict[str, Any]:
        """Trigger feature refresh due to SLA violation"""
        
        refresh_request = {
            'feature_name': feature_name,
            'trigger_reason': trigger_reason,
            'priority': 'high' if trigger_reason == 'sla_violation' else 'normal',
            'requested_at': datetime.utcnow(),
            'request_id': str(uuid.uuid4())
        }
        
        # Send to refresh scheduler
        return await self.refresh_scheduler.schedule_refresh(refresh_request)

class TrainingServingSkewDetector:
    """Detect and mitigate training-serving skew"""
    
    def __init__(self):
        self.baseline_distributions = {}  # feature_name -> statistical distribution
        self.skew_thresholds = {}
        self.mitigation_strategies = {}
        
    async def detect_skew(self, feature_name: str,
                         training_values: List[Any],
                         serving_values: List[Any]) -> Dict[str, Any]:
        """Detect training-serving skew for a feature"""
        
        skew_analysis = {
            'feature_name': feature_name,
            'skew_detected': False,
            'skew_metrics': {},
            'severity': 'none',
            'recommended_actions': []
        }
        
        # Calculate statistical measures
        training_stats = self._calculate_statistics(training_values)
        serving_stats = self._calculate_statistics(serving_values)
        
        # Distribution comparison
        distribution_skew = self._calculate_distribution_skew(training_stats, serving_stats)
        skew_analysis['skew_metrics']['distribution_skew'] = distribution_skew
        
        # Mean shift detection
        mean_shift = abs(training_stats['mean'] - serving_stats['mean'])
        relative_mean_shift = mean_shift / max(abs(training_stats['mean']), abs(serving_stats['mean']), 1e-6)
        skew_analysis['skew_metrics']['mean_shift'] = {
            'absolute': mean_shift,
            'relative': relative_mean_shift
        }
        
        # Variance change detection
        variance_ratio = serving_stats['variance'] / max(training_stats['variance'], 1e-6)
        skew_analysis['skew_metrics']['variance_ratio'] = variance_ratio
        
        # Overall skew assessment
        skew_threshold = self.skew_thresholds.get(feature_name, {
            'distribution_threshold': 0.1,
            'mean_shift_threshold': 0.2,
            'variance_ratio_threshold': 2.0
        })
        
        skew_indicators = []
        
        if distribution_skew > skew_threshold['distribution_threshold']:
            skew_indicators.append('distribution_drift')
        
        if relative_mean_shift > skew_threshold['mean_shift_threshold']:
            skew_indicators.append('mean_shift')
        
        if variance_ratio > skew_threshold['variance_ratio_threshold'] or variance_ratio < (1/skew_threshold['variance_ratio_threshold']):
            skew_indicators.append('variance_change')
        
        if skew_indicators:
            skew_analysis['skew_detected'] = True
            skew_analysis['skew_indicators'] = skew_indicators
            skew_analysis['severity'] = self._assess_skew_severity(skew_analysis['skew_metrics'])
            skew_analysis['recommended_actions'] = self._generate_mitigation_recommendations(
                feature_name, skew_indicators, skew_analysis['severity']
            )
        
        return skew_analysis
    
    def _calculate_statistics(self, values: List[Any]) -> Dict[str, float]:
        """Calculate statistical measures for feature values"""
        
        if not values:
            return {'mean': 0, 'variance': 0, 'min': 0, 'max': 0, 'count': 0}
        
        # Convert to numeric values if possible
        numeric_values = []
        for val in values:
            try:
                if isinstance(val, (int, float)):
                    numeric_values.append(float(val))
                elif isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit():
                    numeric_values.append(float(val))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return {'mean': 0, 'variance': 0, 'min': 0, 'max': 0, 'count': len(values)}
        
        mean_val = np.mean(numeric_values)
        variance_val = np.var(numeric_values)
        
        return {
            'mean': float(mean_val),
            'variance': float(variance_val),
            'min': float(np.min(numeric_values)),
            'max': float(np.max(numeric_values)),
            'count': len(numeric_values)
        }
    
    def _calculate_distribution_skew(self, training_stats: Dict[str, float],
                                   serving_stats: Dict[str, float]) -> float:
        """Calculate distribution skew using KL divergence approximation"""
        
        # Simplified KL divergence calculation using normal distribution assumption
        # In practice, would use more sophisticated distribution comparison
        
        if training_stats['variance'] == 0 or serving_stats['variance'] == 0:
            return 1.0 if training_stats['mean'] != serving_stats['mean'] else 0.0
        
        # Approximate KL divergence for normal distributions
        mean_diff = training_stats['mean'] - serving_stats['mean']
        var_ratio = serving_stats['variance'] / training_stats['variance']
        
        kl_divergence = 0.5 * (
            var_ratio + 
            (mean_diff ** 2) / training_stats['variance'] - 
            1 - 
            np.log(var_ratio)
        )
        
        return max(0, float(kl_divergence))
    
    def _assess_skew_severity(self, skew_metrics: Dict[str, Any]) -> str:
        """Assess overall severity of detected skew"""
        
        severity_score = 0
        
        # Distribution skew contribution
        dist_skew = skew_metrics.get('distribution_skew', 0)
        if dist_skew > 0.5:
            severity_score += 3
        elif dist_skew > 0.2:
            severity_score += 2
        elif dist_skew > 0.1:
            severity_score += 1
        
        # Mean shift contribution
        mean_shift = skew_metrics.get('mean_shift', {}).get('relative', 0)
        if mean_shift > 0.5:
            severity_score += 3
        elif mean_shift > 0.3:
            severity_score += 2
        elif mean_shift > 0.1:
            severity_score += 1
        
        # Variance ratio contribution
        var_ratio = skew_metrics.get('variance_ratio', 1.0)
        var_change = max(var_ratio, 1/var_ratio) - 1
        if var_change > 1.0:
            severity_score += 3
        elif var_change > 0.5:
            severity_score += 2
        elif var_change > 0.2:
            severity_score += 1
        
        if severity_score >= 6:
            return 'critical'
        elif severity_score >= 4:
            return 'high'
        elif severity_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_mitigation_recommendations(self, feature_name: str,
                                           skew_indicators: List[str],
                                           severity: str) -> List[str]:
        """Generate recommendations for mitigating detected skew"""
        
        recommendations = []
        
        if 'distribution_drift' in skew_indicators:
            recommendations.extend([
                'Investigate data source changes or data quality issues',
                'Consider retraining model with recent data',
                'Implement feature distribution monitoring'
            ])
        
        if 'mean_shift' in skew_indicators:
            recommendations.extend([
                'Check for systematic bias in serving data',
                'Validate feature preprocessing consistency',
                'Consider feature normalization or standardization'
            ])
        
        if 'variance_change' in skew_indicators:
            recommendations.extend([
                'Investigate changes in data collection process',
                'Check for outliers or data quality issues',
                'Consider robust feature scaling methods'
            ])
        
        if severity in ['high', 'critical']:
            recommendations.extend([
                'Implement immediate alerts for stakeholders',
                'Consider fallback to previous model version',
                'Expedite model retraining with recent data'
            ])
        
        return list(set(recommendations))  # Remove duplicates
```

This completes Part 4 of Day 4, covering advanced consistency models, feature versioning strategies, and training-serving skew detection systems.