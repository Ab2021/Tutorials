# Day 2.5: Real-Time Feature Engineering & Online Feature Libraries

## 🔧 Streaming Ingestion & Real-Time Feature Pipelines - Part 5

**Focus**: Online Feature Engineering, Stream Joins, and Feature Enrichment  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master real-time feature computation patterns and algorithms
- Understand temporal joins and feature enrichment strategies
- Learn online feature libraries (River, Kafka Streams) implementation
- Implement feature drift detection and adaptation mechanisms

---

## 🔄 Real-Time Feature Engineering Fundamentals

### **Streaming vs Batch Feature Engineering**

#### **Computational Complexity Analysis**
```python
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import time

class FeatureComputationComplexity:
    """Analyze computational complexity of different feature types"""
    
    def __init__(self):
        self.complexity_metrics = {
            'simple_aggregates': {
                'time_complexity': 'O(1)',
                'space_complexity': 'O(1)',
                'examples': ['count', 'sum', 'mean', 'last_value']
            },
            'windowed_aggregates': {
                'time_complexity': 'O(w)', # w = window size
                'space_complexity': 'O(w)',
                'examples': ['moving_average', 'windowed_sum', 'percentiles']
            },
            'temporal_joins': {
                'time_complexity': 'O(log n)', # n = reference data size
                'space_complexity': 'O(n)',
                'examples': ['lookup_enrichment', 'dimension_joins']
            },
            'complex_patterns': {
                'time_complexity': 'O(n²)', # Pattern matching
                'space_complexity': 'O(n)',
                'examples': ['sequence_detection', 'anomaly_patterns']
            }
        }
    
    def estimate_computational_cost(self, feature_spec, event_rate_per_second):
        """Estimate computational cost for feature computation"""
        feature_type = feature_spec['type']
        complexity_info = self.complexity_metrics.get(feature_type, {})
        
        # Base cost estimation
        base_operations_per_event = {
            'simple_aggregates': 1,
            'windowed_aggregates': feature_spec.get('window_size', 100),
            'temporal_joins': np.log2(feature_spec.get('reference_size', 1000)),
            'complex_patterns': feature_spec.get('pattern_complexity', 100) ** 2
        }
        
        ops_per_event = base_operations_per_event.get(feature_type, 1)
        total_ops_per_second = ops_per_event * event_rate_per_second
        
        # Memory estimation
        memory_per_event_bytes = {
            'simple_aggregates': 64,  # 8 bytes * 8 fields
            'windowed_aggregates': 64 * feature_spec.get('window_size', 100),
            'temporal_joins': 1024,  # Reference data caching
            'complex_patterns': 512 * feature_spec.get('pattern_complexity', 100)
        }
        
        memory_mb = (memory_per_event_bytes.get(feature_type, 64) * 
                    event_rate_per_second) / (1024 * 1024)
        
        return {
            'operations_per_second': total_ops_per_second,
            'memory_requirement_mb': memory_mb,
            'cpu_utilization_estimate': min(100, total_ops_per_second / 10000),
            'scalability_bottleneck': self.identify_bottleneck(feature_spec)
        }
    
    def identify_bottleneck(self, feature_spec):
        """Identify potential scalability bottlenecks"""
        feature_type = feature_spec['type']
        
        bottlenecks = {
            'simple_aggregates': 'network_io',
            'windowed_aggregates': 'memory_bandwidth',
            'temporal_joins': 'lookup_latency',
            'complex_patterns': 'cpu_computation'
        }
        
        return bottlenecks.get(feature_type, 'unknown')

class StreamingFeatureEngine:
    """Core streaming feature computation engine"""
    
    def __init__(self, config):
        self.config = config
        self.feature_computors = {}
        self.state_managers = {}
        self.performance_monitor = PerformanceMonitor()
        
    def register_feature(self, feature_name, feature_spec):
        """Register a new streaming feature"""
        computor_class = self.get_computor_class(feature_spec['type'])
        
        self.feature_computors[feature_name] = computor_class(
            feature_name, feature_spec
        )
        
        # Initialize state manager for stateful features
        if feature_spec.get('stateful', False):
            self.state_managers[feature_name] = StateManager(
                feature_name, feature_spec
            )
        
        return {
            'feature_name': feature_name,
            'computor_type': feature_spec['type'],
            'stateful': feature_spec.get('stateful', False),
            'estimated_cost': self.estimate_feature_cost(feature_spec)
        }
    
    def compute_features(self, event, timestamp):
        """Compute all registered features for an event"""
        feature_results = {}
        computation_start = time.time()
        
        for feature_name, computor in self.feature_computors.items():
            try:
                feature_start = time.time()
                
                # Get state if feature is stateful
                state = None
                if feature_name in self.state_managers:
                    state = self.state_managers[feature_name].get_state(
                        event.get('key', 'default')
                    )
                
                # Compute feature value
                feature_value = computor.compute(event, timestamp, state)
                
                # Update state if needed
                if state is not None:
                    self.state_managers[feature_name].update_state(
                        event.get('key', 'default'), 
                        feature_value, 
                        timestamp
                    )
                
                feature_end = time.time()
                feature_latency = (feature_end - feature_start) * 1000
                
                feature_results[feature_name] = {
                    'value': feature_value,
                    'computation_latency_ms': feature_latency,
                    'timestamp': timestamp
                }
                
                # Monitor performance
                self.performance_monitor.record_feature_computation(
                    feature_name, feature_latency
                )
                
            except Exception as e:
                feature_results[feature_name] = {
                    'value': None,
                    'error': str(e),
                    'timestamp': timestamp
                }
        
        total_computation_time = (time.time() - computation_start) * 1000
        
        return {
            'features': feature_results,
            'total_computation_time_ms': total_computation_time,
            'event_timestamp': timestamp,
            'processing_timestamp': time.time()
        }
```

---

## 🔗 Temporal Joins and Enrichment Patterns

### **Stream-Stream Joins**

#### **Time-Based Join Windows**
```python
class TemporalJoinProcessor:
    """Handle temporal joins between streams with time-based semantics"""
    
    def __init__(self, join_config):
        self.join_config = join_config
        self.left_stream_buffer = TimeWindowBuffer(
            join_config['left_window_size_ms']
        )
        self.right_stream_buffer = TimeWindowBuffer(
            join_config['right_window_size_ms']
        )
        self.join_results = {}
        
    def process_left_stream_event(self, event, timestamp):
        """Process event from left stream"""
        # Add to left buffer
        self.left_stream_buffer.add_event(event, timestamp)
        
        # Find matching events in right stream
        join_window_start = timestamp - self.join_config['max_time_difference_ms']
        join_window_end = timestamp + self.join_config['max_time_difference_ms']
        
        matching_right_events = self.right_stream_buffer.get_events_in_range(
            join_window_start, join_window_end
        )
        
        # Perform joins
        join_results = []
        for right_event in matching_right_events:
            if self.join_predicate_matches(event, right_event):
                joined_record = self.create_joined_record(
                    event, right_event, timestamp
                )
                join_results.append(joined_record)
        
        # Cleanup expired events
        self.cleanup_expired_events(timestamp)
        
        return {
            'join_results': join_results,
            'left_buffer_size': self.left_stream_buffer.size(),
            'right_buffer_size': self.right_stream_buffer.size()
        }
    
    def join_predicate_matches(self, left_event, right_event):
        """Check if join predicate is satisfied"""
        join_condition = self.join_config['join_condition']
        
        if join_condition['type'] == 'equality':
            left_key = left_event.get(join_condition['left_key'])
            right_key = right_event.get(join_condition['right_key'])
            return left_key == right_key
        
        elif join_condition['type'] == 'range':
            left_value = left_event.get(join_condition['left_field'])
            right_value = right_event.get(join_condition['right_field'])
            
            return (join_condition['min_difference'] <= 
                   abs(left_value - right_value) <= 
                   join_condition['max_difference'])
        
        elif join_condition['type'] == 'custom':
            # Custom predicate function
            return join_condition['predicate_func'](left_event, right_event)
        
        return False
    
    def create_joined_record(self, left_event, right_event, join_timestamp):
        """Create joined record from matching events"""
        joined_record = {
            'join_timestamp': join_timestamp,
            'left_event_timestamp': left_event.get('timestamp'),
            'right_event_timestamp': right_event.get('timestamp'),
            'time_difference_ms': abs(
                left_event.get('timestamp', 0) - 
                right_event.get('timestamp', 0)
            )
        }
        
        # Merge fields based on configuration
        merge_strategy = self.join_config.get('merge_strategy', 'prefix')
        
        if merge_strategy == 'prefix':
            # Prefix left/right fields
            for key, value in left_event.items():
                joined_record[f'left_{key}'] = value
            
            for key, value in right_event.items():
                joined_record[f'right_{key}'] = value
        
        elif merge_strategy == 'left_priority':
            # Left stream takes priority for conflicts
            joined_record.update(right_event)
            joined_record.update(left_event)
        
        elif merge_strategy == 'custom':
            # Custom merge function
            merge_func = self.join_config['merge_function']
            joined_record.update(merge_func(left_event, right_event))
        
        return joined_record

class TimeWindowBuffer:
    """Time-based circular buffer for join operations"""
    
    def __init__(self, window_size_ms):
        self.window_size = window_size_ms
        self.events = deque()  # (timestamp, event) tuples
        self.index_by_key = {}  # key -> [(timestamp, event)]
        
    def add_event(self, event, timestamp):
        """Add event to time window buffer"""
        self.events.append((timestamp, event))
        
        # Index by join key if specified
        join_key = event.get('join_key')
        if join_key is not None:
            if join_key not in self.index_by_key:
                self.index_by_key[join_key] = []
            self.index_by_key[join_key].append((timestamp, event))
        
        # Cleanup old events
        self.cleanup_expired_events(timestamp)
    
    def get_events_in_range(self, start_time, end_time):
        """Get events within time range"""
        matching_events = []
        
        for timestamp, event in self.events:
            if start_time <= timestamp <= end_time:
                matching_events.append(event)
        
        return matching_events
    
    def get_events_by_key(self, key, start_time, end_time):
        """Get events by join key within time range"""
        if key not in self.index_by_key:
            return []
        
        matching_events = []
        
        for timestamp, event in self.index_by_key[key]:
            if start_time <= timestamp <= end_time:
                matching_events.append(event)
        
        return matching_events
    
    def cleanup_expired_events(self, current_timestamp):
        """Remove events outside the time window"""
        cutoff_time = current_timestamp - self.window_size
        
        # Remove from main buffer
        while self.events and self.events[0][0] < cutoff_time:
            self.events.popleft()
        
        # Remove from indexes
        for key in list(self.index_by_key.keys()):
            self.index_by_key[key] = [
                (ts, event) for ts, event in self.index_by_key[key]
                if ts >= cutoff_time
            ]
            
            # Remove empty indexes
            if not self.index_by_key[key]:
                del self.index_by_key[key]
    
    def size(self):
        """Get current buffer size"""
        return len(self.events)
```

### **Stream-Table Joins (Enrichment)**

#### **Dimension Table Enrichment**
```python
class DimensionTableEnricher:
    """Enrich streaming events with dimension table data"""
    
    def __init__(self, dimension_config):
        self.dimension_config = dimension_config
        self.dimension_cache = LRUCache(
            max_size=dimension_config.get('cache_size', 10000)
        )
        self.cache_stats = CacheStatistics()
        self.refresh_scheduler = RefreshScheduler(dimension_config)
        
    def enrich_event(self, event, timestamp):
        """Enrich event with dimension data"""
        enrichment_start = time.time()
        
        # Extract lookup keys
        lookup_keys = self.extract_lookup_keys(event)
        
        enriched_data = {}
        cache_hits = 0
        cache_misses = 0
        
        for lookup_key, lookup_value in lookup_keys.items():
            # Try cache first
            cached_data = self.dimension_cache.get(lookup_value)
            
            if cached_data is not None:
                cache_hits += 1
                enriched_data[lookup_key] = cached_data
            else:
                cache_misses += 1
                # Cache miss - fetch from source
                dimension_data = self.fetch_dimension_data(
                    lookup_key, lookup_value
                )
                
                if dimension_data is not None:
                    # Add to cache
                    self.dimension_cache.put(lookup_value, dimension_data)
                    enriched_data[lookup_key] = dimension_data
                else:
                    enriched_data[lookup_key] = self.get_default_value(lookup_key)
        
        # Create enriched event
        enriched_event = dict(event)  # Copy original event
        enriched_event['enrichment'] = enriched_data
        enriched_event['enrichment_metadata'] = {
            'timestamp': timestamp,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'enrichment_latency_ms': (time.time() - enrichment_start) * 1000
        }
        
        # Update cache statistics
        self.cache_stats.record_lookups(cache_hits, cache_misses)
        
        return enriched_event
    
    def extract_lookup_keys(self, event):
        """Extract lookup keys from event based on configuration"""
        lookup_keys = {}
        
        for lookup_config in self.dimension_config['lookups']:
            event_field = lookup_config['event_field']
            lookup_key = lookup_config['lookup_key']
            
            if event_field in event:
                lookup_keys[lookup_key] = event[event_field]
        
        return lookup_keys
    
    def fetch_dimension_data(self, lookup_key, lookup_value):
        """Fetch dimension data from external source"""
        source_config = self.dimension_config['sources'][lookup_key]
        
        if source_config['type'] == 'database':
            return self.fetch_from_database(source_config, lookup_value)
        elif source_config['type'] == 'redis':
            return self.fetch_from_redis(source_config, lookup_value)
        elif source_config['type'] == 'rest_api':
            return self.fetch_from_api(source_config, lookup_value)
        else:
            raise ValueError(f"Unsupported source type: {source_config['type']}")
    
    def fetch_from_database(self, db_config, lookup_value):
        """Fetch data from database"""
        # Simplified database fetch
        query = db_config['query_template'].format(lookup_value=lookup_value)
        
        try:
            # Execute query (implementation depends on database type)
            result = self.execute_database_query(db_config, query)
            return result
        except Exception as e:
            # Log error and return None
            print(f"Database fetch error: {e}")
            return None
    
    def calculate_cache_efficiency(self):
        """Calculate cache efficiency metrics"""
        stats = self.cache_stats.get_stats()
        
        if stats['total_lookups'] == 0:
            return {'cache_hit_ratio': 0.0, 'efficiency_score': 0.0}
        
        hit_ratio = stats['cache_hits'] / stats['total_lookups']
        
        # Efficiency score considers hit ratio and latency improvement
        cache_latency_saving = (
            stats['cache_misses'] * self.dimension_config.get('source_latency_ms', 50) -
            stats['cache_hits'] * self.dimension_config.get('cache_latency_ms', 1)
        )
        
        efficiency_score = hit_ratio * (cache_latency_saving / stats['total_lookups'])
        
        return {
            'cache_hit_ratio': hit_ratio,
            'efficiency_score': efficiency_score,
            'latency_saving_ms': cache_latency_saving,
            'cache_utilization': self.dimension_cache.utilization()
        }

class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.key_positions = {}  # key -> position in deque
        
    def get(self, key):
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self._move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        """Put value in cache"""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self._move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
                del self.key_positions[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
            self.key_positions[key] = len(self.access_order) - 1
    
    def _move_to_end(self, key):
        """Move key to end of access order"""
        # Remove from current position
        current_pos = self.key_positions[key]
        # This is simplified - actual implementation would be more efficient
        
        # Add to end
        self.access_order.append(key)
        self.key_positions[key] = len(self.access_order) - 1
    
    def utilization(self):
        """Get cache utilization ratio"""
        return len(self.cache) / self.max_size
```

This completes Part 5 of Day 2, covering advanced real-time feature engineering patterns, temporal joins, and enrichment strategies. The content provides deep theoretical understanding and practical implementation approaches for streaming feature computation.