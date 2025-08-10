# Day 2.3: Stream Processing Engines Deep Dive

## 🔄 Streaming Ingestion & Real-Time Feature Pipelines - Part 3

**Focus**: Apache Flink, Spark Structured Streaming, and Stateful Processing  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master Apache Flink's distributed architecture and execution model
- Understand Spark Structured Streaming's micro-batch vs continuous processing
- Learn stateful processing patterns and checkpoint mechanisms
- Implement exactly-once processing guarantees

---

## ⚡ Apache Flink Architecture Deep Dive

### **Distributed Execution Model**

#### **1. Flink Cluster Architecture**
```
Flink Cluster Topology:

JobManager (Master)
├── JobGraph Management
├── Checkpoint Coordination  
├── Resource Management
└── Task Scheduling

TaskManager 1           TaskManager 2           TaskManager N
├── Task Slot 1        ├── Task Slot 1        ├── Task Slot 1
├── Task Slot 2        ├── Task Slot 2        ├── Task Slot 2
├── Memory Management  ├── Memory Management  ├── Memory Management
└── Network Buffers    └── Network Buffers    └── Network Buffers
```

#### **2. JobGraph to ExecutionGraph Transformation**
```python
class FlinkExecutionModel:
    """Model Flink's job execution pipeline"""
    
    def __init__(self):
        self.transformation_stages = [
            'StreamGraph',      # Logical operators
            'JobGraph',         # Optimized logical plan  
            'ExecutionGraph',   # Physical execution plan
            'Deployment'        # Actual task deployment
        ]
    
    def stream_to_job_graph_optimization(self, stream_operations):
        """Transform StreamGraph to optimized JobGraph"""
        
        # Operator chaining optimization
        chained_operators = self.chain_operators(stream_operations)
        
        # Parallelism inference
        parallelism_config = self.infer_parallelism(chained_operators)
        
        # Resource requirements calculation
        resource_requirements = self.calculate_resources(chained_operators)
        
        return {
            'chained_operators': chained_operators,
            'parallelism': parallelism_config,
            'resource_requirements': resource_requirements,
            'estimated_throughput': self.estimate_throughput(chained_operators)
        }
    
    def chain_operators(self, operations):
        """Chain compatible operators for optimization"""
        chained_groups = []
        current_chain = []
        
        for op in operations:
            if self.can_chain_with_previous(op, current_chain):
                current_chain.append(op)
            else:
                if current_chain:
                    chained_groups.append(current_chain)
                current_chain = [op]
        
        if current_chain:
            chained_groups.append(current_chain)
        
        return chained_groups
    
    def can_chain_with_previous(self, operation, current_chain):
        """Determine if operator can be chained"""
        if not current_chain:
            return True
            
        last_op = current_chain[-1]
        
        # Chaining rules
        chainable_conditions = [
            last_op.parallelism == operation.parallelism,
            not last_op.requires_shuffle,
            not operation.is_stateful,
            last_op.partitioning == operation.partitioning
        ]
        
        return all(chainable_conditions)
    
    def calculate_resources(self, chained_operators):
        """Calculate resource requirements for execution"""
        total_memory_mb = 0
        total_cpu_cores = 0
        network_bandwidth_mbps = 0
        
        for chain in chained_operators:
            # Memory requirements
            chain_memory = sum(op.memory_requirement_mb for op in chain)
            total_memory_mb += chain_memory
            
            # CPU requirements (max parallelism in chain)
            chain_parallelism = max(op.parallelism for op in chain)
            total_cpu_cores += chain_parallelism
            
            # Network bandwidth (for shuffling operators)
            for op in chain:
                if op.requires_shuffle:
                    network_bandwidth_mbps += op.estimated_shuffle_data_mbps
        
        return {
            'total_memory_mb': total_memory_mb,
            'total_cpu_cores': total_cpu_cores,  
            'network_bandwidth_mbps': network_bandwidth_mbps,
            'estimated_cost_per_hour': self.calculate_cost(
                total_memory_mb, total_cpu_cores, network_bandwidth_mbps
            )
        }
```

### **3. Memory Management and Network Stack**

#### **Flink's Off-Heap Memory Model**
```python
class FlinkMemoryManager:
    """Flink's sophisticated memory management system"""
    
    def __init__(self, total_memory_mb):
        self.total_memory = total_memory_mb
        self.memory_segments = self.initialize_memory_segments()
        
        # Memory allocation ratios (Flink defaults)
        self.heap_memory_ratio = 0.7
        self.off_heap_memory_ratio = 0.25  
        self.network_memory_ratio = 0.05
        
    def initialize_memory_segments(self):
        """Initialize memory segments for efficient management"""
        segment_size_kb = 32  # Flink default: 32KB segments
        num_segments = (self.total_memory * 1024) // segment_size_kb
        
        return {
            'segment_size_kb': segment_size_kb,
            'total_segments': num_segments,
            'free_segments': num_segments,
            'allocated_segments': 0,
            'segment_pool': list(range(num_segments))
        }
    
    def allocate_memory_for_operator(self, operator_config):
        """Allocate memory segments for specific operator"""
        required_segments = operator_config.memory_mb * 1024 // 32
        
        if required_segments > self.memory_segments['free_segments']:
            # Trigger memory pressure handling
            return self.handle_memory_pressure(operator_config)
        
        # Allocate segments
        allocated_segments = []
        for _ in range(required_segments):
            segment_id = self.memory_segments['segment_pool'].pop()
            allocated_segments.append(segment_id)
        
        self.memory_segments['free_segments'] -= required_segments
        self.memory_segments['allocated_segments'] += required_segments
        
        return {
            'allocation_successful': True,
            'allocated_segments': allocated_segments,
            'memory_type': self.determine_memory_type(operator_config),
            'spill_strategy': self.get_spill_strategy(operator_config)
        }
    
    def handle_memory_pressure(self, operator_config):
        """Handle memory pressure through spilling and compression"""
        strategies = [
            'compress_in_memory_data',
            'spill_to_disk', 
            'reduce_parallelism',
            'backpressure_upstream'
        ]
        
        for strategy in strategies:
            if self.apply_memory_strategy(strategy, operator_config):
                return self.allocate_memory_for_operator(operator_config)
        
        return {'allocation_successful': False, 'reason': 'insufficient_memory'}
    
    def calculate_spill_threshold(self, operator_type):
        """Calculate when to spill data to disk"""
        spill_thresholds = {
            'hash_join': 0.8,        # Spill at 80% memory usage
            'group_aggregate': 0.75,  # Spill at 75% memory usage
            'sort_operator': 0.85,    # Spill at 85% memory usage
            'window_operator': 0.7    # Spill at 70% memory usage
        }
        
        return spill_thresholds.get(operator_type, 0.8)
```

#### **Network Stack and Backpressure**
```python
class FlinkNetworkStack:
    """Flink's network communication and backpressure management"""
    
    def __init__(self, network_buffer_size_kb=32, num_network_buffers=2048):
        self.buffer_size = network_buffer_size_kb
        self.num_buffers = num_network_buffers
        self.buffer_pool = self.initialize_buffer_pool()
        self.backpressure_manager = BackpressureManager()
    
    def initialize_buffer_pool(self):
        """Initialize network buffer pool"""
        return {
            'total_buffers': self.num_buffers,
            'available_buffers': self.num_buffers,
            'buffer_size_kb': self.buffer_size,
            'total_network_memory_mb': (self.num_buffers * self.buffer_size) / 1024
        }
    
    def handle_record_emission(self, record, target_partition):
        """Handle record emission with backpressure"""
        buffer = self.acquire_network_buffer()
        
        if buffer is None:
            # No buffers available - apply backpressure
            backpressure_action = self.backpressure_manager.apply_backpressure(
                source_operator=record.source,
                severity='high'
            )
            return backpressure_action
        
        # Serialize record into buffer
        serialized_record = self.serialize_record(record)
        
        if len(serialized_record) > buffer.capacity:
            # Record too large for single buffer
            return self.handle_large_record(serialized_record, target_partition)
        
        buffer.write(serialized_record)
        
        # Send buffer when full or on timeout
        if buffer.is_full() or buffer.should_flush():
            self.flush_buffer_to_network(buffer, target_partition)
        
        return {'status': 'success', 'buffer_utilization': buffer.utilization()}

class BackpressureManager:
    """Manage backpressure propagation through the dataflow graph"""
    
    def __init__(self):
        self.backpressure_ratios = {}  # operator_id -> backpressure_ratio
        self.propagation_graph = {}    # operator dependencies
    
    def apply_backpressure(self, source_operator, severity):
        """Apply backpressure and propagate upstream"""
        backpressure_ratio = self.calculate_backpressure_ratio(severity)
        
        # Apply local backpressure
        self.backpressure_ratios[source_operator] = backpressure_ratio
        
        # Propagate upstream
        upstream_operators = self.get_upstream_operators(source_operator)
        
        for upstream_op in upstream_operators:
            upstream_ratio = backpressure_ratio * 0.8  # Damping factor
            self.propagate_backpressure(upstream_op, upstream_ratio)
        
        return {
            'local_backpressure_ratio': backpressure_ratio,
            'affected_upstream_operators': len(upstream_operators),
            'estimated_throughput_reduction': backpressure_ratio * 100
        }
    
    def calculate_backpressure_ratio(self, severity):
        """Calculate backpressure ratio based on severity"""
        severity_ratios = {
            'low': 0.1,     # 10% reduction
            'medium': 0.3,  # 30% reduction
            'high': 0.6,    # 60% reduction
            'critical': 0.9 # 90% reduction
        }
        
        return severity_ratios.get(severity, 0.5)
```

---

## 🌟 Spark Structured Streaming Architecture

### **Micro-Batch vs Continuous Processing**

#### **1. Micro-Batch Processing Model**
```python
class SparkMicroBatchEngine:
    """Spark's micro-batch processing engine"""
    
    def __init__(self, batch_interval_ms=1000):
        self.batch_interval = batch_interval_ms
        self.batch_scheduler = BatchScheduler(batch_interval_ms)
        self.state_store = StateStore()
        
    def process_micro_batch(self, batch_id, input_data):
        """Process a single micro-batch"""
        batch_start_time = time.time()
        
        # Create batch DataFrame
        batch_df = self.create_batch_dataframe(input_data, batch_id)
        
        # Apply transformations
        transformed_df = self.apply_transformations(batch_df)
        
        # Handle stateful operations
        if self.has_stateful_operations(transformed_df):
            transformed_df = self.apply_stateful_processing(
                transformed_df, batch_id
            )
        
        # Write output
        output_metrics = self.write_output(transformed_df, batch_id)
        
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time
        
        return {
            'batch_id': batch_id,
            'processing_time_ms': batch_processing_time * 1000,
            'input_records': len(input_data),
            'output_records': output_metrics['records_written'],
            'state_operations': output_metrics.get('state_operations', 0),
            'watermark_advanced': self.advance_watermark(batch_id)
        }
    
    def calculate_optimal_batch_interval(self, throughput_requirements):
        """Calculate optimal batch interval based on requirements"""
        # Factors affecting batch interval
        avg_processing_time = throughput_requirements.get('avg_processing_time_ms', 500)
        latency_requirement = throughput_requirements.get('max_latency_ms', 5000)
        input_rate = throughput_requirements.get('records_per_second', 10000)
        
        # Batch interval should be:
        # 1. Greater than processing time (for stability)
        # 2. Less than latency requirement 
        # 3. Sized appropriately for input rate
        
        min_interval = avg_processing_time * 1.5  # 50% buffer
        max_interval = min(latency_requirement * 0.8, 10000)  # 80% of latency SLA
        
        # Calculate interval based on input rate
        target_records_per_batch = min(input_rate * 2, 100000)  # Max 100k per batch
        rate_based_interval = (target_records_per_batch / input_rate) * 1000
        
        optimal_interval = max(min_interval, min(max_interval, rate_based_interval))
        
        return {
            'recommended_interval_ms': optimal_interval,
            'expected_batch_size': input_rate * (optimal_interval / 1000),
            'expected_latency_ms': optimal_interval + avg_processing_time,
            'throughput_capacity': input_rate
        }
```

#### **2. Continuous Processing Model**
```python
class SparkContinuousEngine:
    """Spark's continuous processing engine (experimental)"""
    
    def __init__(self, checkpoint_interval_ms=1000):
        self.checkpoint_interval = checkpoint_interval_ms
        self.epoch_manager = EpochManager()
        self.continuous_readers = {}
        
    def start_continuous_processing(self, query_config):
        """Start continuous processing with epoch-based coordination"""
        
        # Initialize continuous readers for each source
        for source_config in query_config.sources:
            reader = self.create_continuous_reader(source_config)
            self.continuous_readers[source_config.id] = reader
        
        # Start epoch coordinator
        epoch_coordinator = self.epoch_manager.start_coordinator()
        
        # Start processing loop
        return self.continuous_processing_loop(query_config, epoch_coordinator)
    
    def continuous_processing_loop(self, query_config, epoch_coordinator):
        """Main continuous processing loop"""
        current_epoch = 0
        
        while True:
            epoch_start_time = time.time()
            
            # Process records continuously until epoch boundary
            processed_records = 0
            
            while not epoch_coordinator.should_checkpoint(current_epoch):
                for reader in self.continuous_readers.values():
                    records = reader.read_next_batch(max_records=1000)
                    
                    if records:
                        self.process_records_continuously(records, current_epoch)
                        processed_records += len(records)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms
            
            # Epoch boundary reached - perform checkpoint
            checkpoint_result = self.perform_epoch_checkpoint(current_epoch)
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            yield {
                'epoch': current_epoch,
                'duration_ms': epoch_duration * 1000,
                'processed_records': processed_records,
                'checkpoint_result': checkpoint_result,
                'throughput_rps': processed_records / epoch_duration
            }
            
            current_epoch += 1
    
    def process_records_continuously(self, records, epoch):
        """Process records in continuous mode"""
        for record in records:
            # Apply transformations immediately
            transformed_record = self.apply_transformations(record)
            
            # Update state if needed
            if self.requires_state_update(transformed_record):
                self.update_state_continuously(transformed_record, epoch)
            
            # Emit result immediately
            self.emit_result(transformed_record, epoch)
```

---

## 🔄 Stateful Processing Deep Dive

### **State Management Patterns**

#### **1. Keyed State in Flink**
```python
class FlinkKeyedState:
    """Flink's keyed state management"""
    
    def __init__(self, key_selector, state_backend='filesystem'):
        self.key_selector = key_selector
        self.state_backend = self.initialize_state_backend(state_backend)
        self.value_states = {}
        self.list_states = {}
        self.map_states = {}
        
    def initialize_state_backend(self, backend_type):
        """Initialize appropriate state backend"""
        backends = {
            'memory': MemoryStateBackend(),
            'filesystem': FsStateBackend('/tmp/flink-checkpoints'),
            'rocksdb': RocksDBStateBackend('/tmp/flink-rocksdb')
        }
        
        return backends.get(backend_type, backends['filesystem'])
    
    def create_value_state(self, state_name, state_type, default_value=None):
        """Create value state for current key"""
        state_descriptor = ValueStateDescriptor(
            name=state_name,
            type_info=state_type,  
            default_value=default_value
        )
        
        self.value_states[state_name] = self.state_backend.getState(state_descriptor)
        
        return self.value_states[state_name]
    
    def create_list_state(self, state_name, element_type):
        """Create list state for accumulating values"""
        state_descriptor = ListStateDescriptor(
            name=state_name,
            element_type=element_type
        )
        
        self.list_states[state_name] = self.state_backend.getListState(state_descriptor)
        
        return self.list_states[state_name]
    
    def create_map_state(self, state_name, key_type, value_type):
        """Create map state for key-value storage"""
        state_descriptor = MapStateDescriptor(
            name=state_name,
            key_type=key_type,
            value_type=value_type
        )
        
        self.map_states[state_name] = self.state_backend.getMapState(state_descriptor)
        
        return self.map_states[state_name]

class StatefulWindowOperator:
    """Example stateful window operator implementation"""
    
    def __init__(self, window_size_ms, key_selector):
        self.window_size = window_size_ms
        self.key_selector = key_selector
        self.window_state = {}  # key -> window_data
        
    def process_element(self, element, timestamp, watermark):
        """Process element and maintain window state"""
        key = self.key_selector(element)
        
        # Determine which window this element belongs to
        window_start = self.get_window_start(timestamp)
        window_end = window_start + self.window_size
        window_id = f"{key}_{window_start}_{window_end}"
        
        # Initialize window state if needed
        if window_id not in self.window_state:
            self.window_state[window_id] = {
                'window_start': window_start,
                'window_end': window_end,
                'elements': [],
                'aggregated_value': self.get_initial_aggregate_value(),
                'element_count': 0
            }
        
        # Add element to window
        window_data = self.window_state[window_id]
        window_data['elements'].append(element)
        window_data['element_count'] += 1
        
        # Update aggregate incrementally
        window_data['aggregated_value'] = self.update_aggregate(
            window_data['aggregated_value'], 
            element
        )
        
        # Check if window should be triggered
        if watermark >= window_end:
            return self.trigger_window(window_id, window_data)
        
        return None  # Window not ready yet
    
    def trigger_window(self, window_id, window_data):
        """Trigger window computation and cleanup"""
        result = {
            'window_id': window_id,
            'window_start': window_data['window_start'],
            'window_end': window_data['window_end'],
            'element_count': window_data['element_count'],
            'result': window_data['aggregated_value']
        }
        
        # Clean up window state
        del self.window_state[window_id]
        
        return result
    
    def cleanup_expired_windows(self, current_watermark):
        """Cleanup windows that are beyond allowed lateness"""
        expired_windows = []
        
        for window_id, window_data in list(self.window_state.items()):
            if current_watermark > window_data['window_end'] + self.allowed_lateness:
                expired_windows.append(window_id)
                del self.window_state[window_id]
        
        return expired_windows
```

#### **2. State TTL (Time-To-Live)**
```python
class StateTTLManager:
    """Manage state TTL to prevent unbounded state growth"""
    
    def __init__(self, ttl_duration_ms=3600000):  # 1 hour default
        self.ttl_duration = ttl_duration_ms
        self.access_timestamps = {}  # state_key -> last_access_time
        self.cleanup_strategy = 'incremental'  # 'incremental' or 'full'
        
    def access_state(self, state_key, current_timestamp):
        """Update access timestamp when state is accessed"""
        self.access_timestamps[state_key] = current_timestamp
        
        # Trigger incremental cleanup if configured
        if self.cleanup_strategy == 'incremental':
            self.incremental_cleanup(current_timestamp)
    
    def incremental_cleanup(self, current_timestamp, cleanup_fraction=0.1):
        """Perform incremental TTL cleanup"""
        # Clean up a fraction of state entries each time
        total_entries = len(self.access_timestamps)
        entries_to_check = max(1, int(total_entries * cleanup_fraction))
        
        expired_keys = []
        entries_checked = 0
        
        for state_key, last_access in list(self.access_timestamps.items()):
            if entries_checked >= entries_to_check:
                break
                
            if current_timestamp - last_access > self.ttl_duration:
                expired_keys.append(state_key)
                del self.access_timestamps[state_key]
            
            entries_checked += 1
        
        return expired_keys
    
    def full_cleanup(self, current_timestamp):
        """Perform full TTL cleanup (expensive)"""
        expired_keys = []
        
        for state_key, last_access in list(self.access_timestamps.items()):
            if current_timestamp - last_access > self.ttl_duration:
                expired_keys.append(state_key)
                del self.access_timestamps[state_key]
        
        return expired_keys
    
    def estimate_state_size(self):
        """Estimate current state size and cleanup efficiency"""
        current_time = time.time() * 1000
        total_entries = len(self.access_timestamps)
        
        expired_entries = sum(
            1 for last_access in self.access_timestamps.values()
            if current_time - last_access > self.ttl_duration
        )
        
        return {
            'total_state_entries': total_entries,
            'expired_entries': expired_entries,
            'cleanup_efficiency': expired_entries / total_entries if total_entries > 0 else 0,
            'memory_waste_ratio': expired_entries / total_entries if total_entries > 0 else 0
        }
```

This completes Part 3 of Day 2, covering the advanced internals of stream processing engines and stateful processing patterns. The content provides deep theoretical understanding of how Flink and Spark Structured Streaming handle distributed stream processing.