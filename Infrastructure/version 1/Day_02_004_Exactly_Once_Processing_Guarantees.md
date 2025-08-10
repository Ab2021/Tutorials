# Day 2.4: Exactly-Once Processing Guarantees

## 🎯 Streaming Ingestion & Real-Time Feature Pipelines - Part 4

**Focus**: Fault Tolerance, Exactly-Once Semantics, and Checkpoint Mechanisms  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master exactly-once processing semantics and implementation strategies
- Understand distributed checkpointing algorithms and coordination
- Learn transactional processing patterns in streaming systems
- Implement end-to-end exactly-once guarantees across the entire pipeline

---

## 🔒 Exactly-Once Processing Fundamentals

### **Processing Guarantees Hierarchy**

#### **1. At-Most-Once Processing**
```
Mathematical Definition:
For each input event e, the number of times e affects the final result ≤ 1

Failure Scenario:
Input: [e1, e2, e3, e4, e5]
Process: [e1, e2] → FAILURE → Restart → [e4, e5]
Result: e3 is lost (processed 0 times)
```

#### **2. At-Least-Once Processing**
```
Mathematical Definition:
For each input event e, the number of times e affects the final result ≥ 1

Failure Scenario:
Input: [e1, e2, e3, e4, e5]
Process: [e1, e2, e3] → FAILURE → Restart from e2 → [e2, e3, e4, e5]
Result: e2, e3 are processed multiple times (duplicates)
```

#### **3. Exactly-Once Processing**
```
Mathematical Definition:
For each input event e, the number of times e affects the final result = 1

Implementation Challenge:
Ensure each event affects the final result exactly once, even with:
- Network failures
- Process crashes  
- Partial processing
- Out-of-order delivery
```

### **Theoretical Foundations**

#### **The Two-Phase Commit Problem in Streaming**
```python
class ExactlyOnceCoordinator:
    """Coordinate exactly-once processing across distributed components"""
    
    def __init__(self, participants):
        self.participants = participants  # Sources, processors, sinks
        self.transaction_log = TransactionLog()
        self.state = 'IDLE'
        
    def begin_transaction(self, transaction_id):
        """Begin distributed transaction for exactly-once processing"""
        self.state = 'PREPARING'
        
        # Phase 1: Prepare all participants
        prepare_results = []
        
        for participant in self.participants:
            try:
                result = participant.prepare(transaction_id)
                prepare_results.append({
                    'participant_id': participant.id,
                    'status': 'PREPARED' if result.success else 'ABORTED',
                    'checkpoint_id': result.checkpoint_id,
                    'state_snapshot': result.state_snapshot
                })
            except Exception as e:
                prepare_results.append({
                    'participant_id': participant.id,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        # Check if all participants prepared successfully
        all_prepared = all(r['status'] == 'PREPARED' for r in prepare_results)
        
        if all_prepared:
            return self.commit_transaction(transaction_id, prepare_results)
        else:
            return self.abort_transaction(transaction_id, prepare_results)
    
    def commit_transaction(self, transaction_id, prepare_results):
        """Phase 2: Commit transaction across all participants"""
        self.state = 'COMMITTING'
        
        # Log commit decision
        self.transaction_log.log_commit_decision(transaction_id, prepare_results)
        
        commit_results = []
        
        for participant in self.participants:
            try:
                participant.commit(transaction_id)
                commit_results.append({
                    'participant_id': participant.id,
                    'status': 'COMMITTED'
                })
            except Exception as e:
                # This is problematic - participant failed to commit after prepare
                # Need recovery mechanism
                commit_results.append({
                    'participant_id': participant.id,
                    'status': 'COMMIT_FAILED',
                    'error': str(e)
                })
                
                # Trigger recovery for this participant
                self.trigger_participant_recovery(participant, transaction_id)
        
        self.state = 'IDLE'
        return commit_results
    
    def abort_transaction(self, transaction_id, prepare_results):
        """Abort transaction and rollback all participants"""
        self.state = 'ABORTING'
        
        # Log abort decision
        self.transaction_log.log_abort_decision(transaction_id, prepare_results)
        
        abort_results = []
        
        for participant in self.participants:
            try:
                participant.abort(transaction_id)
                abort_results.append({
                    'participant_id': participant.id,
                    'status': 'ABORTED'
                })
            except Exception as e:
                abort_results.append({
                    'participant_id': participant.id,
                    'status': 'ABORT_FAILED',
                    'error': str(e)
                })
        
        self.state = 'IDLE'
        return abort_results
```

---

## 🔄 Checkpointing Algorithms Deep Dive

### **Chandy-Lamport Distributed Snapshot Algorithm**

#### **Theoretical Foundation**
The Chandy-Lamport algorithm enables taking consistent distributed snapshots of a running system without stopping the computation.

```python
class ChandyLamportCheckpoint:
    """Implementation of Chandy-Lamport distributed snapshot algorithm"""
    
    def __init__(self, node_id, connected_nodes):
        self.node_id = node_id
        self.connected_nodes = connected_nodes
        self.local_state = {}
        self.channel_states = {}  # channel_id -> recorded messages
        self.snapshot_initiated = False
        self.marker_received_from = set()
        
    def initiate_snapshot(self, checkpoint_id):
        """Initiate distributed snapshot from this node"""
        if self.snapshot_initiated:
            return  # Already participating in snapshot
        
        # Step 1: Record local state
        self.local_state[checkpoint_id] = self.capture_local_state()
        self.snapshot_initiated = True
        
        # Step 2: Send marker to all outgoing channels
        for neighbor in self.connected_nodes:
            self.send_marker(neighbor, checkpoint_id)
        
        # Step 3: Start recording messages on incoming channels
        for neighbor in self.connected_nodes:
            channel_id = f"{neighbor}_{self.node_id}"
            self.channel_states[channel_id] = []
        
        return {
            'checkpoint_id': checkpoint_id,
            'local_state_size_bytes': len(str(self.local_state[checkpoint_id])),
            'channels_being_recorded': len(self.channel_states)
        }
    
    def receive_marker(self, sender_id, checkpoint_id):
        """Handle received marker message"""
        channel_id = f"{sender_id}_{self.node_id}"
        
        if not self.snapshot_initiated:
            # First marker received - start snapshot process
            self.initiate_snapshot(checkpoint_id)
        
        if sender_id not in self.marker_received_from:
            # Stop recording messages from this channel
            self.marker_received_from.add(sender_id)
            
            # Channel state = messages recorded between snapshot initiation
            # and marker receipt from this channel
            final_channel_state = self.channel_states.get(channel_id, [])
            
            return {
                'channel_id': channel_id,
                'recorded_messages': len(final_channel_state),
                'channel_state_complete': True
            }
        
        return {'status': 'duplicate_marker_ignored'}
    
    def receive_message(self, sender_id, message, checkpoint_id):
        """Handle regular message during snapshot"""
        channel_id = f"{sender_id}_{self.node_id}"
        
        # Process message normally
        self.process_message(message)
        
        # If we're recording this channel, save the message
        if (self.snapshot_initiated and 
            sender_id not in self.marker_received_from and
            channel_id in self.channel_states):
            
            self.channel_states[channel_id].append({
                'message': message,
                'timestamp': time.time(),
                'checkpoint_id': checkpoint_id
            })
    
    def is_snapshot_complete(self):
        """Check if snapshot is complete for this node"""
        if not self.snapshot_initiated:
            return False
        
        # Snapshot complete when we've received markers from all neighbors
        return len(self.marker_received_from) == len(self.connected_nodes)
    
    def get_snapshot_data(self, checkpoint_id):
        """Get complete snapshot data for this checkpoint"""
        if not self.is_snapshot_complete():
            return None
        
        return {
            'node_id': self.node_id,
            'checkpoint_id': checkpoint_id,
            'local_state': self.local_state.get(checkpoint_id),
            'channel_states': dict(self.channel_states),
            'snapshot_size_bytes': self.calculate_snapshot_size(checkpoint_id),
            'consistency_hash': self.calculate_consistency_hash(checkpoint_id)
        }
```

### **Asynchronous Barrier Snapshotting (Flink's Approach)**

#### **Barrier Alignment and Processing**
```python
class FlinkCheckpointBarrier:
    """Flink's checkpoint barrier mechanism"""
    
    def __init__(self, operator_id, input_channels, output_channels):
        self.operator_id = operator_id
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.barriers_received = {}  # checkpoint_id -> {channel_id -> barrier}
        self.blocked_channels = set()
        self.buffered_records = {}  # channel_id -> [records]
        
    def receive_barrier(self, checkpoint_id, source_channel, barrier_timestamp):
        """Process received checkpoint barrier"""
        
        if checkpoint_id not in self.barriers_received:
            self.barriers_received[checkpoint_id] = {}
        
        self.barriers_received[checkpoint_id][source_channel] = {
            'timestamp': barrier_timestamp,
            'received_at': time.time()
        }
        
        # Check if this is the first barrier for this checkpoint
        if len(self.barriers_received[checkpoint_id]) == 1:
            # First barrier - start blocking this channel
            self.blocked_channels.add(source_channel)
            self.buffered_records[source_channel] = []
            
            return {'action': 'block_channel', 'channel': source_channel}
        
        # Check if we've received barriers from all input channels
        if len(self.barriers_received[checkpoint_id]) == len(self.input_channels):
            return self.trigger_checkpoint(checkpoint_id)
        
        # More barriers expected - block this channel too
        self.blocked_channels.add(source_channel)
        self.buffered_records[source_channel] = []
        
        return {'action': 'block_additional_channel', 'channel': source_channel}
    
    def trigger_checkpoint(self, checkpoint_id):
        """Trigger checkpoint when all barriers received"""
        checkpoint_start_time = time.time()
        
        # 1. Take local state snapshot
        local_snapshot = self.take_local_snapshot(checkpoint_id)
        
        # 2. Emit barriers to downstream operators
        for output_channel in self.output_channels:
            self.emit_barrier(checkpoint_id, output_channel)
        
        # 3. Unblock all input channels and process buffered records
        unblock_results = self.unblock_channels()
        
        # 4. Report checkpoint completion
        checkpoint_end_time = time.time()
        checkpoint_duration = checkpoint_end_time - checkpoint_start_time
        
        return {
            'checkpoint_id': checkpoint_id,
            'operator_id': self.operator_id,
            'local_snapshot_size_bytes': len(str(local_snapshot)),
            'checkpoint_duration_ms': checkpoint_duration * 1000,
            'buffered_records_processed': sum(
                len(records) for records in self.buffered_records.values()
            ),
            'barriers_emitted': len(self.output_channels)
        }
    
    def receive_record(self, record, source_channel):
        """Handle regular record during checkpoint process"""
        
        if source_channel in self.blocked_channels:
            # Channel is blocked - buffer the record
            if source_channel not in self.buffered_records:
                self.buffered_records[source_channel] = []
            
            self.buffered_records[source_channel].append(record)
            
            return {'action': 'buffered', 'buffer_size': len(self.buffered_records[source_channel])}
        
        # Channel not blocked - process normally
        return self.process_record(record)
    
    def unblock_channels(self):
        """Unblock channels and process buffered records"""
        unblock_results = {}
        
        for channel in list(self.blocked_channels):
            buffered_count = len(self.buffered_records.get(channel, []))
            
            # Process all buffered records
            for record in self.buffered_records.get(channel, []):
                self.process_record(record)
            
            # Clear buffers and unblock
            self.buffered_records[channel] = []
            self.blocked_channels.remove(channel)
            
            unblock_results[channel] = {
                'records_processed': buffered_count,
                'channel_unblocked': True
            }
        
        return unblock_results
    
    def calculate_checkpoint_latency(self, checkpoint_id):
        """Calculate end-to-end checkpoint latency"""
        if checkpoint_id not in self.barriers_received:
            return None
        
        barriers = self.barriers_received[checkpoint_id]
        
        # Find earliest and latest barrier timestamps
        barrier_times = [b['timestamp'] for b in barriers.values()]
        receive_times = [b['received_at'] for b in barriers.values()]
        
        return {
            'barrier_alignment_time_ms': (max(receive_times) - min(receive_times)) * 1000,
            'total_checkpoint_latency_ms': (max(receive_times) - min(barrier_times)) * 1000,
            'slowest_input_channel': max(barriers.keys(), 
                                       key=lambda ch: barriers[ch]['received_at'])
        }
```

---

## 💳 Transactional Streaming Patterns

### **Two-Phase Commit for Sinks**

#### **Transactional Kafka Producer Pattern**
```python
class TransactionalKafkaProducer:
    """Kafka producer with exactly-once semantics"""
    
    def __init__(self, bootstrap_servers, transactional_id):
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'transactional.id': transactional_id,
            'enable.idempotence': True,
            'acks': 'all',
            'retries': 2147483647,  # Max retries
            'max.in.flight.requests.per.connection': 5,
            'compression.type': 'snappy'
        }
        
        self.producer = KafkaProducer(**self.producer_config)
        self.current_transaction = None
        self.producer.init_transactions()
        
    def begin_transaction(self, checkpoint_id):
        """Begin new transaction for checkpoint"""
        if self.current_transaction:
            raise Exception(f"Transaction {self.current_transaction} already active")
        
        self.current_transaction = checkpoint_id
        self.producer.begin_transaction()
        
        return {
            'transaction_id': checkpoint_id,
            'transaction_state': 'ACTIVE',
            'producer_id': self.producer._producer_id,
            'epoch': self.producer._epoch
        }
    
    def send_record(self, topic, key, value, headers=None):
        """Send record within current transaction"""
        if not self.current_transaction:
            raise Exception("No active transaction")
        
        # Add transaction metadata to headers
        tx_headers = headers or {}
        tx_headers.update({
            'transaction_id': str(self.current_transaction),
            'producer_id': str(self.producer._producer_id),
            'sequence_number': str(self.get_next_sequence_number())
        })
        
        future = self.producer.send(
            topic=topic,
            key=key,
            value=value,
            headers=tx_headers
        )
        
        return future
    
    def prepare_commit(self):
        """Prepare phase of two-phase commit"""
        if not self.current_transaction:
            return {'status': 'NO_TRANSACTION'}
        
        try:
            # Flush all pending sends
            self.producer.flush()
            
            # Verify all messages were sent successfully
            # (This is simplified - actual implementation would track futures)
            
            return {
                'status': 'PREPARED',
                'transaction_id': self.current_transaction,
                'pending_messages': 0,
                'can_commit': True
            }
            
        except Exception as e:
            return {
                'status': 'PREPARE_FAILED',
                'transaction_id': self.current_transaction,
                'error': str(e),
                'can_commit': False
            }
    
    def commit_transaction(self):
        """Commit phase of two-phase commit"""
        if not self.current_transaction:
            raise Exception("No active transaction to commit")
        
        try:
            self.producer.commit_transaction()
            committed_tx = self.current_transaction
            self.current_transaction = None
            
            return {
                'status': 'COMMITTED',
                'transaction_id': committed_tx,
                'commit_timestamp': time.time()
            }
            
        except Exception as e:
            # Transaction failed to commit - this is serious
            # Need to abort and potentially trigger recovery
            self.abort_transaction()
            
            return {
                'status': 'COMMIT_FAILED',
                'transaction_id': self.current_transaction,
                'error': str(e),
                'recovery_required': True
            }
    
    def abort_transaction(self):
        """Abort current transaction"""
        if not self.current_transaction:
            return {'status': 'NO_TRANSACTION'}
        
        try:
            self.producer.abort_transaction()
            aborted_tx = self.current_transaction
            self.current_transaction = None
            
            return {
                'status': 'ABORTED',
                'transaction_id': aborted_tx,
                'abort_timestamp': time.time()
            }
            
        except Exception as e:
            # Failed to abort - producer may be in inconsistent state
            return {
                'status': 'ABORT_FAILED',
                'transaction_id': self.current_transaction,
                'error': str(e),
                'producer_state': 'INCONSISTENT'
            }
```

### **Idempotent Processing Patterns**

#### **Deduplication with Bloom Filters**
```python
import mmh3
import numpy as np

class BloomFilterDeduplicator:
    """Memory-efficient deduplication using Bloom filters"""
    
    def __init__(self, expected_elements=1000000, false_positive_prob=0.01):
        self.expected_elements = expected_elements
        self.false_positive_prob = false_positive_prob
        
        # Calculate optimal parameters
        self.bit_array_size = self.calculate_bit_array_size()
        self.hash_functions_count = self.calculate_hash_functions()
        
        # Initialize bit array
        self.bit_array = np.zeros(self.bit_array_size, dtype=bool)
        
        # Backup exact set for handling false positives (limited size)
        self.exact_set = set()
        self.exact_set_max_size = 10000
        
    def calculate_bit_array_size(self):
        """Calculate optimal bit array size"""
        return int(-self.expected_elements * np.log(self.false_positive_prob) / 
                  (np.log(2) ** 2))
    
    def calculate_hash_functions(self):
        """Calculate optimal number of hash functions"""
        return int(self.bit_array_size * np.log(2) / self.expected_elements)
    
    def add_element(self, element):
        """Add element to Bloom filter"""
        element_str = str(element)
        
        # Add to exact set if there's space
        if len(self.exact_set) < self.exact_set_max_size:
            self.exact_set.add(element_str)
        
        # Add to Bloom filter
        for i in range(self.hash_functions_count):
            hash_val = mmh3.hash(element_str, i) % self.bit_array_size
            self.bit_array[hash_val] = True
    
    def might_contain(self, element):
        """Check if element might be in the set (Bloom filter)"""
        element_str = str(element)
        
        # Check exact set first
        if element_str in self.exact_set:
            return True, 1.0  # Definitely contains, confidence = 1.0
        
        # Check Bloom filter
        for i in range(self.hash_functions_count):
            hash_val = mmh3.hash(element_str, i) % self.bit_array_size
            if not self.bit_array[hash_val]:
                return False, 1.0  # Definitely doesn't contain
        
        # Might contain (could be false positive)
        return True, 1.0 - self.false_positive_prob
    
    def estimate_elements_added(self):
        """Estimate number of elements added to filter"""
        bits_set = np.sum(self.bit_array)
        
        if bits_set == 0:
            return 0
        
        # Use Bloom filter formula to estimate count
        estimated_count = -(self.bit_array_size * np.log(
            1 - bits_set / self.bit_array_size
        )) / self.hash_functions_count
        
        return int(estimated_count)

class ExactlyOnceProcessor:
    """Exactly-once processor using deduplication"""
    
    def __init__(self, checkpoint_interval_ms=60000):
        self.checkpoint_interval = checkpoint_interval_ms
        self.deduplicator = BloomFilterDeduplicator()
        self.processed_offsets = {}  # partition -> max_processed_offset
        self.last_checkpoint_time = time.time()
        
    def process_record(self, record):
        """Process record with exactly-once guarantees"""
        # Create unique identifier for record deduplication
        record_id = self.create_record_id(record)
        
        # Check if already processed
        might_be_duplicate, confidence = self.deduplicator.might_contain(record_id)
        
        if might_be_duplicate and confidence > 0.99:
            return {
                'action': 'skipped_duplicate',
                'record_id': record_id,
                'confidence': confidence
            }
        
        # Process the record
        try:
            result = self.apply_business_logic(record)
            
            # Mark as processed
            self.deduplicator.add_element(record_id)
            self.update_processed_offset(record)
            
            # Check if checkpoint needed
            if self.should_checkpoint():
                self.trigger_checkpoint()
            
            return {
                'action': 'processed',
                'record_id': record_id,
                'result': result
            }
            
        except Exception as e:
            return {
                'action': 'failed',
                'record_id': record_id,
                'error': str(e),
                'retry_required': True
            }
    
    def create_record_id(self, record):
        """Create unique identifier for record"""
        # Combine multiple fields to create unique ID
        id_components = [
            str(getattr(record, 'partition', '')),
            str(getattr(record, 'offset', '')),
            str(getattr(record, 'timestamp', '')),
            str(hash(str(getattr(record, 'key', ''))))[:8],
            str(hash(str(getattr(record, 'value', ''))))[:8]
        ]
        
        return '|'.join(id_components)
    
    def should_checkpoint(self):
        """Determine if checkpoint should be triggered"""
        current_time = time.time()
        time_since_checkpoint = (current_time - self.last_checkpoint_time) * 1000
        
        return time_since_checkpoint >= self.checkpoint_interval
    
    def trigger_checkpoint(self):
        """Trigger checkpoint and cleanup"""
        checkpoint_id = int(time.time() * 1000)
        
        # Save current state
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'processed_offsets': dict(self.processed_offsets),
            'deduplicator_stats': {
                'estimated_elements': self.deduplicator.estimate_elements_added(),
                'bit_array_utilization': np.sum(self.deduplicator.bit_array) / 
                                       len(self.deduplicator.bit_array)
            }
        }
        
        # Reset deduplicator if it's getting full
        utilization = checkpoint_data['deduplicator_stats']['bit_array_utilization']
        if utilization > 0.7:  # 70% full
            self.deduplicator = BloomFilterDeduplicator()
        
        self.last_checkpoint_time = time.time()
        
        return checkpoint_data
```

This completes Part 4 of Day 2, covering the complex theoretical and practical aspects of exactly-once processing guarantees in distributed streaming systems. The content provides deep understanding of fault tolerance mechanisms and transactional processing patterns.