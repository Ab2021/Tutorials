# Day 2: Storage Optimization Strategies

## Table of Contents
1. [Storage Hierarchy and Access Patterns](#storage-hierarchy)
2. [Caching Strategies](#caching-strategies)
3. [Tiered Storage Management](#tiered-storage)
4. [Memory-Mapped Files](#memory-mapped-files)
5. [Distributed Storage](#distributed-storage)
6. [Study Questions](#study-questions)
7. [Code Examples](#code-examples)

---

## Storage Hierarchy and Access Patterns

Understanding storage characteristics is crucial for optimizing index performance across different storage tiers.

### Storage Performance Characteristics

#### **Storage Tier Comparison**
```python
STORAGE_CHARACTERISTICS = {
    'CPU_L1_Cache': {
        'capacity': '32KB - 64KB',
        'latency_ns': 1,
        'bandwidth_gbps': 1000,
        'cost_per_gb': 10000,
        'volatility': True
    },
    'CPU_L3_Cache': {
        'capacity': '8MB - 32MB', 
        'latency_ns': 10,
        'bandwidth_gbps': 200,
        'cost_per_gb': 1000,
        'volatility': True
    },
    'RAM': {
        'capacity': '16GB - 1TB',
        'latency_ns': 100,
        'bandwidth_gbps': 50,
        'cost_per_gb': 10,
        'volatility': True
    },
    'NVMe_SSD': {
        'capacity': '500GB - 8TB',
        'latency_ns': 100000,
        'bandwidth_gbps': 7,
        'cost_per_gb': 0.20,
        'volatility': False
    },
    'SATA_SSD': {
        'capacity': '250GB - 4TB',
        'latency_ns': 500000,
        'bandwidth_gbps': 0.6,
        'cost_per_gb': 0.15,
        'volatility': False
    },
    'HDD': {
        'capacity': '1TB - 20TB',
        'latency_ns': 10000000,
        'bandwidth_gbps': 0.15,
        'cost_per_gb': 0.02,
        'volatility': False
    }
}
```

### Access Pattern Analysis

#### **Index Access Patterns**
```python
class AccessPatternAnalyzer:
    def __init__(self):
        self.access_log = []
        self.term_access_frequency = defaultdict(int)
        self.temporal_locality = {}
        self.spatial_locality = {}
    
    def log_access(self, term, timestamp, query_context):
        """Log index access for pattern analysis"""
        access_event = {
            'term': term,
            'timestamp': timestamp,
            'query_context': query_context,
            'posting_list_size': self.get_posting_list_size(term)
        }
        
        self.access_log.append(access_event)
        self.term_access_frequency[term] += 1
        
        # Analyze temporal locality
        self.update_temporal_locality(term, timestamp)
        
        # Analyze spatial locality
        self.update_spatial_locality(term, query_context)
    
    def get_hot_terms(self, percentile=0.1):
        """Identify frequently accessed terms (hot data)"""
        total_accesses = sum(self.term_access_frequency.values())
        sorted_terms = sorted(
            self.term_access_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        hot_threshold = int(len(sorted_terms) * percentile)
        hot_terms = dict(sorted_terms[:hot_threshold])
        
        hot_access_ratio = sum(hot_terms.values()) / total_accesses
        
        return {
            'hot_terms': hot_terms,
            'hot_term_count': len(hot_terms),
            'hot_access_ratio': hot_access_ratio
        }
    
    def analyze_access_patterns(self):
        """Comprehensive access pattern analysis"""
        if not self.access_log:
            return {}
        
        # Temporal analysis
        time_intervals = []
        for i in range(1, len(self.access_log)):
            interval = self.access_log[i]['timestamp'] - self.access_log[i-1]['timestamp']
            time_intervals.append(interval)
        
        # Spatial analysis (co-occurrence in queries)
        query_terms = defaultdict(set)
        for access in self.access_log:
            query_id = access['query_context'].get('query_id')
            if query_id:
                query_terms[query_id].add(access['term'])
        
        # Calculate co-occurrence matrix
        term_cooccurrence = defaultdict(lambda: defaultdict(int))
        for terms in query_terms.values():
            terms_list = list(terms)
            for i, term1 in enumerate(terms_list):
                for term2 in terms_list[i+1:]:
                    term_cooccurrence[term1][term2] += 1
                    term_cooccurrence[term2][term1] += 1
        
        return {
            'total_accesses': len(self.access_log),
            'unique_terms': len(self.term_access_frequency),
            'avg_time_interval': np.mean(time_intervals) if time_intervals else 0,
            'hot_terms_analysis': self.get_hot_terms(),
            'term_cooccurrence': dict(term_cooccurrence)
        }
```

---

## Caching Strategies

Multi-level caching optimizes frequently accessed index components.

### Cache Hierarchy Design

#### **Multi-Level Cache Architecture**
```python
from collections import OrderedDict
import time

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def get_hit_ratio(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

class HierarchicalIndexCache:
    def __init__(self):
        # L1: Small, fast cache for hot terms
        self.l1_cache = LRUCache(capacity=1000)
        
        # L2: Medium cache for warm terms  
        self.l2_cache = LRUCache(capacity=10000)
        
        # L3: Large cache for posting list blocks
        self.l3_cache = LRUCache(capacity=100000)
        
        # Admission control
        self.access_frequency = defaultdict(int)
        self.l2_admission_threshold = 5
        self.l3_admission_threshold = 2
    
    def get_posting_list(self, term):
        """Retrieve posting list through cache hierarchy"""
        
        # Try L1 cache first
        result = self.l1_cache.get(term)
        if result is not None:
            return result
        
        # Try L2 cache
        result = self.l2_cache.get(term)
        if result is not None:
            # Promote to L1 if frequently accessed
            if self.access_frequency[term] > 10:
                self.l1_cache.put(term, result)
            return result
        
        # Try L3 cache
        result = self.l3_cache.get(term)
        if result is not None:
            # Promote to L2 based on access frequency
            if self.access_frequency[term] > self.l2_admission_threshold:
                self.l2_cache.put(term, result)
            return result
        
        # Cache miss - load from storage
        result = self.load_from_storage(term)
        
        # Insert into appropriate cache level
        self.access_frequency[term] += 1
        
        if self.access_frequency[term] > self.l2_admission_threshold:
            self.l2_cache.put(term, result)
        elif self.access_frequency[term] > self.l3_admission_threshold:
            self.l3_cache.put(term, result)
        
        return result
    
    def load_from_storage(self, term):
        """Load posting list from persistent storage"""
        # Simulate storage access
        time.sleep(0.001)  # 1ms storage latency
        return f"posting_list_for_{term}"
    
    def get_cache_statistics(self):
        """Get comprehensive cache statistics"""
        return {
            'l1_hit_ratio': self.l1_cache.get_hit_ratio(),
            'l2_hit_ratio': self.l2_cache.get_hit_ratio(),
            'l3_hit_ratio': self.l3_cache.get_hit_ratio(),
            'l1_size': len(self.l1_cache.cache),
            'l2_size': len(self.l2_cache.cache),
            'l3_size': len(self.l3_cache.cache)
        }
```

### Adaptive Caching Policies

#### **Frequency-Based Admission Control**
```python
import heapq
from collections import defaultdict

class AdaptiveFrequencyCache:
    def __init__(self, capacity, window_size=10000):
        self.capacity = capacity
        self.window_size = window_size
        self.cache = {}
        
        # Frequency tracking
        self.access_history = []
        self.current_frequencies = defaultdict(int)
        
        # Admission control
        self.admission_heap = []  # Min-heap of (frequency, term) pairs
        
    def access(self, term):
        """Process cache access with adaptive admission"""
        
        # Update access history
        self.access_history.append(term)
        if len(self.access_history) > self.window_size:
            # Remove oldest access from frequency count
            old_term = self.access_history.pop(0)
            self.current_frequencies[old_term] -= 1
            if self.current_frequencies[old_term] <= 0:
                del self.current_frequencies[old_term]
        
        # Update current frequency
        self.current_frequencies[term] += 1
        
        # Check if in cache
        if term in self.cache:
            return self.cache[term]
        
        # Cache miss - decide on admission
        if len(self.cache) < self.capacity:
            # Cache not full - admit directly
            value = self.load_data(term)
            self.cache[term] = value
            return value
        
        # Cache full - check if term should be admitted
        current_freq = self.current_frequencies[term]
        
        if self.should_admit(term, current_freq):
            # Evict least frequent item
            victim_term = self.find_victim()
            del self.cache[victim_term]
            
            # Admit new term
            value = self.load_data(term)
            self.cache[term] = value
            return value
        
        # Not admitted to cache
        return self.load_data(term)
    
    def should_admit(self, term, frequency):
        """Decide whether to admit term to cache"""
        
        # Simple policy: admit if frequency is above median
        all_frequencies = list(self.current_frequencies.values())
        if not all_frequencies:
            return True
        
        median_freq = sorted(all_frequencies)[len(all_frequencies) // 2]
        return frequency >= median_freq
    
    def find_victim(self):
        """Find least frequent item in cache for eviction"""
        min_freq = float('inf')
        victim = None
        
        for term in self.cache:
            freq = self.current_frequencies.get(term, 0)
            if freq < min_freq:
                min_freq = freq
                victim = term
        
        return victim
```

### Query Result Caching

#### **Semantic Query Caching**
```python
import hashlib
from datetime import datetime, timedelta

class QueryResultCache:
    def __init__(self, max_size=10000, ttl_minutes=60):
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def generate_query_key(self, query_terms, filters=None):
        """Generate unique key for query"""
        # Normalize query terms
        normalized_terms = sorted([term.lower().strip() for term in query_terms])
        
        # Include filters in key
        filter_str = ""
        if filters:
            filter_items = sorted(filters.items())
            filter_str = str(filter_items)
        
        # Create hash
        key_string = "|".join(normalized_terms) + "|" + filter_str
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query_terms, filters=None):
        """Retrieve cached query results"""
        key = self.generate_query_key(query_terms, filters)
        current_time = datetime.now()
        
        if key in self.cache:
            # Check if result is still valid
            cache_time = self.access_times[key]
            if current_time - cache_time < self.ttl:
                self.hit_count += 1
                self.access_times[key] = current_time  # Update access time
                return self.cache[key]
            else:
                # Expired - remove from cache
                del self.cache[key]
                del self.access_times[key]
        
        self.miss_count += 1
        return None
    
    def put(self, query_terms, filters, results):
        """Cache query results"""
        key = self.generate_query_key(query_terms, filters)
        current_time = datetime.now()
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.evict_lru()
        
        self.cache[key] = results
        self.access_times[key] = current_time
    
    def evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_statistics(self):
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_ratio': hit_ratio,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_size': len(self.cache),
            'cache_utilization': len(self.cache) / self.max_size
        }
```

---

## Tiered Storage Management

Automatically migrate data between storage tiers based on access patterns.

### Hot-Warm-Cold Data Classification

#### **Data Temperature Classification**
```python
import time
from enum import Enum

class DataTemperature(Enum):
    HOT = "hot"      # Frequently accessed, keep in fast storage
    WARM = "warm"    # Occasionally accessed, SSD storage
    COLD = "cold"    # Rarely accessed, archival storage

class TieredStorageManager:
    def __init__(self):
        self.hot_storage = {}     # In-memory/SSD
        self.warm_storage = {}    # SSD
        self.cold_storage = {}    # HDD/Cloud
        
        # Access tracking
        self.access_counts = defaultdict(int)
        self.last_access = defaultdict(float)
        
        # Thresholds for tier transitions
        self.hot_threshold = 100    # accesses per day
        self.warm_threshold = 10    # accesses per day
        self.observation_window = 86400  # 24 hours in seconds
    
    def access_data(self, key):
        """Access data item and update tier if necessary"""
        current_time = time.time()
        
        # Update access statistics
        self.access_counts[key] += 1
        self.last_access[key] = current_time
        
        # Try to find data in tiers (hot -> warm -> cold)
        if key in self.hot_storage:
            return self.hot_storage[key]
        elif key in self.warm_storage:
            data = self.warm_storage[key]
            # Consider promoting to hot tier
            if self.should_promote_to_hot(key):
                self.promote_to_hot(key, data)
            return data
        elif key in self.cold_storage:
            data = self.cold_storage[key]
            # Consider promoting to warm tier
            if self.should_promote_to_warm(key):
                self.promote_to_warm(key, data)
            return data
        else:
            # Data not found
            return None
    
    def store_data(self, key, data, initial_tier=DataTemperature.WARM):
        """Store new data in appropriate tier"""
        if initial_tier == DataTemperature.HOT:
            self.hot_storage[key] = data
        elif initial_tier == DataTemperature.WARM:
            self.warm_storage[key] = data
        else:
            self.cold_storage[key] = data
        
        # Initialize access tracking
        self.access_counts[key] = 1
        self.last_access[key] = time.time()
    
    def should_promote_to_hot(self, key):
        """Determine if item should be promoted to hot tier"""
        current_time = time.time()
        time_window = current_time - self.observation_window
        
        # Calculate recent access rate
        if self.last_access[key] > time_window:
            # Estimate access rate (simplified)
            time_since_first_access = current_time - (self.last_access[key] - 
                                                     self.access_counts[key] * 60)  # Estimate
            if time_since_first_access > 0:
                access_rate = self.access_counts[key] / time_since_first_access * 86400
                return access_rate > self.hot_threshold
        
        return False
    
    def should_promote_to_warm(self, key):
        """Determine if item should be promoted to warm tier"""
        current_time = time.time()
        time_window = current_time - self.observation_window
        
        if self.last_access[key] > time_window:
            time_since_first_access = current_time - (self.last_access[key] - 
                                                     self.access_counts[key] * 300)  # Estimate
            if time_since_first_access > 0:
                access_rate = self.access_counts[key] / time_since_first_access * 86400
                return access_rate > self.warm_threshold
        
        return False
    
    def promote_to_hot(self, key, data):
        """Promote data to hot tier"""
        self.hot_storage[key] = data
        if key in self.warm_storage:
            del self.warm_storage[key]
        if key in self.cold_storage:
            del self.cold_storage[key]
    
    def promote_to_warm(self, key, data):
        """Promote data to warm tier"""
        self.warm_storage[key] = data
        if key in self.cold_storage:
            del self.cold_storage[key]
    
    def demote_cold_data(self):
        """Demote infrequently accessed data to cold tier"""
        current_time = time.time()
        cutoff_time = current_time - self.observation_window
        
        # Check warm storage for demotion candidates
        warm_keys = list(self.warm_storage.keys())
        for key in warm_keys:
            if self.last_access[key] < cutoff_time:
                # Move to cold storage
                data = self.warm_storage[key]
                self.cold_storage[key] = data
                del self.warm_storage[key]
        
        # Check hot storage for demotion candidates
        hot_keys = list(self.hot_storage.keys())
        for key in hot_keys:
            if self.last_access[key] < cutoff_time:
                # Move to warm storage (gradual demotion)
                data = self.hot_storage[key]
                self.warm_storage[key] = data
                del self.hot_storage[key]
    
    def get_tier_statistics(self):
        """Get statistics about data distribution across tiers"""
        return {
            'hot_tier_size': len(self.hot_storage),
            'warm_tier_size': len(self.warm_storage),
            'cold_tier_size': len(self.cold_storage),
            'total_items': (len(self.hot_storage) + 
                           len(self.warm_storage) + 
                           len(self.cold_storage))
        }
```

---

## Memory-Mapped Files

Efficient access to large index files using memory mapping.

### Memory Mapping Implementation

#### **Cross-Platform Memory Mapping**
```python
import mmap
import os
import struct

class MemoryMappedIndex:
    def __init__(self, filename, create_if_missing=True):
        self.filename = filename
        self.file_handle = None
        self.mmap_handle = None
        self.index_header = None
        
        if create_if_missing and not os.path.exists(filename):
            self.create_empty_index()
        
        self.open_index()
    
    def create_empty_index(self):
        """Create empty index file with header"""
        with open(self.filename, 'wb') as f:
            # Write index header
            header = struct.pack('<4sIIII', 
                               b'MIDX',  # Magic number
                               1,        # Version
                               0,        # Number of terms
                               32,       # Header size
                               32)       # Free space offset
            f.write(header)
            
            # Pad to initial size
            f.write(b'\x00' * (1024 * 1024 - len(header)))  # 1MB initial size
    
    def open_index(self):
        """Open index file and create memory mapping"""
        self.file_handle = open(self.filename, 'r+b')
        self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0)
        
        # Read header
        self.read_header()
    
    def read_header(self):
        """Read index file header"""
        header_data = self.mmap_handle[0:20]
        magic, version, term_count, header_size, free_offset = struct.unpack('<4sIIII', header_data)
        
        if magic != b'MIDX':
            raise ValueError("Invalid index file format")
        
        self.index_header = {
            'magic': magic,
            'version': version,
            'term_count': term_count,
            'header_size': header_size,
            'free_offset': free_offset
        }
    
    def write_header(self):
        """Write updated header to file"""
        header = struct.pack('<4sIIII',
                           self.index_header['magic'],
                           self.index_header['version'], 
                           self.index_header['term_count'],
                           self.index_header['header_size'],
                           self.index_header['free_offset'])
        
        self.mmap_handle[0:20] = header
        self.mmap_handle.flush()
    
    def add_posting_list(self, term, posting_list):
        """Add posting list to memory-mapped index"""
        # Serialize posting list
        serialized = self.serialize_posting_list(posting_list)
        
        # Check if we need to expand file
        required_space = len(serialized) + len(term) + 8  # +8 for metadata
        if self.index_header['free_offset'] + required_space > len(self.mmap_handle):
            self.expand_file(required_space * 2)  # Double space for growth
        
        # Write term and posting list
        offset = self.index_header['free_offset']
        
        # Write term length and term
        term_bytes = term.encode('utf-8')
        self.mmap_handle[offset:offset+4] = struct.pack('<I', len(term_bytes))
        offset += 4
        
        self.mmap_handle[offset:offset+len(term_bytes)] = term_bytes
        offset += len(term_bytes)
        
        # Write posting list length and data
        self.mmap_handle[offset:offset+4] = struct.pack('<I', len(serialized))
        offset += 4
        
        self.mmap_handle[offset:offset+len(serialized)] = serialized
        offset += len(serialized)
        
        # Update header
        self.index_header['term_count'] += 1
        self.index_header['free_offset'] = offset
        self.write_header()
    
    def serialize_posting_list(self, posting_list):
        """Serialize posting list to bytes"""
        serialized = b''
        
        # Number of postings
        serialized += struct.pack('<I', len(posting_list))
        
        # Each posting: doc_id (4 bytes) + tf (4 bytes)
        for posting in posting_list:
            serialized += struct.pack('<II', posting.doc_id, posting.tf)
        
        return serialized
    
    def expand_file(self, additional_size):
        """Expand memory-mapped file"""
        current_size = len(self.mmap_handle)
        new_size = current_size + additional_size
        
        # Close current mapping
        self.mmap_handle.close()
        
        # Expand file
        self.file_handle.seek(0, 2)  # Seek to end
        self.file_handle.write(b'\x00' * additional_size)
        self.file_handle.flush()
        
        # Create new mapping
        self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0)
    
    def find_term(self, target_term):
        """Find term in memory-mapped index"""
        offset = self.index_header['header_size']
        target_bytes = target_term.encode('utf-8')
        
        for _ in range(self.index_header['term_count']):
            # Read term length
            term_length = struct.unpack('<I', self.mmap_handle[offset:offset+4])[0]
            offset += 4
            
            # Read term
            term_bytes = self.mmap_handle[offset:offset+term_length]
            offset += term_length
            
            # Read posting list length
            posting_list_length = struct.unpack('<I', self.mmap_handle[offset:offset+4])[0]
            offset += 4
            
            if term_bytes == target_bytes:
                # Found term - read posting list
                posting_data = self.mmap_handle[offset:offset+posting_list_length]
                return self.deserialize_posting_list(posting_data)
            
            # Skip posting list
            offset += posting_list_length
        
        return None  # Term not found
    
    def deserialize_posting_list(self, data):
        """Deserialize posting list from bytes"""
        offset = 0
        
        # Read number of postings
        num_postings = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        postings = []
        for _ in range(num_postings):
            doc_id, tf = struct.unpack('<II', data[offset:offset+8])
            postings.append(SimplePosting(doc_id, tf))
            offset += 8
        
        return postings
    
    def close(self):
        """Close memory-mapped index"""
        if self.mmap_handle:
            self.mmap_handle.close()
        if self.file_handle:
            self.file_handle.close()

class SimplePosting:
    def __init__(self, doc_id, tf):
        self.doc_id = doc_id
        self.tf = tf
    
    def __repr__(self):
        return f"Posting({self.doc_id}, {self.tf})"
```

---

## Distributed Storage

Scale index storage across multiple machines and data centers.

### Distributed Index Architecture

#### **Sharding Strategies**
```python
import hashlib
import consistent_hashing  # Hypothetical library

class DistributedIndexManager:
    def __init__(self, nodes):
        self.nodes = nodes
        self.consistent_hash = consistent_hashing.ConsistentHash(nodes)
        self.replication_factor = 3
        
        # Shard management
        self.term_to_shards = {}
        self.doc_to_shards = {}
        
        # Health monitoring
        self.node_health = {node: True for node in nodes}
    
    def get_shard_for_term(self, term):
        """Determine which shard(s) store a term's posting list"""
        # Use consistent hashing for term distribution
        primary_node = self.consistent_hash.get_node(term)
        
        # Add replicas
        replica_nodes = self.consistent_hash.get_nodes(term, self.replication_factor)
        
        return {
            'primary': primary_node,
            'replicas': replica_nodes[1:],  # Exclude primary
            'all_nodes': replica_nodes
        }
    
    def store_posting_list(self, term, posting_list):
        """Store posting list across appropriate shards"""
        shard_info = self.get_shard_for_term(term)
        
        # Store on all replica nodes
        success_count = 0
        failed_nodes = []
        
        for node in shard_info['all_nodes']:
            if self.node_health[node]:
                try:
                    self.store_on_node(node, term, posting_list)
                    success_count += 1
                except Exception as e:
                    failed_nodes.append((node, str(e)))
                    self.mark_node_unhealthy(node)
        
        # Check if we have sufficient replicas
        if success_count < (self.replication_factor + 1) // 2:  # Majority
            raise Exception(f"Insufficient replicas stored: {success_count}")
        
        return {
            'success_count': success_count,
            'failed_nodes': failed_nodes
        }
    
    def retrieve_posting_list(self, term):
        """Retrieve posting list with fault tolerance"""
        shard_info = self.get_shard_for_term(term)
        
        # Try primary first
        if self.node_health[shard_info['primary']]:
            try:
                return self.retrieve_from_node(shard_info['primary'], term)
            except Exception:
                self.mark_node_unhealthy(shard_info['primary'])
        
        # Try replicas
        for replica_node in shard_info['replicas']:
            if self.node_health[replica_node]:
                try:
                    return self.retrieve_from_node(replica_node, term)
                except Exception:
                    self.mark_node_unhealthy(replica_node)
        
        raise Exception(f"Could not retrieve posting list for term: {term}")
    
    def store_on_node(self, node, term, posting_list):
        """Store posting list on specific node"""
        # Implementation would use network communication
        # For demo, we'll simulate storage
        if not hasattr(self, 'node_storage'):
            self.node_storage = {node: {} for node in self.nodes}
        
        self.node_storage[node][term] = posting_list
    
    def retrieve_from_node(self, node, term):
        """Retrieve posting list from specific node"""
        if not hasattr(self, 'node_storage'):
            raise Exception("No data stored")
        
        if node not in self.node_storage or term not in self.node_storage[node]:
            raise Exception("Term not found on node")
        
        return self.node_storage[node][term]
    
    def mark_node_unhealthy(self, node):
        """Mark node as unhealthy"""
        self.node_health[node] = False
        print(f"Node {node} marked as unhealthy")
    
    def handle_node_failure(self, failed_node):
        """Handle permanent node failure"""
        print(f"Handling failure of node: {failed_node}")
        
        # Remove from consistent hash
        self.consistent_hash.remove_node(failed_node)
        
        # Initiate data migration from remaining replicas
        self.migrate_data_from_failed_node(failed_node)
    
    def migrate_data_from_failed_node(self, failed_node):
        """Migrate data from failed node to maintain replication"""
        # This is a simplified version
        # Real implementation would involve complex data migration
        
        affected_terms = []
        if hasattr(self, 'node_storage') and failed_node in self.node_storage:
            affected_terms = list(self.node_storage[failed_node].keys())
        
        for term in affected_terms:
            # Find new node for replica
            current_shards = self.get_shard_for_term(term)
            
            # Retrieve from healthy replica
            for node in current_shards['all_nodes']:
                if self.node_health[node] and node != failed_node:
                    try:
                        posting_list = self.retrieve_from_node(node, term)
                        
                        # Store on new replica node
                        new_replica = self.consistent_hash.get_node(term + "_replica")
                        self.store_on_node(new_replica, term, posting_list)
                        break
                    except Exception:
                        continue
```

---

## Study Questions

### Beginner Level
1. What are the key differences between different storage tiers (RAM, SSD, HDD)?
2. How does LRU caching work and why is it effective for index data?
3. What are the benefits of memory-mapped files for large indexes?
4. How does data temperature (hot/warm/cold) affect storage decisions?

### Intermediate Level  
1. Compare different cache replacement policies for index data access patterns.
2. How do you determine optimal cache sizes for each tier in a hierarchical cache?
3. What are the trade-offs between replication and sharding in distributed storage?
4. How does consistent hashing help with load balancing in distributed indexes?

### Advanced Level
1. Design a storage system that automatically migrates data based on access patterns.
2. How would you implement cross-datacenter replication with consistency guarantees?
3. Design a caching strategy that adapts to changing query patterns in real-time.
4. How do you handle storage failures while maintaining index availability?

### Tricky Questions
1. **Cache Pollution**: How do you prevent infrequent bulk operations from evicting useful cache entries?
2. **Storage Hierarchy**: When might storing frequently accessed data on slower storage actually improve performance?
3. **Distributed Consistency**: How do you maintain index consistency during node failures and network partitions?
4. **Cost Optimization**: How do you balance storage cost, performance, and reliability in a multi-tier system?

---

## Code Examples

### Complete Tiered Storage System
```python
import threading
import time
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

class ComprehensiveTieredStorage:
    def __init__(self):
        # Storage tiers
        self.memory_tier = OrderedDict()  # LRU for memory
        self.ssd_tier = {}
        self.hdd_tier = {}
        
        # Capacity limits
        self.memory_capacity = 1000
        self.ssd_capacity = 10000
        self.hdd_capacity = 100000
        
        # Access tracking
        self.access_frequency = defaultdict(int)
        self.last_access_time = defaultdict(float)
        self.access_history = defaultdict(list)
        
        # Background management
        self.management_thread = threading.Thread(target=self._background_management)
        self.management_thread.daemon = True
        self.management_thread.start()
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'ssd_hits': 0, 
            'hdd_hits': 0,
            'misses': 0,
            'promotions': 0,
            'demotions': 0
        }
    
    def get(self, key):
        """Retrieve data with automatic tier management"""
        current_time = time.time()
        
        # Update access statistics
        self.access_frequency[key] += 1
        self.last_access_time[key] = current_time
        self.access_history[key].append(current_time)
        
        # Keep only recent history (last hour)
        cutoff_time = current_time - 3600
        self.access_history[key] = [t for t in self.access_history[key] if t > cutoff_time]
        
        # Try memory tier first
        if key in self.memory_tier:
            # Move to end (most recently used)
            value = self.memory_tier.pop(key)
            self.memory_tier[key] = value
            self.stats['memory_hits'] += 1
            return value
        
        # Try SSD tier
        if key in self.ssd_tier:
            value = self.ssd_tier[key]
            self.stats['ssd_hits'] += 1
            
            # Consider promoting to memory
            if self._should_promote_to_memory(key):
                self._promote_to_memory(key, value)
            
            return value
        
        # Try HDD tier
        if key in self.hdd_tier:
            value = self.hdd_tier[key]
            self.stats['hdd_hits'] += 1
            
            # Consider promoting to SSD
            if self._should_promote_to_ssd(key):
                self._promote_to_ssd(key, value)
            
            return value
        
        # Not found
        self.stats['misses'] += 1
        return None
    
    def put(self, key, value):
        """Store data in appropriate tier"""
        current_time = time.time()
        self.access_frequency[key] += 1
        self.last_access_time[key] = current_time
        
        # Determine appropriate tier based on access pattern
        if self._should_start_in_memory(key):
            self._store_in_memory(key, value)
        elif self._should_start_in_ssd(key):
            self._store_in_ssd(key, value)
        else:
            self._store_in_hdd(key, value)
    
    def _should_promote_to_memory(self, key):
        """Determine if item should be promoted to memory tier"""
        recent_accesses = len(self.access_history[key])
        return recent_accesses >= 10  # 10 accesses in last hour
    
    def _should_promote_to_ssd(self, key):
        """Determine if item should be promoted to SSD tier"""
        recent_accesses = len(self.access_history[key])
        return recent_accesses >= 3   # 3 accesses in last hour
    
    def _should_start_in_memory(self, key):
        """Determine if new item should start in memory"""
        return False  # Conservative: start in lower tiers
    
    def _should_start_in_ssd(self, key):
        """Determine if new item should start in SSD"""
        return True   # Default to SSD for new items
    
    def _promote_to_memory(self, key, value):
        """Promote item to memory tier"""
        # Remove from lower tiers
        self.ssd_tier.pop(key, None)
        self.hdd_tier.pop(key, None)
        
        # Add to memory
        self._store_in_memory(key, value)
        self.stats['promotions'] += 1
    
    def _promote_to_ssd(self, key, value):
        """Promote item to SSD tier"""
        # Remove from HDD
        self.hdd_tier.pop(key, None)
        
        # Add to SSD
        self._store_in_ssd(key, value)
        self.stats['promotions'] += 1
    
    def _store_in_memory(self, key, value):
        """Store in memory tier with LRU eviction"""
        if len(self.memory_tier) >= self.memory_capacity and key not in self.memory_tier:
            # Evict LRU item to SSD
            lru_key, lru_value = self.memory_tier.popitem(last=False)
            self._store_in_ssd(lru_key, lru_value)
            self.stats['demotions'] += 1
        
        self.memory_tier[key] = value
    
    def _store_in_ssd(self, key, value):
        """Store in SSD tier"""
        if len(self.ssd_tier) >= self.ssd_capacity and key not in self.ssd_tier:
            # Evict least recently accessed item to HDD
            lru_key = min(self.ssd_tier.keys(), 
                         key=lambda k: self.last_access_time.get(k, 0))
            lru_value = self.ssd_tier.pop(lru_key)
            self._store_in_hdd(lru_key, lru_value)
            self.stats['demotions'] += 1
        
        self.ssd_tier[key] = value
    
    def _store_in_hdd(self, key, value):
        """Store in HDD tier"""
        if len(self.hdd_tier) >= self.hdd_capacity and key not in self.hdd_tier:
            # Evict oldest item
            oldest_key = min(self.hdd_tier.keys(),
                           key=lambda k: self.last_access_time.get(k, 0))
            del self.hdd_tier[oldest_key]
        
        self.hdd_tier[key] = value
    
    def _background_management(self):
        """Background thread for tier management"""
        while True:
            time.sleep(60)  # Run every minute
            self._rebalance_tiers()
    
    def _rebalance_tiers(self):
        """Rebalance data across tiers based on access patterns"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Check for cold data in memory tier
        cold_memory_items = []
        for key in list(self.memory_tier.keys()):
            if self.last_access_time.get(key, 0) < hour_ago:
                cold_memory_items.append(key)
        
        # Demote cold items from memory
        for key in cold_memory_items:
            value = self.memory_tier.pop(key)
            self._store_in_ssd(key, value)
            self.stats['demotions'] += 1
        
        # Check for cold data in SSD tier
        cold_ssd_items = []
        for key in list(self.ssd_tier.keys()):
            if self.last_access_time.get(key, 0) < hour_ago:
                cold_ssd_items.append(key)
        
        # Demote cold items from SSD
        for key in cold_ssd_items[:len(cold_ssd_items)//2]:  # Demote half
            value = self.ssd_tier.pop(key)
            self._store_in_hdd(key, value)
            self.stats['demotions'] += 1
    
    def get_statistics(self):
        """Get comprehensive storage statistics"""
        total_accesses = sum(self.stats[k] for k in ['memory_hits', 'ssd_hits', 'hdd_hits'])
        
        return {
            'tier_sizes': {
                'memory': len(self.memory_tier),
                'ssd': len(self.ssd_tier), 
                'hdd': len(self.hdd_tier)
            },
            'tier_utilization': {
                'memory': len(self.memory_tier) / self.memory_capacity,
                'ssd': len(self.ssd_tier) / self.ssd_capacity,
                'hdd': len(self.hdd_tier) / self.hdd_capacity
            },
            'hit_rates': {
                'memory': self.stats['memory_hits'] / max(total_accesses, 1),
                'ssd': self.stats['ssd_hits'] / max(total_accesses, 1),
                'hdd': self.stats['hdd_hits'] / max(total_accesses, 1)
            },
            'operations': {
                'promotions': self.stats['promotions'],
                'demotions': self.stats['demotions'],
                'misses': self.stats['misses']
            }
        }

# Example usage
if __name__ == "__main__":
    storage = ComprehensiveTieredStorage()
    
    # Simulate workload
    import random
    
    # Add data
    for i in range(5000):
        storage.put(f"key_{i}", f"value_{i}")
    
    # Simulate access patterns
    hot_keys = [f"key_{i}" for i in range(100)]  # First 100 keys are hot
    warm_keys = [f"key_{i}" for i in range(100, 500)]  # Next 400 are warm
    
    # Access hot keys frequently
    for _ in range(1000):
        key = random.choice(hot_keys)
        storage.get(key)
    
    # Access warm keys occasionally  
    for _ in range(200):
        key = random.choice(warm_keys)
        storage.get(key)
    
    # Access random keys rarely
    for _ in range(50):
        key = f"key_{random.randint(500, 4999)}"
        storage.get(key)
    
    # Show statistics
    stats = storage.get_statistics()
    print("Storage Statistics:")
    for category, data in stats.items():
        print(f"  {category}: {data}")
```

---

## Key Takeaways
1. **Storage Hierarchy**: Understanding storage characteristics enables optimal data placement
2. **Caching Strategies**: Multi-level caches dramatically improve index access performance  
3. **Automatic Tiering**: Data temperature-based migration optimizes cost and performance
4. **Distributed Design**: Replication and sharding enable scalability and fault tolerance
5. **Adaptive Management**: Systems should automatically adjust to changing access patterns

---

**Next**: In day2_ranking_fundamentals.md, we'll explore relevance ranking principles and algorithms that determine the order of search results.