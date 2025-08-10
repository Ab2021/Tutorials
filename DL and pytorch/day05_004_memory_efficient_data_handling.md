# Day 5.4: Memory-Efficient Data Handling Strategies

## Overview
Memory-efficient data handling has become increasingly critical as machine learning datasets grow exponentially in size while computational resources remain constrained. Modern deep learning applications often deal with datasets that exceed available system memory by orders of magnitude, requiring sophisticated strategies for data management, streaming, and optimization. This comprehensive exploration examines the theoretical foundations, practical techniques, and advanced algorithms that enable efficient processing of large-scale datasets while maintaining training performance and system stability.

## Theoretical Foundations of Memory Management

### Memory Hierarchy and Access Patterns

**Computer Memory Architecture**
Understanding the memory hierarchy is fundamental to designing efficient data handling systems:

**Memory Hierarchy Levels**:
1. **CPU Registers**: ~1 cycle access time, ~1KB capacity
2. **L1 Cache**: ~2-4 cycles, ~32KB-64KB per core
3. **L2 Cache**: ~10-25 cycles, ~256KB-8MB per core
4. **L3 Cache**: ~40-75 cycles, ~8MB-64MB shared
5. **Main Memory (RAM)**: ~200-300 cycles, ~GB-TB capacity
6. **SSD Storage**: ~10-100µs, ~TB capacity
7. **HDD Storage**: ~5-10ms, ~TB capacity

**Memory Access Patterns**:
- **Sequential Access**: Optimal for cache utilization, ~10x faster than random
- **Random Access**: Poor cache performance, high latency
- **Strided Access**: Intermediate performance depending on stride size
- **Block Access**: Good cache utilization for appropriately sized blocks

**Cache-Aware Algorithm Design**:
For optimal performance, algorithms should maximize:
$$\text{Cache Hit Ratio} = \frac{\text{Cache Hits}}{\text{Total Memory Accesses}}$$

**Temporal Locality**: Recently accessed data likely to be accessed again
**Spatial Locality**: Data near recently accessed locations likely to be accessed

### Mathematical Models of Memory Usage

**Memory Complexity Analysis**:
For dataset of size $N$ with sample size $s$:
- **Full Loading**: $O(N \cdot s)$ memory
- **Batch Loading**: $O(B \cdot s)$ memory where $B$ is batch size
- **Streaming**: $O(s)$ memory (single sample)

**Memory-Time Tradeoffs**:
$$\text{Total Time} = \text{Computation Time} + \text{I/O Time} + \text{Memory Management Overhead}$$

**Working Set Theory**:
The working set $W(t, \tau)$ at time $t$ with window $\tau$ is the set of pages referenced in the time interval $[t-\tau, t]$.

Optimal memory allocation: $|M| \geq |W(t, \tau)|$ to avoid thrashing.

**Memory Pressure Models**:
$$P(t) = \frac{\text{Memory Demand}(t)}{\text{Available Memory}}$$

When $P(t) > 1$, system enters memory pressure state requiring:
- **Paging**: Move data between RAM and storage
- **Compression**: Reduce memory footprint
- **Eviction**: Remove less important data

### Information Theory and Data Compression

**Entropy and Compression Bounds**:
For data with entropy $H(X)$, theoretical compression limit is:
$$\text{Compression Ratio} \geq \frac{H(X)}{\log_2(|\mathcal{X}|)}$$

**Practical Compression Algorithms**:
- **Lossless**: Huffman coding, LZ77/78, DEFLATE
- **Lossy**: Quantization, truncation, approximate representations
- **Domain-specific**: JPEG for images, MP3 for audio

**Rate-Distortion Theory**:
For lossy compression with distortion $D$:
$$R(D) = \min_{p(\hat{x}|x): E[d(x,\hat{x})] \leq D} I(X; \hat{X})$$

Where $R(D)$ is minimum rate (bits per sample) for distortion $D$.

## Memory-Efficient Data Structures

### Sparse Data Representations

**Coordinate (COO) Format**:
Store only non-zero elements with their coordinates:
```python
class COOTensor:
    def __init__(self, indices, values, shape):
        self.indices = indices  # [num_dims, num_nonzeros]
        self.values = values    # [num_nonzeros]
        self.shape = shape
    
    def memory_usage(self):
        """Calculate memory usage"""
        coord_memory = self.indices.numel() * self.indices.element_size()
        value_memory = self.values.numel() * self.values.element_size()
        return coord_memory + value_memory
    
    def sparsity_ratio(self):
        """Calculate sparsity ratio"""
        total_elements = torch.prod(torch.tensor(self.shape))
        return 1.0 - (self.values.numel() / total_elements)
```

**Compressed Sparse Row (CSR) Format**:
Efficient for matrix operations:
```python
class CSRMatrix:
    def __init__(self, data, indices, indptr, shape):
        self.data = data        # Non-zero values
        self.indices = indices  # Column indices
        self.indptr = indptr    # Row pointers
        self.shape = shape
    
    def __matmul__(self, other):
        """Matrix multiplication with CSR format"""
        if other.dim() == 1:
            return self._matvec(other)
        else:
            return self._matmat(other)
    
    def _matvec(self, vector):
        """CSR matrix-vector multiplication"""
        result = torch.zeros(self.shape[0], dtype=self.data.dtype)
        
        for i in range(self.shape[0]):
            start, end = self.indptr[i], self.indptr[i + 1]
            for j in range(start, end):
                col_idx = self.indices[j]
                result[i] += self.data[j] * vector[col_idx]
        
        return result
```

**Block Sparse Formats**:
For structured sparsity patterns:
```python
class BlockSparseMatrix:
    def __init__(self, blocks, block_indices, block_size, shape):
        self.blocks = blocks              # Dense blocks
        self.block_indices = block_indices # Block coordinates
        self.block_size = block_size      # (block_h, block_w)
        self.shape = shape
    
    def to_dense(self):
        """Convert to dense format"""
        dense = torch.zeros(self.shape)
        bh, bw = self.block_size
        
        for idx, block in zip(self.block_indices, self.blocks):
            row_start = idx[0] * bh
            col_start = idx[1] * bw
            dense[row_start:row_start+bh, col_start:col_start+bw] = block
        
        return dense
```

### Quantized Data Representations

**Fixed-Point Quantization**:
$$q = \text{round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \cdot (2^b - 1)\right)$$

Where $b$ is number of bits.

**Dequantization**:
$$\hat{x} = x_{min} + \frac{q}{2^b - 1} \cdot (x_{max} - x_{min})$$

```python
class LinearQuantizer:
    def __init__(self, num_bits=8, symmetric=False):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.scale = None
        self.zero_point = None
        self.qmin = 0
        self.qmax = 2**num_bits - 1
    
    def calibrate(self, data):
        """Calibrate quantization parameters"""
        data_min = data.min().item()
        data_max = data.max().item()
        
        if self.symmetric:
            abs_max = max(abs(data_min), abs(data_max))
            data_min, data_max = -abs_max, abs_max
            self.zero_point = (self.qmax + self.qmin) // 2
        else:
            self.zero_point = self.qmin
        
        self.scale = (data_max - data_min) / (self.qmax - self.qmin)
        return self
    
    def quantize(self, data):
        """Quantize data to lower precision"""
        if self.scale is None:
            raise ValueError("Must calibrate quantizer first")
        
        quantized = torch.round(data / self.scale + self.zero_point)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        
        return quantized.to(torch.uint8)
    
    def dequantize(self, quantized_data):
        """Convert back to floating point"""
        return (quantized_data.float() - self.zero_point) * self.scale
```

**Dynamic Quantization**:
```python
class DynamicQuantizer:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
    
    def quantize_tensor(self, tensor):
        """Quantize tensor with per-tensor scaling"""
        # Compute per-tensor scale
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        qmin, qmax = 0, 2**self.num_bits - 1
        scale = (tensor_max - tensor_min) / (qmax - qmin)
        zero_point = qmin - torch.round(tensor_min / scale)
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Store metadata
        metadata = {
            'scale': scale.item(),
            'zero_point': zero_point.item(),
            'original_shape': tensor.shape,
            'dtype': tensor.dtype
        }
        
        return quantized.to(torch.uint8), metadata
    
    def dequantize_tensor(self, quantized, metadata):
        """Dequantize tensor"""
        tensor = (quantized.float() - metadata['zero_point']) * metadata['scale']
        return tensor.to(metadata['dtype']).reshape(metadata['original_shape'])
```

### Hierarchical Data Structures

**Memory-Mapped Data Structures**:
```python
import mmap
import numpy as np

class MemoryMappedArray:
    def __init__(self, filename, dtype, shape, mode='r'):
        self.filename = filename
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self.mode = mode
        
        # Calculate total bytes needed
        self.total_bytes = np.prod(shape) * self.dtype.itemsize
        
        # Open file and create memory map
        self.file = open(filename, 'r+b' if mode == 'r+' else 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create numpy array view
        self.array = np.frombuffer(self.mmap, dtype=self.dtype).reshape(shape)
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __setitem__(self, key, value):
        if self.mode == 'r':
            raise ValueError("Cannot modify read-only memory-mapped array")
        self.array[key] = value
    
    def close(self):
        """Close memory map and file"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()
    
    def __del__(self):
        self.close()
```

**Chunked Data Structures**:
```python
class ChunkedArray:
    def __init__(self, total_size, chunk_size, dtype=torch.float32):
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.dtype = dtype
        
        # Calculate number of chunks
        self.num_chunks = (total_size + chunk_size - 1) // chunk_size
        
        # Initialize chunk storage
        self.chunks = {}
        self.loaded_chunks = set()
        self.max_loaded_chunks = 3  # LRU cache size
    
    def _load_chunk(self, chunk_idx):
        """Load chunk from storage"""
        if chunk_idx in self.loaded_chunks:
            return
        
        # Implement LRU eviction if needed
        if len(self.loaded_chunks) >= self.max_loaded_chunks:
            oldest_chunk = next(iter(self.loaded_chunks))
            self._evict_chunk(oldest_chunk)
        
        # Load chunk (from disk, database, etc.)
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_size)
        chunk_data = self._fetch_chunk_data(start_idx, end_idx)
        
        self.chunks[chunk_idx] = chunk_data
        self.loaded_chunks.add(chunk_idx)
    
    def _evict_chunk(self, chunk_idx):
        """Evict chunk from memory"""
        if chunk_idx in self.chunks:
            del self.chunks[chunk_idx]
        self.loaded_chunks.discard(chunk_idx)
    
    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        self._load_chunk(chunk_idx)
        return self.chunks[chunk_idx][local_idx]
```

## Streaming Data Processing

### Stream Processing Algorithms

**Online Algorithms for Streaming Data**:
```python
class OnlineStatistics:
    """Compute statistics incrementally"""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from current mean
    
    def update(self, value):
        """Update statistics with new value (Welford's algorithm)"""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    def variance(self):
        """Compute sample variance"""
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)
    
    def std(self):
        """Compute standard deviation"""
        return self.variance() ** 0.5

class OnlineMinMax:
    """Track minimum and maximum values"""
    
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.n = 0
    
    def update(self, value):
        """Update with new value"""
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.n += 1
```

**Reservoir Sampling**:
```python
import random

class ReservoirSampler:
    """Maintain random sample of fixed size from stream"""
    
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.n = 0
    
    def add(self, item):
        """Add item to reservoir"""
        self.n += 1
        
        if len(self.reservoir) < self.reservoir_size:
            # Fill reservoir
            self.reservoir.append(item)
        else:
            # Replace with probability reservoir_size/n
            j = random.randint(1, self.n)
            if j <= self.reservoir_size:
                self.reservoir[j - 1] = item
    
    def get_sample(self):
        """Get current reservoir sample"""
        return self.reservoir.copy()

class WeightedReservoirSampler:
    """Reservoir sampling with weights"""
    
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.keys = []
    
    def add(self, item, weight):
        """Add weighted item"""
        key = random.random() ** (1.0 / weight)
        
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
            self.keys.append(key)
        elif key > min(self.keys):
            # Replace minimum key
            min_idx = self.keys.index(min(self.keys))
            self.reservoir[min_idx] = item
            self.keys[min_idx] = key
```

**Sketch Data Structures**:
```python
import hashlib

class CountMinSketch:
    """Approximate frequency counting"""
    
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.counts = [[0] * width for _ in range(depth)]
        
        # Generate hash functions
        self.hash_functions = [
            lambda x, i=i: int(hashlib.md5(f"{x}_{i}".encode()).hexdigest(), 16) % width
            for i in range(depth)
        ]
    
    def add(self, item, count=1):
        """Add item with count"""
        for i in range(self.depth):
            j = self.hash_functions[i](item)
            self.counts[i][j] += count
    
    def query(self, item):
        """Estimate count of item"""
        estimates = [
            self.counts[i][self.hash_functions[i](item)]
            for i in range(self.depth)
        ]
        return min(estimates)

class HyperLogLog:
    """Approximate cardinality estimation"""
    
    def __init__(self, precision):
        self.precision = precision
        self.m = 2 ** precision
        self.registers = [0] * self.m
    
    def add(self, item):
        """Add item to set"""
        h = int(hashlib.md5(str(item).encode()).hexdigest(), 16)
        
        # Use first 'precision' bits for bucket
        bucket = h & (self.m - 1)
        
        # Count leading zeros in remaining bits
        remaining = h >> self.precision
        leading_zeros = self._leading_zeros(remaining) + 1
        
        # Update register
        self.registers[bucket] = max(self.registers[bucket], leading_zeros)
    
    def cardinality(self):
        """Estimate cardinality"""
        raw_estimate = self._alpha_m() * (self.m ** 2) / sum(2 ** (-x) for x in self.registers)
        
        # Apply corrections for small/large estimates
        if raw_estimate <= 2.5 * self.m:
            # Small range correction
            zeros = self.registers.count(0)
            if zeros != 0:
                return self.m * math.log(self.m / float(zeros))
        
        return raw_estimate
    
    def _leading_zeros(self, x):
        """Count leading zeros in binary representation"""
        if x == 0:
            return 32  # Assuming 32-bit integers
        return (x ^ (x - 1)).bit_length() - 1
    
    def _alpha_m(self):
        """Bias correction constant"""
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)
```

### Incremental Learning with Memory Constraints

**Stochastic Gradient Descent with Limited Memory**:
```python
class MemoryConstrainedSGD:
    """SGD optimizer with memory constraints"""
    
    def __init__(self, parameters, lr=0.01, memory_budget=1e9):  # 1GB
        self.param_groups = [{'params': list(parameters), 'lr': lr}]
        self.memory_budget = memory_budget
        self.current_memory = 0
        
        # Track memory usage of gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.current_memory += p.numel() * p.element_size()
    
    def zero_grad(self):
        """Zero gradients and free memory if needed"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if self.current_memory > self.memory_budget:
                        # Aggressive memory management
                        p.grad = None
                    else:
                        p.grad.zero_()
    
    def step(self):
        """Perform optimization step with memory awareness"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Apply update
                    p.data.add_(p.grad, alpha=-group['lr'])
                    
                    # Free gradient if memory constrained
                    if self.current_memory > self.memory_budget * 0.8:
                        p.grad = None

class GradientCompression:
    """Compress gradients to save memory"""
    
    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio
    
    def compress_gradient(self, gradient):
        """Compress gradient using top-k sparsification"""
        flat_grad = gradient.flatten()
        k = int(len(flat_grad) * self.compression_ratio)
        
        if k == 0:
            return torch.zeros_like(gradient), {}
        
        # Find top-k elements by magnitude
        _, indices = torch.topk(torch.abs(flat_grad), k)
        
        # Create sparse representation
        values = flat_grad[indices]
        
        metadata = {
            'indices': indices,
            'original_shape': gradient.shape,
            'compression_ratio': self.compression_ratio
        }
        
        return values, metadata
    
    def decompress_gradient(self, values, metadata):
        """Decompress gradient"""
        # Reconstruct sparse gradient
        flat_grad = torch.zeros(torch.prod(torch.tensor(metadata['original_shape'])))
        flat_grad[metadata['indices']] = values
        
        return flat_grad.reshape(metadata['original_shape'])
```

## Large-Scale Dataset Management

### Distributed Storage Systems

**Sharded Dataset Implementation**:
```python
import os
import pickle
from pathlib import Path

class ShardedDataset:
    """Dataset split across multiple files/shards"""
    
    def __init__(self, data_dir, shard_prefix='shard', max_shard_size=1e6):
        self.data_dir = Path(data_dir)
        self.shard_prefix = shard_prefix
        self.max_shard_size = max_shard_size
        
        # Discover existing shards
        self.shards = []
        self.shard_sizes = []
        self._discover_shards()
        
        # Create shard index for fast lookup
        self.shard_index = self._build_index()
    
    def _discover_shards(self):
        """Find all shard files"""
        shard_files = sorted(self.data_dir.glob(f"{self.shard_prefix}_*.pkl"))
        
        for shard_file in shard_files:
            self.shards.append(shard_file)
            
            # Get shard size (number of samples)
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
                self.shard_sizes.append(len(shard_data))
    
    def _build_index(self):
        """Build cumulative index for fast sample lookup"""
        cumulative_sizes = []
        total = 0
        for size in self.shard_sizes:
            total += size
            cumulative_sizes.append(total)
        return cumulative_sizes
    
    def __len__(self):
        return self.shard_index[-1] if self.shard_index else 0
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        # Find which shard contains this index
        shard_idx = self._find_shard(idx)
        
        # Calculate local index within shard
        local_idx = idx
        if shard_idx > 0:
            local_idx = idx - self.shard_index[shard_idx - 1]
        
        # Load and return item
        return self._load_item(shard_idx, local_idx)
    
    def _find_shard(self, global_idx):
        """Binary search to find shard containing index"""
        left, right = 0, len(self.shard_index) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if global_idx < self.shard_index[mid]:
                if mid == 0 or global_idx >= self.shard_index[mid - 1]:
                    return mid
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
    
    def _load_item(self, shard_idx, local_idx):
        """Load specific item from shard"""
        shard_file = self.shards[shard_idx]
        
        # Simple caching - keep last loaded shard in memory
        if not hasattr(self, '_cached_shard_idx') or self._cached_shard_idx != shard_idx:
            with open(shard_file, 'rb') as f:
                self._cached_shard_data = pickle.load(f)
                self._cached_shard_idx = shard_idx
        
        return self._cached_shard_data[local_idx]

class DistributedShardedDataset(ShardedDataset):
    """Sharded dataset for distributed training"""
    
    def __init__(self, data_dir, rank=0, world_size=1, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.rank = rank
        self.world_size = world_size
        
        # Assign shards to this worker
        self.assigned_shards = self._assign_shards()
        self._rebuild_index()
    
    def _assign_shards(self):
        """Assign shards to this worker for load balancing"""
        total_shards = len(self.shards)
        shards_per_worker = total_shards // self.world_size
        remainder = total_shards % self.world_size
        
        start_idx = self.rank * shards_per_worker + min(self.rank, remainder)
        end_idx = start_idx + shards_per_worker + (1 if self.rank < remainder else 0)
        
        return list(range(start_idx, end_idx))
    
    def _rebuild_index(self):
        """Rebuild index for assigned shards only"""
        self.local_shard_sizes = [self.shard_sizes[i] for i in self.assigned_shards]
        self.local_shards = [self.shards[i] for i in self.assigned_shards]
        
        # Rebuild cumulative index
        self.shard_index = []
        total = 0
        for size in self.local_shard_sizes:
            total += size
            self.shard_index.append(total)
```

### Hierarchical Data Management

**Multi-Level Caching System**:
```python
import threading
import time
from collections import OrderedDict

class HierarchicalCache:
    """Multi-level cache with different eviction policies"""
    
    def __init__(self, 
                 l1_size=100,      # Fast cache (in-memory)
                 l2_size=1000,     # Medium cache (compressed)
                 l3_size=10000):   # Slow cache (disk)
        
        # L1 Cache: LRU cache for hot data
        self.l1_cache = OrderedDict()
        self.l1_size = l1_size
        
        # L2 Cache: Compressed data
        self.l2_cache = OrderedDict()
        self.l2_size = l2_size
        self.compressor = self._get_compressor()
        
        # L3 Cache: Disk cache
        self.l3_cache = OrderedDict()
        self.l3_size = l3_size
        self.l3_base_path = Path("/tmp/l3_cache")
        self.l3_base_path.mkdir(exist_ok=True)
        
        # Access statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get(self, key):
        """Get item from hierarchical cache"""
        with self.lock:
            # Try L1 cache
            if key in self.l1_cache:
                self.stats['l1_hits'] += 1
                # Move to end (most recently used)
                self.l1_cache.move_to_end(key)
                return self.l1_cache[key]
            
            self.stats['l1_misses'] += 1
            
            # Try L2 cache
            if key in self.l2_cache:
                self.stats['l2_hits'] += 1
                # Decompress and promote to L1
                compressed_data = self.l2_cache[key]
                data = self.compressor.decompress(compressed_data)
                self._put_l1(key, data)
                
                # Move to end in L2
                self.l2_cache.move_to_end(key)
                return data
            
            self.stats['l2_misses'] += 1
            
            # Try L3 cache
            if key in self.l3_cache:
                self.stats['l3_hits'] += 1
                # Load from disk and promote
                file_path = self.l3_cache[key]
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Promote to higher levels
                self._put_l2(key, data)
                self._put_l1(key, data)
                
                # Move to end in L3
                self.l3_cache.move_to_end(key)
                return data
            
            self.stats['l3_misses'] += 1
            return None
    
    def put(self, key, data):
        """Put item in cache"""
        with self.lock:
            # Always put in L1 first
            self._put_l1(key, data)
    
    def _put_l1(self, key, data):
        """Put item in L1 cache"""
        self.l1_cache[key] = data
        
        # Evict if necessary
        if len(self.l1_cache) > self.l1_size:
            # Remove oldest item
            old_key, old_data = self.l1_cache.popitem(last=False)
            # Demote to L2
            self._put_l2(old_key, old_data)
    
    def _put_l2(self, key, data):
        """Put item in L2 cache (compressed)"""
        compressed_data = self.compressor.compress(data)
        self.l2_cache[key] = compressed_data
        
        # Evict if necessary
        if len(self.l2_cache) > self.l2_size:
            old_key, old_compressed = self.l2_cache.popitem(last=False)
            # Demote to L3 (decompress first)
            old_data = self.compressor.decompress(old_compressed)
            self._put_l3(old_key, old_data)
    
    def _put_l3(self, key, data):
        """Put item in L3 cache (disk)"""
        file_path = self.l3_base_path / f"{key}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.l3_cache[key] = file_path
        
        # Evict if necessary
        if len(self.l3_cache) > self.l3_size:
            old_key, old_path = self.l3_cache.popitem(last=False)
            # Remove file
            if old_path.exists():
                old_path.unlink()
    
    def _get_compressor(self):
        """Get data compressor"""
        import zlib
        
        class SimpleCompressor:
            def compress(self, data):
                serialized = pickle.dumps(data)
                return zlib.compress(serialized)
            
            def decompress(self, compressed_data):
                serialized = zlib.decompress(compressed_data)
                return pickle.loads(serialized)
        
        return SimpleCompressor()
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            total_requests = sum(self.stats.values())
            if total_requests == 0:
                return self.stats
            
            hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits'] + 
                       self.stats['l3_hits']) / total_requests
            
            stats_with_rates = self.stats.copy()
            stats_with_rates['overall_hit_rate'] = hit_rate
            return stats_with_rates
```

## Memory-Efficient Training Strategies

### Gradient Checkpointing and Activation Recomputation

**Selective Activation Checkpointing**:
```python
class CheckpointedModule(torch.nn.Module):
    """Module with selective activation checkpointing"""
    
    def __init__(self, module, checkpoint_ratio=0.5):
        super().__init__()
        self.module = module
        self.checkpoint_ratio = checkpoint_ratio
        
        # Analyze module to determine checkpointing strategy
        self.checkpoint_layers = self._select_checkpoint_layers()
    
    def _select_checkpoint_layers(self):
        """Select which layers to checkpoint based on memory usage"""
        checkpoint_layers = []
        
        # Estimate memory usage for each layer
        for name, layer in self.module.named_children():
            memory_estimate = self._estimate_layer_memory(layer)
            if memory_estimate > self._memory_threshold():
                checkpoint_layers.append(name)
        
        return checkpoint_layers
    
    def _estimate_layer_memory(self, layer):
        """Estimate memory usage of a layer"""
        param_memory = sum(p.numel() * p.element_size() for p in layer.parameters())
        
        # Rough estimate of activation memory (depends on input size)
        # This would need to be refined for specific layer types
        activation_memory = param_memory * 2  # Simplified estimate
        
        return param_memory + activation_memory
    
    def _memory_threshold(self):
        """Memory threshold for checkpointing decision"""
        available_memory = torch.cuda.get_device_properties(0).total_memory
        return available_memory * 0.01  # 1% of total GPU memory
    
    def forward(self, x):
        """Forward pass with selective checkpointing"""
        def create_checkpointed_forward(layer_name, layer):
            def checkpointed_forward(input_tensor):
                def forward_func(*inputs):
                    return layer(*inputs)
                return torch.utils.checkpoint.checkpoint(forward_func, input_tensor)
            return checkpointed_forward
        
        current_input = x
        for name, layer in self.module.named_children():
            if name in self.checkpoint_layers:
                # Use checkpointing for this layer
                forward_func = create_checkpointed_forward(name, layer)
                current_input = forward_func(current_input)
            else:
                # Normal forward pass
                current_input = layer(current_input)
        
        return current_input

class AdaptiveCheckpointing:
    """Dynamically adjust checkpointing based on memory pressure"""
    
    def __init__(self, memory_threshold=0.8):
        self.memory_threshold = memory_threshold
        self.checkpoint_enabled = False
        self.memory_history = []
    
    def should_checkpoint(self):
        """Decide whether to enable checkpointing"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = allocated / total
            
            self.memory_history.append(usage_ratio)
            if len(self.memory_history) > 10:
                self.memory_history.pop(0)
            
            # Enable checkpointing if memory usage is consistently high
            avg_usage = sum(self.memory_history) / len(self.memory_history)
            self.checkpoint_enabled = avg_usage > self.memory_threshold
        
        return self.checkpoint_enabled
    
    def checkpoint_forward(self, func, *args):
        """Conditionally apply checkpointing"""
        if self.should_checkpoint():
            return torch.utils.checkpoint.checkpoint(func, *args)
        else:
            return func(*args)
```

### Mixed Precision Training

**Automatic Mixed Precision with Memory Optimization**:
```python
class MemoryOptimizedAMP:
    """AMP with additional memory optimizations"""
    
    def __init__(self, model, optimizer, memory_efficient=True):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.memory_efficient = memory_efficient
        
        # Memory optimization settings
        self.gradient_accumulation_steps = 1
        self.max_memory_usage = 0.9  # 90% of GPU memory
    
    def adjust_batch_size(self, current_batch_size):
        """Dynamically adjust batch size based on memory usage"""
        if not torch.cuda.is_available():
            return current_batch_size
        
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        usage_ratio = allocated / total
        
        if usage_ratio > self.max_memory_usage:
            # Reduce batch size
            new_batch_size = max(1, current_batch_size // 2)
            self.gradient_accumulation_steps = current_batch_size // new_batch_size
            return new_batch_size
        elif usage_ratio < self.max_memory_usage * 0.5:
            # Increase batch size if memory allows
            new_batch_size = min(current_batch_size * 2, current_batch_size * 4)
            self.gradient_accumulation_steps = 1
            return new_batch_size
        
        return current_batch_size
    
    def train_step(self, batch):
        """Memory-optimized training step"""
        # Adjust batch size if needed
        if self.memory_efficient:
            batch_size = len(batch)
            new_batch_size = self.adjust_batch_size(batch_size)
            
            if new_batch_size != batch_size:
                # Split batch for gradient accumulation
                mini_batches = self._split_batch(batch, new_batch_size)
            else:
                mini_batches = [batch]
        else:
            mini_batches = [batch]
        
        # Process mini-batches with gradient accumulation
        total_loss = 0
        self.optimizer.zero_grad()
        
        for i, mini_batch in enumerate(mini_batches):
            with torch.cuda.amp.autocast():
                loss = self.model(mini_batch)
                # Scale loss for gradient accumulation
                loss = loss / len(mini_batches)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            
            # Clear intermediate activations to save memory
            if self.memory_efficient and i < len(mini_batches) - 1:
                torch.cuda.empty_cache()
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss
    
    def _split_batch(self, batch, mini_batch_size):
        """Split batch into mini-batches"""
        if isinstance(batch, torch.Tensor):
            return [batch[i:i+mini_batch_size] 
                   for i in range(0, len(batch), mini_batch_size)]
        elif isinstance(batch, (list, tuple)):
            mini_batches = []
            for i in range(0, len(batch[0]), mini_batch_size):
                mini_batch = tuple(item[i:i+mini_batch_size] for item in batch)
                mini_batches.append(mini_batch)
            return mini_batches
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
```

### Model Parallelism for Large Models

**Pipeline Parallelism Implementation**:
```python
class PipelineParallelModel(torch.nn.Module):
    """Model with pipeline parallelism for memory efficiency"""
    
    def __init__(self, layers, devices, micro_batch_size=1):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.devices = devices
        self.micro_batch_size = micro_batch_size
        
        # Assign layers to devices
        self.layer_devices = {}
        layers_per_device = len(layers) // len(devices)
        
        for i, layer in enumerate(self.layers):
            device_idx = min(i // layers_per_device, len(devices) - 1)
            device = devices[device_idx]
            layer.to(device)
            self.layer_devices[i] = device
    
    def forward(self, x):
        """Forward pass with pipeline parallelism"""
        batch_size = x.size(0)
        num_micro_batches = (batch_size + self.micro_batch_size - 1) // self.micro_batch_size
        
        # Split input into micro-batches
        micro_batches = self._split_input(x, num_micro_batches)
        
        # Process micro-batches through pipeline
        outputs = []
        for micro_batch in micro_batches:
            output = self._forward_micro_batch(micro_batch)
            outputs.append(output)
        
        # Concatenate outputs
        return torch.cat(outputs, dim=0)
    
    def _split_input(self, x, num_micro_batches):
        """Split input into micro-batches"""
        micro_batch_size = x.size(0) // num_micro_batches
        micro_batches = []
        
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            if i == num_micro_batches - 1:
                # Last micro-batch gets remaining samples
                end_idx = x.size(0)
            else:
                end_idx = start_idx + micro_batch_size
            
            micro_batches.append(x[start_idx:end_idx])
        
        return micro_batches
    
    def _forward_micro_batch(self, x):
        """Forward pass for single micro-batch"""
        current_input = x
        
        for i, layer in enumerate(self.layers):
            device = self.layer_devices[i]
            current_input = current_input.to(device)
            current_input = layer(current_input)
            
            # Clear previous device memory if possible
            if i > 0 and device != self.layer_devices[i-1]:
                prev_device = self.layer_devices[i-1]
                with torch.cuda.device(prev_device):
                    torch.cuda.empty_cache()
        
        return current_input

class MemoryEfficientAttention(torch.nn.Module):
    """Memory-efficient attention using Flash Attention concepts"""
    
    def __init__(self, embed_dim, num_heads, block_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = embed_dim // num_heads
        
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        """Memory-efficient attention computation"""
        B, L, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [tensor.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) 
                  for tensor in qkv]
        
        # Block-wise attention computation
        output = torch.zeros_like(q).transpose(1, 2).contiguous().view(B, L, D)
        
        for i in range(0, L, self.block_size):
            end_i = min(i + self.block_size, L)
            q_block = q[:, :, i:end_i, :]
            
            # Compute attention for this block
            attn_output = self._compute_block_attention(q_block, k, v, i, end_i)
            output[:, i:end_i, :] = attn_output
        
        return self.out_proj(output)
    
    def _compute_block_attention(self, q_block, k, v, start_idx, end_idx):
        """Compute attention for a single block"""
        # Standard attention computation for the block
        scores = torch.matmul(q_block, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask if needed
        if start_idx > 0:
            # This block can attend to all previous tokens
            attn_weights = torch.softmax(scores, dim=-1)
        else:
            # Apply causal mask within block
            mask = torch.triu(torch.ones(end_idx - start_idx, k.size(-2)), diagonal=1)
            scores.masked_fill_(mask.bool(), float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        B, H, block_len, head_dim = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, block_len, H * head_dim)
        
        return attn_output
```

## Key Questions for Review

### Memory Architecture
1. **Memory Hierarchy**: How does understanding computer memory hierarchy influence the design of efficient data processing algorithms?

2. **Cache Optimization**: What principles guide the design of cache-aware algorithms for machine learning applications?

3. **Memory Models**: How do different memory access patterns (sequential, random, strided) affect performance in deep learning workloads?

### Data Structures
4. **Sparse Representations**: When and how should sparse data structures be used to optimize memory usage in machine learning?

5. **Quantization**: What are the theoretical limits and practical considerations for data quantization in memory-constrained environments?

6. **Compression**: How do different compression techniques trade off between memory savings and computational overhead?

### Streaming Algorithms
7. **Online Processing**: What are the fundamental limitations and advantages of online algorithms for streaming data processing?

8. **Approximation Algorithms**: How do sketch data structures provide memory-efficient approximations for large-scale data analytics?

9. **Incremental Learning**: What strategies enable effective learning from streaming data with limited memory resources?

### Training Optimization
10. **Gradient Checkpointing**: How does gradient checkpointing trade computation for memory, and when is this beneficial?

11. **Mixed Precision**: What are the memory and computational benefits of mixed precision training, and what are its limitations?

12. **Model Parallelism**: How do different parallelism strategies (data, model, pipeline) address memory constraints in large model training?

## Advanced Topics and Future Directions

### Neuromorphic and Edge Computing

**Event-Driven Data Processing**:
```python
class EventDrivenProcessor:
    """Process data only when events occur"""
    
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.last_state = None
        self.event_buffer = []
        self.max_buffer_size = 1000
    
    def process_sample(self, sample):
        """Process sample only if significant change detected"""
        if self.last_state is None:
            self.last_state = sample
            return self._full_processing(sample)
        
        # Compute change magnitude
        change = torch.norm(sample - self.last_state).item()
        
        if change > self.threshold:
            # Significant change - process normally
            result = self._full_processing(sample)
            self.last_state = sample
            return result
        else:
            # Minor change - use cached result or skip
            return self._lightweight_processing(sample)
    
    def _full_processing(self, sample):
        """Complete processing pipeline"""
        # Expensive operations here
        return self.complex_transform(sample)
    
    def _lightweight_processing(self, sample):
        """Minimal processing for minor changes"""
        # Cheap approximation or cached result
        return self.last_result * self._compute_scaling(sample)
```

**Federated Learning with Memory Constraints**:
```python
class MemoryConstrainedFederatedClient:
    """Federated learning client with memory limitations"""
    
    def __init__(self, model, memory_budget=1e8):  # 100MB
        self.model = model
        self.memory_budget = memory_budget
        self.local_data_cache = {}
        self.gradient_buffer = None
        
        # Memory monitoring
        self.memory_tracker = MemoryTracker()
    
    def local_training_step(self, data_batch):
        """Training step with memory monitoring"""
        current_memory = self.memory_tracker.get_memory_usage()
        
        if current_memory > self.memory_budget * 0.9:
            # Emergency memory management
            self._emergency_cleanup()
        
        # Gradient accumulation if memory constrained
        if current_memory > self.memory_budget * 0.7:
            return self._memory_efficient_training(data_batch)
        else:
            return self._standard_training(data_batch)
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        # Clear cached data
        self.local_data_cache.clear()
        
        # Clear gradient buffer
        if self.gradient_buffer is not None:
            del self.gradient_buffer
            self.gradient_buffer = None
        
        # Force garbage collection
        torch.cuda.empty_cache()
        gc.collect()
    
    def _memory_efficient_training(self, data_batch):
        """Training with aggressive memory optimization"""
        # Process in smaller sub-batches
        sub_batch_size = max(1, len(data_batch) // 4)
        total_loss = 0
        
        for i in range(0, len(data_batch), sub_batch_size):
            sub_batch = data_batch[i:i+sub_batch_size]
            
            # Forward pass
            with torch.cuda.amp.autocast():
                loss = self.model(sub_batch)
            
            # Backward pass with gradient accumulation
            loss.backward()
            total_loss += loss.item()
            
            # Clear intermediate tensors
            del loss
            torch.cuda.empty_cache()
        
        return total_loss
```

### Quantum-Inspired Algorithms

**Tensor Network Decompositions for Memory Efficiency**:
```python
class TensorTrainDecomposition:
    """Tensor Train decomposition for memory-efficient storage"""
    
    def __init__(self, max_rank=10):
        self.max_rank = max_rank
        self.cores = []
        self.original_shape = None
    
    def decompose(self, tensor):
        """Decompose tensor into TT format"""
        self.original_shape = tensor.shape
        
        # Reshape tensor for decomposition
        current_tensor = tensor.clone()
        self.cores = []
        
        # TT decomposition algorithm
        for i in range(len(tensor.shape) - 1):
            # Reshape current tensor
            left_dim = current_tensor.shape[0]
            right_dim = current_tensor.numel() // left_dim
            
            matrix = current_tensor.view(left_dim, right_dim)
            
            # SVD with rank truncation
            U, S, V = torch.svd(matrix)
            
            # Truncate to max rank
            rank = min(self.max_rank, len(S))
            U_trunc = U[:, :rank]
            S_trunc = S[:rank]
            V_trunc = V[:, :rank]
            
            # Store core tensor
            if i == 0:
                core_shape = (1, left_dim, rank)
                core = U_trunc.T.view(core_shape)
            else:
                prev_rank = self.cores[-1].shape[-1]
                core_shape = (prev_rank, left_dim, rank)
                core = U_trunc.T.view(core_shape)
            
            self.cores.append(core)
            
            # Prepare for next iteration
            current_tensor = (torch.diag(S_trunc) @ V_trunc.T).view(-1, *tensor.shape[i+1:])
        
        # Last core
        last_core_shape = (self.cores[-1].shape[-1], current_tensor.shape[0], 1)
        self.cores.append(current_tensor.view(last_core_shape))
        
        return self
    
    def reconstruct(self):
        """Reconstruct original tensor from TT format"""
        if not self.cores:
            raise ValueError("No decomposition found")
        
        # Contract TT cores
        result = self.cores[0].squeeze(0)
        
        for core in self.cores[1:]:
            # Tensor contraction
            result = torch.tensordot(result, core, dims=([‑1], [0]))
        
        return result.squeeze(-1).view(self.original_shape)
    
    def memory_compression_ratio(self, original_tensor):
        """Calculate memory compression ratio"""
        original_memory = original_tensor.numel() * original_tensor.element_size()
        
        compressed_memory = sum(
            core.numel() * core.element_size() for core in self.cores
        )
        
        return original_memory / compressed_memory
```

## Conclusion

Memory-efficient data handling represents a critical capability for modern machine learning systems, enabling the processing of large-scale datasets within constrained computational environments. This comprehensive exploration has established:

**Theoretical Foundations**: Deep understanding of memory hierarchy, access patterns, and mathematical models of memory usage provides the foundation for designing algorithms that optimize for both computational efficiency and memory constraints.

**Advanced Data Structures**: Comprehensive coverage of sparse representations, quantized formats, and hierarchical storage systems enables practitioners to choose optimal data structures for their specific memory and performance requirements.

**Streaming Algorithms**: Understanding of online processing, approximation algorithms, and incremental learning techniques enables efficient processing of large-scale data streams with limited memory resources.

**Training Optimizations**: Advanced techniques including gradient checkpointing, mixed precision training, and model parallelism provide strategies for training large models within memory constraints while maintaining training effectiveness.

**System-Level Optimizations**: Integration of caching systems, distributed storage, and adaptive memory management ensures scalable and robust data processing pipelines that can adapt to varying resource constraints.

**Future Directions**: Awareness of emerging paradigms including neuromorphic computing, federated learning, and quantum-inspired algorithms provides insight into the evolution of memory-efficient computing and its applications to machine learning.

The design and implementation of memory-efficient data handling systems requires careful consideration of the entire data processing pipeline, from storage and access patterns through computation and result generation. As datasets continue to grow exponentially while memory resources remain relatively constrained, the importance of sophisticated memory management becomes increasingly critical for the practical deployment of machine learning systems.

The techniques and principles covered in this module provide the foundation for building data processing systems that can handle massive datasets efficiently, enabling machine learning applications that would otherwise be impossible due to memory constraints. These approaches are essential for democratizing access to large-scale machine learning by making it possible to train and deploy sophisticated models on resource-constrained hardware.