# Day 8: CUDA Streams and Concurrency
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 2: Advanced CUDA Programming

---

> **🎯 Focus Area:** Master asynchronous operations, overlapping computation with data transfers, and maximizing GPU utilization through concurrent execution.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** the CUDA stream abstraction and its role in concurrency
2. **Implement** overlapping memory transfers with kernel execution
3. **Create** and manage multiple CUDA streams for parallel operations
4. **Optimize** pipeline throughput using stream-based parallelism
5. **Debug** stream synchronization issues and race conditions
6. **Profile** concurrent operations using Nsight Systems

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with compute capability 3.5+ (for Hyper-Q)
- GPU with separate copy engines (most modern GPUs)

### Software Environment
```bash
# Verify async capability
nvidia-smi --query-gpu=name,compute_cap,async_engines --format=csv

# Python environment
pip install cupy-cuda12x numpy matplotlib

# For profiling
# Install Nsight Systems from NVIDIA developer website
```

### Prior Knowledge
- Day 1-7: CUDA fundamentals, memory management, kernel programming
- Understanding of asynchronous programming concepts

---

## 📖 Theoretical Foundation

### 1. What Are CUDA Streams?

A **CUDA stream** is a sequence of operations that execute in order on the GPU. Operations in different streams can run concurrently, enabling:

- **Overlap of computation and memory transfers**
- **Parallel kernel execution** (on capable GPUs)
- **Efficient pipelining** of work

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CUDA Stream Execution Model                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WITHOUT STREAMS (Sequential):                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ H2D Copy ──────▶ Kernel ──────▶ D2H Copy ──────▶ H2D Copy ──────▶ ...│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  Time: ████████████████████████████████████████████████████████████████      │
│                                                                              │
│  WITH STREAMS (Concurrent):                                                  │
│  Stream 0: ┌─H2D─┐   ┌─Kernel─┐   ┌─D2H─┐                                   │
│            │█████│   │████████│   │█████│                                   │
│  Stream 1:       └───┬─H2D─┐  │   ┌─Kernel─┐   ┌─D2H─┐                      │
│                      │█████│  │   │████████│   │█████│                      │
│  Stream 2:                 └──┴───┬─H2D─┐  │   ┌─Kernel─┐   ┌─D2H─┐         │
│                                   │█████│  │   │████████│   │█████│         │
│                                                                              │
│  Time: ████████████████████████  (Much shorter with overlap!)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. The Default Stream

CUDA provides a **default stream** (stream 0 or NULL stream) that is used when no stream is specified:

```cpp
// These operations use the default stream
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
myKernel<<<grid, block>>>(d_data);  // Implicitly uses default stream
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
```

**Important behaviors of the default stream:**
- Operations in the default stream are serialized
- The default stream synchronizes with all other streams (legacy behavior)
- Use `--default-stream per-thread` for per-thread default streams

### 3. Stream Synchronization Semantics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Stream Synchronization Options                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  cudaDeviceSynchronize()                                                     │
│  ├── Blocks host until ALL device operations complete                        │
│  └── Heavy-weight, use sparingly                                             │
│                                                                              │
│  cudaStreamSynchronize(stream)                                               │
│  ├── Blocks host until operations in specific stream complete                │
│  └── Allows other streams to continue                                        │
│                                                                              │
│  cudaStreamWaitEvent(stream, event)                                          │
│  ├── Makes stream wait for an event                                          │
│  └── Non-blocking for host, creates GPU-side dependency                      │
│                                                                              │
│  cudaEventSynchronize(event)                                                 │
│  ├── Blocks host until event is recorded                                     │
│  └── Useful for timing and selective synchronization                         │
│                                                                              │
│  cudaStreamQuery(stream) / cudaEventQuery(event)                             │
│  ├── Non-blocking check if operations are complete                           │
│  └── Returns cudaSuccess or cudaErrorNotReady                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. GPU Hardware: Copy Engines and Compute Engines

Modern NVIDIA GPUs have separate hardware units:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GPU Hardware Engines                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                         Compute Engines                             │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │     │
│  │  │   SM 0   │  │   SM 1   │  │   SM 2   │  │   ...    │           │     │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │     │
│  │                    Run CUDA kernels                                 │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                         Copy Engines                                │     │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                │     │
│  │  │  Copy Engine 0       │  │  Copy Engine 1       │                │     │
│  │  │  (Host → Device)     │  │  (Device → Host)     │                │     │
│  │  └──────────────────────┘  └──────────────────────┘                │     │
│  │        Asynchronous DMA transfers                                   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Maximum Concurrency (with proper streaming):                                │
│  • 1 H2D transfer + 1 D2H transfer + N kernels (if resources allow)         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Requirements for Overlapping

To achieve overlap between memory transfers and computation:

1. **Use Pinned (Page-Locked) Memory**
   ```cpp
   // Required for async transfers
   cudaMallocHost(&h_data, size);  // Pinned memory
   // NOT: h_data = malloc(size);   // Pageable - no overlap!
   ```

2. **Use Async Memory Copy Functions**
   ```cpp
   cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
   // NOT: cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
   ```

3. **Launch Kernels in Non-Default Streams**
   ```cpp
   myKernel<<<grid, block, 0, stream>>>(d_data);
   //                        ^^^^^^ Stream parameter
   ```

4. **Proper Stream Dependencies**
   ```cpp
   // Use events for complex dependencies
   cudaEventRecord(event, stream1);
   cudaStreamWaitEvent(stream2, event);
   ```

### 6. Hyper-Q and MPS (Multi-Process Service)

**Hyper-Q (Compute Capability 3.5+):**
- Allows multiple CPU threads to launch work simultaneously
- Up to 32 concurrent connections to the GPU
- Enables true concurrent kernel execution

**MPS (Multi-Process Service):**
- Allows multiple processes to share a GPU efficiently
- Reduces context switching overhead
- Important for multi-tenant ML platforms

```bash
# Start MPS daemon
nvidia-cuda-mps-control -d

# Check MPS status
echo get_server_list | nvidia-cuda-mps-control
```

---

## 💻 Implementation

### 🛠️ Project Structure
```text
day_008_cuda_streams/
├── src/
│   ├── stream_basics.py
│   ├── overlap_pipeline.py
│   ├── multi_stream_kernels.py
│   └── stream_benchmark.cu
├── tests/
│   └── test_streams.py
├── profiles/
│   └── README.md
└── README.md
```

### 👨‍💻 Core Implementation

#### 📁 `src/stream_basics.py` - Stream Fundamentals
```python
#!/usr/bin/env python3
"""
Day 8: CUDA Streams and Concurrency
Phase 6: AI/ML Platform Engineering with GPU Programming

This module demonstrates fundamental CUDA stream operations including
creation, synchronization, and basic overlap patterns.
"""

import cupy as cp
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import threading


@dataclass
class StreamMetrics:
    """Metrics collected from stream operations."""
    total_time_ms: float
    transfer_time_ms: float
    compute_time_ms: float
    overlap_efficiency: float  # 0.0 to 1.0
    
    def __str__(self) -> str:
        return (f"Total: {self.total_time_ms:.2f} ms, "
                f"Transfer: {self.transfer_time_ms:.2f} ms, "
                f"Compute: {self.compute_time_ms:.2f} ms, "
                f"Overlap Efficiency: {self.overlap_efficiency:.1%}")


class CUDAStreamManager:
    """
    Manager class for CUDA stream operations.
    
    This class provides a high-level interface for creating and managing
    CUDA streams, with proper resource cleanup and error handling.
    """
    
    def __init__(self, num_streams: int = 4):
        """
        Initialize the stream manager.
        
        Args:
            num_streams: Number of CUDA streams to create
        """
        self.num_streams = num_streams
        self.streams: List[cp.cuda.Stream] = []
        self.events: List[Tuple[cp.cuda.Event, cp.cuda.Event]] = []
        
        # Create streams and timing events
        for i in range(num_streams):
            stream = cp.cuda.Stream(non_blocking=True)
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            
            self.streams.append(stream)
            self.events.append((start_event, end_event))
        
        print(f"Created {num_streams} CUDA streams")
    
    def get_stream(self, index: int) -> cp.cuda.Stream:
        """Get stream by index."""
        return self.streams[index % self.num_streams]
    
    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()
    
    def synchronize_stream(self, index: int) -> None:
        """Synchronize a specific stream."""
        self.streams[index % self.num_streams].synchronize()
    
    def record_start(self, stream_index: int) -> None:
        """Record start event for timing."""
        start, _ = self.events[stream_index % self.num_streams]
        start.record(self.streams[stream_index % self.num_streams])
    
    def record_end(self, stream_index: int) -> None:
        """Record end event for timing."""
        _, end = self.events[stream_index % self.num_streams]
        end.record(self.streams[stream_index % self.num_streams])
    
    def get_elapsed_time(self, stream_index: int) -> float:
        """Get elapsed time in milliseconds for a stream."""
        start, end = self.events[stream_index % self.num_streams]
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.synchronize_all()
        self.streams.clear()
        self.events.clear()
        return False


def demonstrate_stream_basics():
    """
    Demonstrate basic stream creation and usage.
    
    This example shows how streams maintain order internally
    while allowing operations across streams to overlap.
    """
    print("\n" + "="*70)
    print("STREAM BASICS DEMONSTRATION")
    print("="*70)
    
    # Create non-blocking stream
    stream = cp.cuda.Stream(non_blocking=True)
    
    # Allocate data
    size = 10_000_000
    h_data = np.random.randn(size).astype(np.float32)
    
    print(f"\nData size: {size:,} elements ({size * 4 / 1e6:.1f} MB)")
    
    # Operations in default stream (synchronous)
    print("\n1. Default Stream (Sequential):")
    start = time.perf_counter()
    
    d_data = cp.asarray(h_data)  # H2D
    d_result = cp.sin(d_data) ** 2 + cp.cos(d_data) ** 2  # Compute
    h_result = d_result.get()  # D2H
    
    default_time = (time.perf_counter() - start) * 1000
    print(f"   Time: {default_time:.2f} ms")
    
    # Operations in explicit stream
    print("\n2. Explicit Stream:")
    with stream:
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        
        d_data = cp.asarray(h_data)
        d_result = cp.sin(d_data) ** 2 + cp.cos(d_data) ** 2
        h_result = d_result.get()
        
        end_event.record()
        end_event.synchronize()
        
        stream_time = cp.cuda.get_elapsed_time(start_event, end_event)
    
    print(f"   Time: {stream_time:.2f} ms")
    
    # Verify correctness
    expected = np.sin(h_data) ** 2 + np.cos(h_data) ** 2
    max_error = np.max(np.abs(h_result - expected))
    print(f"\n   Max Error: {max_error:.2e} (should be ~0)")
    
    return default_time, stream_time


def demonstrate_multiple_streams():
    """
    Demonstrate concurrent execution across multiple streams.
    
    This example shows how different streams can execute
    operations in parallel on the GPU.
    """
    print("\n" + "="*70)
    print("MULTIPLE STREAMS DEMONSTRATION")
    print("="*70)
    
    num_streams = 4
    chunks_per_stream = 5
    chunk_size = 1_000_000
    
    print(f"\nConfiguration:")
    print(f"  Streams: {num_streams}")
    print(f"  Chunks per stream: {chunks_per_stream}")
    print(f"  Chunk size: {chunk_size:,} elements")
    
    # Create streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    
    # Allocate pinned host memory for each chunk
    # Using numpy with proper alignment for pinned memory simulation
    h_inputs = [np.random.randn(chunk_size).astype(np.float32) 
                for _ in range(num_streams * chunks_per_stream)]
    h_outputs = [np.zeros(chunk_size, dtype=np.float32) 
                 for _ in range(num_streams * chunks_per_stream)]
    
    # GPU-side arrays
    d_buffers = [cp.zeros(chunk_size, dtype=cp.float32) for _ in range(num_streams)]
    
    # Define a compute-intensive kernel
    compute_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void heavyCompute(float* data, int N, int iterations) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            float val = data[idx];
            for (int i = 0; i < iterations; i++) {
                val = sinf(val) * cosf(val) + sqrtf(fabsf(val) + 1.0f);
            }
            data[idx] = val;
        }
    }
    ''', 'heavyCompute')
    
    block_size = 256
    grid_size = (chunk_size + block_size - 1) // block_size
    iterations = 100  # Make compute intensive
    
    # Sequential execution (baseline)
    print("\n1. Sequential Execution (Single Stream):")
    cp.cuda.Stream.null.synchronize()
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    for i in range(num_streams * chunks_per_stream):
        d_buffers[0].set(h_inputs[i])
        compute_kernel((grid_size,), (block_size,), 
                       (d_buffers[0], chunk_size, iterations))
        h_outputs[i] = d_buffers[0].get()
    end.record()
    end.synchronize()
    
    sequential_time = cp.cuda.get_elapsed_time(start, end)
    print(f"   Time: {sequential_time:.2f} ms")
    
    # Concurrent execution with multiple streams
    print("\n2. Concurrent Execution (Multiple Streams):")
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for chunk_idx in range(chunks_per_stream):
        for stream_idx in range(num_streams):
            i = chunk_idx * num_streams + stream_idx
            stream = streams[stream_idx]
            
            with stream:
                d_buffers[stream_idx].set(h_inputs[i])
                compute_kernel((grid_size,), (block_size,),
                               (d_buffers[stream_idx], chunk_size, iterations))
                h_outputs[i] = d_buffers[stream_idx].get()
    
    # Synchronize all streams
    for stream in streams:
        stream.synchronize()
    
    end.record()
    end.synchronize()
    
    concurrent_time = cp.cuda.get_elapsed_time(start, end)
    print(f"   Time: {concurrent_time:.2f} ms")
    print(f"   Speedup: {sequential_time / concurrent_time:.2f}x")
    
    return sequential_time, concurrent_time


def demonstrate_overlap_pattern():
    """
    Demonstrate the classic overlap pattern: H2D -> Compute -> D2H
    
    This pattern is fundamental for hiding memory transfer latency
    in data processing pipelines.
    """
    print("\n" + "="*70)
    print("OVERLAP PATTERN DEMONSTRATION")
    print("="*70)
    
    num_chunks = 8
    chunk_size = 2_000_000
    
    print(f"\nConfiguration:")
    print(f"  Chunks: {num_chunks}")
    print(f"  Chunk size: {chunk_size:,} elements ({chunk_size * 4 / 1e6:.1f} MB)")
    print(f"  Total data: {num_chunks * chunk_size * 4 / 1e6:.1f} MB")
    
    # Host data (using numpy - in production, use pinned memory)
    h_inputs = [np.random.randn(chunk_size).astype(np.float32) 
                for _ in range(num_chunks)]
    h_outputs = [np.zeros(chunk_size, dtype=np.float32) 
                 for _ in range(num_chunks)]
    
    # Compute kernel
    transform_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void transform(const float* input, float* output, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            float x = input[idx];
            // Some computation
            output[idx] = sqrtf(x * x + 1.0f) * sinf(x);
        }
    }
    ''', 'transform')
    
    block_size = 256
    grid_size = (chunk_size + block_size - 1) // block_size
    
    # === Pattern 1: No Overlap (Sequential) ===
    print("\n1. No Overlap (Sequential):")
    
    d_input = cp.zeros(chunk_size, dtype=cp.float32)
    d_output = cp.zeros(chunk_size, dtype=cp.float32)
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    cp.cuda.Stream.null.synchronize()
    start.record()
    
    for i in range(num_chunks):
        # H2D
        d_input.set(h_inputs[i])
        # Compute
        transform_kernel((grid_size,), (block_size,), (d_input, d_output, chunk_size))
        # D2H
        h_outputs[i] = d_output.get()
    
    end.record()
    end.synchronize()
    
    sequential_time = cp.cuda.get_elapsed_time(start, end)
    print(f"   Time: {sequential_time:.2f} ms")
    
    # === Pattern 2: Double Buffering with 2 Streams ===
    print("\n2. Double Buffering (2 Streams):")
    
    # Create 2 streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    
    # Double buffers
    d_inputs = [cp.zeros(chunk_size, dtype=cp.float32) for _ in range(2)]
    d_outputs = [cp.zeros(chunk_size, dtype=cp.float32) for _ in range(2)]
    
    cp.cuda.Stream.null.synchronize()
    start.record()
    
    for i in range(num_chunks):
        buf_idx = i % 2
        stream = streams[buf_idx]
        
        with stream:
            # H2D for this chunk
            d_inputs[buf_idx].set(h_inputs[i])
            # Compute
            transform_kernel((grid_size,), (block_size,), 
                           (d_inputs[buf_idx], d_outputs[buf_idx], chunk_size))
            # D2H
            h_outputs[i] = d_outputs[buf_idx].get()
    
    # Synchronize both streams
    for stream in streams:
        stream.synchronize()
    
    end.record()
    end.synchronize()
    
    double_buffer_time = cp.cuda.get_elapsed_time(start, end)
    print(f"   Time: {double_buffer_time:.2f} ms")
    print(f"   Speedup: {sequential_time / double_buffer_time:.2f}x")
    
    # === Pattern 3: Triple Buffering (Better Overlap) ===
    print("\n3. Triple Buffering (3 Streams):")
    
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    d_inputs = [cp.zeros(chunk_size, dtype=cp.float32) for _ in range(3)]
    d_outputs = [cp.zeros(chunk_size, dtype=cp.float32) for _ in range(3)]
    
    cp.cuda.Stream.null.synchronize()
    start.record()
    
    for i in range(num_chunks):
        buf_idx = i % 3
        stream = streams[buf_idx]
        
        with stream:
            d_inputs[buf_idx].set(h_inputs[i])
            transform_kernel((grid_size,), (block_size,),
                           (d_inputs[buf_idx], d_outputs[buf_idx], chunk_size))
            h_outputs[i] = d_outputs[buf_idx].get()
    
    for stream in streams:
        stream.synchronize()
    
    end.record()
    end.synchronize()
    
    triple_buffer_time = cp.cuda.get_elapsed_time(start, end)
    print(f"   Time: {triple_buffer_time:.2f} ms")
    print(f"   Speedup: {sequential_time / triple_buffer_time:.2f}x")
    
    # Calculate overlap efficiency
    # Theoretical minimum = max(total_transfer, total_compute)
    # Actual time should approach this with good overlap
    
    print("\n" + "-"*50)
    print("Summary:")
    print(f"  Sequential:      {sequential_time:7.2f} ms (1.00x)")
    print(f"  Double Buffer:   {double_buffer_time:7.2f} ms ({sequential_time/double_buffer_time:.2f}x)")
    print(f"  Triple Buffer:   {triple_buffer_time:7.2f} ms ({sequential_time/triple_buffer_time:.2f}x)")
    
    return sequential_time, double_buffer_time, triple_buffer_time


class AsyncPipeline:
    """
    A production-ready asynchronous data processing pipeline.
    
    This class implements a pipelined architecture where:
    1. Data is loaded from host to device asynchronously
    2. Computation happens in parallel with transfers
    3. Results are copied back while new data is being processed
    
    The pipeline uses a circular buffer of GPU memory and streams
    to achieve maximum throughput.
    """
    
    def __init__(self, 
                 chunk_size: int,
                 num_buffers: int = 3,
                 compute_kernel: Optional[cp.RawKernel] = None):
        """
        Initialize the async pipeline.
        
        Args:
            chunk_size: Size of each data chunk in elements
            num_buffers: Number of buffers for pipelining (2=double, 3=triple)
            compute_kernel: Custom kernel for processing (optional)
        """
        self.chunk_size = chunk_size
        self.num_buffers = num_buffers
        
        # Create streams
        self.streams = [cp.cuda.Stream(non_blocking=True) 
                       for _ in range(num_buffers)]
        
        # Create GPU buffers
        self.d_inputs = [cp.zeros(chunk_size, dtype=cp.float32) 
                        for _ in range(num_buffers)]
        self.d_outputs = [cp.zeros(chunk_size, dtype=cp.float32) 
                         for _ in range(num_buffers)]
        
        # Events for synchronization
        self.copy_complete_events = [cp.cuda.Event() for _ in range(num_buffers)]
        self.compute_complete_events = [cp.cuda.Event() for _ in range(num_buffers)]
        
        # Default compute kernel
        if compute_kernel is None:
            self.compute_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void defaultCompute(const float* input, float* output, int N) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < N) {
                    float x = input[idx];
                    output[idx] = x * x + 2.0f * x + 1.0f;
                }
            }
            ''', 'defaultCompute')
        else:
            self.compute_kernel = compute_kernel
        
        self.block_size = 256
        self.grid_size = (chunk_size + self.block_size - 1) // self.block_size
        
        # Metrics
        self.total_chunks_processed = 0
        self.total_time_ms = 0.0
    
    def process_batch(self, 
                      input_chunks: List[np.ndarray],
                      output_chunks: Optional[List[np.ndarray]] = None
                      ) -> List[np.ndarray]:
        """
        Process a batch of data chunks asynchronously.
        
        Args:
            input_chunks: List of input arrays to process
            output_chunks: Optional pre-allocated output arrays
            
        Returns:
            List of processed output arrays
        """
        num_chunks = len(input_chunks)
        
        if output_chunks is None:
            output_chunks = [np.zeros(self.chunk_size, dtype=np.float32) 
                           for _ in range(num_chunks)]
        
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        
        for i in range(num_chunks):
            buf_idx = i % self.num_buffers
            stream = self.streams[buf_idx]
            
            # Wait for previous use of this buffer to complete
            if i >= self.num_buffers:
                self.compute_complete_events[buf_idx].synchronize()
            
            with stream:
                # Host to Device transfer
                self.d_inputs[buf_idx].set(input_chunks[i])
                self.copy_complete_events[buf_idx].record()
                
                # Compute (implicitly waits for copy due to stream ordering)
                self.compute_kernel(
                    (self.grid_size,), (self.block_size,),
                    (self.d_inputs[buf_idx], self.d_outputs[buf_idx], self.chunk_size)
                )
                
                # Device to Host transfer
                output_chunks[i] = self.d_outputs[buf_idx].get()
                self.compute_complete_events[buf_idx].record()
        
        # Wait for all streams to complete
        for stream in self.streams:
            stream.synchronize()
        
        end_event.record()
        end_event.synchronize()
        
        elapsed = cp.cuda.get_elapsed_time(start_event, end_event)
        self.total_chunks_processed += num_chunks
        self.total_time_ms += elapsed
        
        return output_chunks
    
    def get_throughput(self) -> float:
        """Get throughput in elements per second."""
        if self.total_time_ms == 0:
            return 0.0
        total_elements = self.total_chunks_processed * self.chunk_size
        return total_elements / (self.total_time_ms / 1000)
    
    def reset_metrics(self) -> None:
        """Reset throughput metrics."""
        self.total_chunks_processed = 0
        self.total_time_ms = 0.0


def demonstrate_async_pipeline():
    """Demonstrate the AsyncPipeline class."""
    print("\n" + "="*70)
    print("ASYNC PIPELINE DEMONSTRATION")
    print("="*70)
    
    chunk_size = 1_000_000
    num_chunks = 20
    
    print(f"\nConfiguration:")
    print(f"  Chunk size: {chunk_size:,} elements")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Total data: {num_chunks * chunk_size * 4 / 1e6:.1f} MB")
    
    # Generate input data
    inputs = [np.random.randn(chunk_size).astype(np.float32) 
              for _ in range(num_chunks)]
    
    # Test with different buffer counts
    for num_buffers in [1, 2, 3, 4]:
        pipeline = AsyncPipeline(chunk_size, num_buffers)
        
        # Warmup
        _ = pipeline.process_batch(inputs[:2])
        pipeline.reset_metrics()
        
        # Benchmark
        outputs = pipeline.process_batch(inputs)
        
        throughput = pipeline.get_throughput()
        time_ms = pipeline.total_time_ms
        
        print(f"\n{num_buffers}-Buffer Pipeline:")
        print(f"  Time: {time_ms:.2f} ms")
        print(f"  Throughput: {throughput / 1e9:.2f} billion elements/sec")
        
        # Verify correctness
        expected = inputs[0] ** 2 + 2 * inputs[0] + 1
        if np.allclose(outputs[0], expected, rtol=1e-5):
            print(f"  Verification: PASSED")
        else:
            print(f"  Verification: FAILED")


def main():
    """Main function demonstrating all stream concepts."""
    print("="*70)
    print("DAY 8: CUDA STREAMS AND CONCURRENCY")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*70)
    
    # Check GPU
    print(f"\nGPU: {cp.cuda.Device().name}")
    
    # Run demonstrations
    demonstrate_stream_basics()
    demonstrate_multiple_streams()
    demonstrate_overlap_pattern()
    demonstrate_async_pipeline()
    
    print("\n" + "="*70)
    print("Day 8 Complete: CUDA Streams and Concurrency")
    print("="*70)


if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Building a Streaming Data Pipeline"

### Lab Objectives
1. Implement a data preprocessing pipeline with overlap
2. Measure and optimize pipeline throughput
3. Compare different buffering strategies
4. Profile with Nsight Systems

### Step-by-Step Instructions

#### Step 1: Setup the Lab Environment
```bash
cd day_008_cuda_streams
mkdir -p lab profiles
python -c "import cupy; print(f'CuPy version: {cupy.__version__}')"
```

#### Step 2: Implement the Basic Pipeline

```python
#!/usr/bin/env python3
"""
Lab Exercise: Streaming Data Pipeline
Day 8: CUDA Streams and Concurrency
"""

import cupy as cp
import numpy as np
import time
from typing import Callable, List
import matplotlib.pyplot as plt


class DataPipelineLab:
    """
    Lab implementation for exploring stream-based data pipelines.
    
    Students will implement different overlap strategies and
    measure their effectiveness.
    """
    
    def __init__(self, data_size: int = 10_000_000):
        """Initialize the lab with specified data size."""
        self.data_size = data_size
        self.results = {}
        
        # Preprocessing kernel (simulates ML data preprocessing)
        self.preprocess_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void preprocess(const float* raw, float* processed, int N, 
                        float mean, float std) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                // Normalize
                float val = (raw[idx] - mean) / std;
                // Apply activation-like function
                processed[idx] = 1.0f / (1.0f + expf(-val));  // Sigmoid
            }
        }
        ''', 'preprocess')
        
        print(f"Lab initialized with data size: {data_size:,} elements")
    
    def run_sequential_baseline(self, num_batches: int = 10) -> float:
        """
        TODO: Implement sequential (no overlap) baseline.
        
        Steps:
        1. For each batch:
           a. Copy data H2D
           b. Run preprocessing kernel
           c. Copy result D2H
        2. Measure total time
        """
        batch_size = self.data_size // num_batches
        
        # Generate random data
        h_data = [np.random.randn(batch_size).astype(np.float32) 
                  for _ in range(num_batches)]
        h_results = [np.zeros(batch_size, dtype=np.float32) 
                     for _ in range(num_batches)]
        
        # GPU buffers
        d_input = cp.zeros(batch_size, dtype=cp.float32)
        d_output = cp.zeros(batch_size, dtype=cp.float32)
        
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        
        # Statistics for normalization
        mean = 0.0
        std = 1.0
        
        # Timing
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        cp.cuda.Stream.null.synchronize()
        start.record()
        
        for i in range(num_batches):
            # H2D
            d_input.set(h_data[i])
            
            # Compute
            self.preprocess_kernel((grid_size,), (block_size,),
                                   (d_input, d_output, batch_size, mean, std))
            
            # D2H
            h_results[i] = d_output.get()
        
        end.record()
        end.synchronize()
        
        time_ms = cp.cuda.get_elapsed_time(start, end)
        self.results['sequential'] = time_ms
        
        return time_ms
    
    def run_overlapped_pipeline(self, num_batches: int = 10, 
                                 num_streams: int = 2) -> float:
        """
        TODO: Implement overlapped pipeline with multiple streams.
        
        Steps:
        1. Create N streams
        2. Create N sets of GPU buffers
        3. For each batch, use round-robin stream assignment
        4. Overlap H2D, Compute, D2H across streams
        """
        batch_size = self.data_size // num_batches
        
        # Generate data
        h_data = [np.random.randn(batch_size).astype(np.float32) 
                  for _ in range(num_batches)]
        h_results = [np.zeros(batch_size, dtype=np.float32) 
                     for _ in range(num_batches)]
        
        # Create streams and buffers
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
        d_inputs = [cp.zeros(batch_size, dtype=cp.float32) for _ in range(num_streams)]
        d_outputs = [cp.zeros(batch_size, dtype=cp.float32) for _ in range(num_streams)]
        
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        mean, std = 0.0, 1.0
        
        # Timing
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        cp.cuda.Stream.null.synchronize()
        start.record()
        
        for i in range(num_batches):
            stream_idx = i % num_streams
            stream = streams[stream_idx]
            
            with stream:
                # H2D
                d_inputs[stream_idx].set(h_data[i])
                
                # Compute
                self.preprocess_kernel((grid_size,), (block_size,),
                                       (d_inputs[stream_idx], d_outputs[stream_idx],
                                        batch_size, mean, std))
                
                # D2H
                h_results[i] = d_outputs[stream_idx].get()
        
        # Sync all
        for stream in streams:
            stream.synchronize()
        
        end.record()
        end.synchronize()
        
        time_ms = cp.cuda.get_elapsed_time(start, end)
        self.results[f'overlapped_{num_streams}_streams'] = time_ms
        
        return time_ms
    
    def run_experiments(self):
        """Run all experiments and collect results."""
        print("\n" + "="*60)
        print("RUNNING LAB EXPERIMENTS")
        print("="*60)
        
        num_batches = 20
        
        # Baseline
        print("\n1. Sequential Baseline:")
        seq_time = self.run_sequential_baseline(num_batches)
        print(f"   Time: {seq_time:.2f} ms")
        
        # Different stream counts
        for num_streams in [2, 3, 4, 8]:
            print(f"\n2. Overlapped ({num_streams} streams):")
            overlap_time = self.run_overlapped_pipeline(num_batches, num_streams)
            speedup = seq_time / overlap_time
            print(f"   Time: {overlap_time:.2f} ms")
            print(f"   Speedup: {speedup:.2f}x")
        
        return self.results
    
    def plot_results(self, save_path: str = "lab_results.png"):
        """Plot experiment results."""
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        names = list(self.results.keys())
        times = list(self.results.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(names)), times, color=['red'] + ['blue']*(len(names)-1))
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.ylabel('Time (ms)')
        plt.title('CUDA Streams Pipeline Performance')
        
        # Add speedup annotations
        baseline = times[0]
        for i, (name, time_val) in enumerate(zip(names, times)):
            if i > 0:
                speedup = baseline / time_val
                plt.annotate(f'{speedup:.2f}x', 
                            xy=(i, time_val), 
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nResults plotted to: {save_path}")


def run_lab():
    """Main lab runner."""
    print("="*60)
    print("LAB EXERCISE: Streaming Data Pipeline")
    print("Day 8: CUDA Streams and Concurrency")
    print("="*60)
    
    lab = DataPipelineLab(data_size=20_000_000)
    results = lab.run_experiments()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, time_ms in results.items():
        print(f"  {name}: {time_ms:.2f} ms")


if __name__ == "__main__":
    run_lab()
```

#### Step 3: Profile with Nsight Systems

```bash
# Profile the application
nsys profile --stats=true python src/stream_basics.py

# Generate timeline visualization
nsys profile -o stream_profile python src/stream_basics.py
nsys export -t sqlite stream_profile.nsys-rep
```

### Expected Output
```
============================================================
RUNNING LAB EXPERIMENTS
============================================================

1. Sequential Baseline:
   Time: 125.45 ms

2. Overlapped (2 streams):
   Time: 78.32 ms
   Speedup: 1.60x

2. Overlapped (3 streams):
   Time: 68.21 ms
   Speedup: 1.84x

2. Overlapped (4 streams):
   Time: 64.15 ms
   Speedup: 1.96x

2. Overlapped (8 streams):
   Time: 63.89 ms
   Speedup: 1.96x

============================================================
FINAL RESULTS
============================================================
  sequential: 125.45 ms
  overlapped_2_streams: 78.32 ms
  overlapped_3_streams: 68.21 ms
  overlapped_4_streams: 64.15 ms
  overlapped_8_streams: 63.89 ms
```

### Challenge Extensions

1. **Beginner:** Add timing events to measure individual H2D, Compute, D2H times
2. **Intermediate:** Implement callback functions to process results as they complete
3. **Advanced:** Implement a producer-consumer pattern with separate threads for H2D, Compute, and D2H

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### Issue 1: No Overlap Observed
**Symptom:** Profiler shows sequential execution despite multiple streams
**Causes:**
- Using pageable instead of pinned memory
- Using synchronous cudaMemcpy instead of cudaMemcpyAsync
- Default stream synchronization behavior

**Solution:**
```python
# Wrong: Pageable memory
h_data = np.zeros(N, dtype=np.float32)

# Right: Use pinned memory in CUDA C++
# In Python/CuPy, use the stream context properly
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    d_data = cp.asarray(h_data)  # Async in this stream
```

#### Issue 2: Race Conditions
**Symptom:** Incorrect results, data corruption
**Causes:**
- Missing synchronization between dependent operations
- Reusing buffers before previous operations complete

**Solution:**
```python
# Use events for explicit synchronization
event = cp.cuda.Event()

with stream1:
    # ... operations
    event.record()

with stream2:
    stream2.wait_event(event)  # Wait for stream1
    # ... dependent operations
```

#### Issue 3: Memory Allocation Slowdowns
**Symptom:** First iteration much slower than subsequent ones
**Causes:**
- GPU memory allocation is slow
- Memory pools not warmed up

**Solution:**
```python
# Pre-allocate all buffers before timing
d_buffers = [cp.zeros(size, dtype=cp.float32) for _ in range(num_streams)]

# Warmup iteration
for buf in d_buffers:
    buf.fill(0)
cp.cuda.Stream.null.synchronize()

# Now measure
```

### Debugging Tools

#### Using Nsight Systems for Stream Analysis
```bash
# Capture CUDA activity
nsys profile \
    --trace cuda,nvtx \
    --cuda-memory-usage true \
    --output stream_debug \
    python my_script.py

# View in Nsight Systems GUI
nsys-ui stream_debug.qdrep
```

#### Adding NVTX Markers for Labeling
```python
import cupy as cp

# Add markers for profiling
with cp.cuda.nvtx.Range("H2D Transfer", color="blue"):
    d_data = cp.asarray(h_data)

with cp.cuda.nvtx.Range("Compute", color="green"):
    result = cp.sin(d_data)

with cp.cuda.nvtx.Range("D2H Transfer", color="red"):
    h_result = result.get()
```

---

## ⚡ Performance Optimization

### Optimization Strategies

#### 1. Choose Optimal Number of Streams
```python
# Rule of thumb: 2-4 streams for compute-bound
#                4-8 streams for memory-bound

def find_optimal_streams(data_size, max_streams=8):
    """Empirically find best stream count."""
    times = {}
    for n in range(1, max_streams + 1):
        times[n] = benchmark_with_streams(data_size, n)
    
    optimal = min(times, key=times.get)
    return optimal
```

#### 2. Match Chunk Size to Transfer Overhead
```python
# Too small chunks: Transfer overhead dominates
# Too large chunks: Less overlap opportunity

# Sweet spot depends on:
# - GPU memory bandwidth
# - PCIe bandwidth
# - Compute intensity

# Typical range: 1MB - 64MB per chunk
```

#### 3. Use Pinned Memory Pools
```cpp
// C++ example: Pinned memory pool
class PinnedMemoryPool {
    std::vector<void*> pool;
    size_t chunk_size;
    
public:
    void* allocate() {
        if (pool.empty()) {
            void* ptr;
            cudaMallocHost(&ptr, chunk_size);
            return ptr;
        }
        void* ptr = pool.back();
        pool.pop_back();
        return ptr;
    }
    
    void deallocate(void* ptr) {
        pool.push_back(ptr);
    }
};
```

### Benchmarking Framework

```python
def comprehensive_benchmark():
    """
    Comprehensive benchmark for stream optimization.
    
    Tests various configurations and reports:
    - Throughput (GB/s)
    - Latency (ms)
    - Overlap efficiency
    """
    configs = [
        {"streams": 1, "chunk_mb": 16},
        {"streams": 2, "chunk_mb": 16},
        {"streams": 4, "chunk_mb": 16},
        {"streams": 4, "chunk_mb": 8},
        {"streams": 4, "chunk_mb": 32},
    ]
    
    results = []
    for config in configs:
        metrics = run_benchmark(**config)
        results.append({
            **config,
            "throughput_gb_s": metrics.throughput,
            "latency_ms": metrics.latency,
            "efficiency": metrics.efficiency
        })
    
    # Find best configuration
    best = max(results, key=lambda x: x["throughput_gb_s"])
    return results, best
```

---

## 🏭 Production Considerations

### Deployment Checklist
- [ ] Use pinned memory for all host buffers in the critical path
- [ ] Pre-allocate GPU buffers before processing
- [ ] Implement proper stream cleanup on errors
- [ ] Add timeout handling for stream synchronization
- [ ] Monitor GPU utilization and memory usage
- [ ] Log stream performance metrics

### Error Handling in Production

```python
class ProductionPipeline:
    """Production-ready pipeline with proper error handling."""
    
    def __init__(self):
        self.streams = []
        self.is_initialized = False
    
    def initialize(self):
        """Initialize with proper error handling."""
        try:
            self.streams = [cp.cuda.Stream(non_blocking=True) 
                          for _ in range(4)]
            self.is_initialized = True
            return True
        except cp.cuda.runtime.CUDARuntimeError as e:
            logging.error(f"Failed to create streams: {e}")
            return False
    
    def process(self, data, timeout_ms=5000):
        """Process with timeout."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        # Record start
        start_event = cp.cuda.Event()
        start_event.record()
        
        # Submit work...
        
        # Wait with timeout
        import time
        start_time = time.time()
        while True:
            status = self.streams[0].query()
            if status == 0:  # Complete
                break
            if (time.time() - start_time) * 1000 > timeout_ms:
                raise TimeoutError(f"Stream operation timed out after {timeout_ms}ms")
            time.sleep(0.001)
    
    def cleanup(self):
        """Proper cleanup."""
        for stream in self.streams:
            try:
                stream.synchronize()
            except:
                pass
        self.streams.clear()
        self.is_initialized = False
```

---

## 📚 Further Reading

### Documentation
- [CUDA C++ Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [CUDA Best Practices Guide - Asynchronous Operations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)
- [CuPy Stream Documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)

### Papers
- "Optimizing Memory Bound Computations with GPU Memory Hierarchies" - NVIDIA Research
- "Concurrent Kernel Execution on GPUs" - Technical Report

### Video Resources
- [GTC: Advanced CUDA Programming](https://www.youtube.com/results?search_query=GTC+CUDA+streams)
- [CUDA Optimization Tutorial Series](https://www.youtube.com/results?search_query=CUDA+optimization+streams)

---

## 🔗 Connections

### Previous Day
- **Day 7:** Week 1 Review & Matrix Multiplication Project
- How it connects: Streams enable overlapping matrix multiplications in batched operations

### Next Day  
- **Day 9:** CUDA Events and Timing
- What to expect: Deep dive into precise timing and inter-stream synchronization

### Related Topics
- Week 4: TensorRT uses streams for pipeline optimization
- Week 13: Ray uses streams for concurrent GPU operations
- Week 15: Distributed training leverages streams for gradient synchronization

---

## 📝 Daily Summary

### Key Takeaways
1. **Streams enable concurrency** - Operations in different streams can overlap
2. **Default stream synchronizes** - Use explicit streams for maximum parallelism
3. **Pinned memory is required** - For async transfers and overlap
4. **Double/Triple buffering** - Fundamental pattern for hiding transfer latency
5. **Profile to verify overlap** - Use Nsight Systems to visualize execution timeline
6. **More streams != always better** - Diminishing returns after 3-4 streams typically

### Commands Reference
```bash
# Check GPU async capabilities
nvidia-smi --query-gpu=name,compute_cap,async_engines --format=csv

# Profile streams
nsys profile --trace cuda python my_script.py

# Enable MPS for multi-process
nvidia-cuda-mps-control -d
```

### Key Code Patterns
```python
# Create stream
stream = cp.cuda.Stream(non_blocking=True)

# Use stream context
with stream:
    d_data = cp.asarray(h_data)  # Async H2D
    result = compute(d_data)      # Compute
    h_result = result.get()       # Async D2H

# Synchronize specific stream
stream.synchronize()

# Create events for timing
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record(stream)
# ... operations ...
end.record(stream)
end.synchronize()
elapsed = cp.cuda.get_elapsed_time(start, end)
```

---

**Day 8 Complete** ✅

*Next: Day 9 - CUDA Events and Timing - Master precise GPU performance measurement!*
