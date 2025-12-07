# Day 8: CUDA Streams and Concurrency - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Stream Basics (Beginner)

### Objective
Create and use multiple CUDA streams.

### Code

```python
import torch
import time

def stream_basics():
    """Demonstrate basic stream operations."""
    
    # Create streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    print("Stream Basics Demo")
    print("=" * 40)
    
    # Default stream
    print(f"Default stream: {torch.cuda.current_stream()}")
    print(f"Stream 1: {stream1}")
    print(f"Stream 2: {stream2}")
    
    # Run operations on different streams
    a = torch.randn(1000, 1000, device='cuda')
    
    with torch.cuda.stream(stream1):
        b = a @ a
        print("Matrix multiply on stream1")
    
    with torch.cuda.stream(stream2):
        c = a + a
        print("Addition on stream2")
    
    # Synchronize all streams
    torch.cuda.synchronize()
    print("All operations complete")

if __name__ == "__main__":
    stream_basics()
```

---

## 🔬 Exercise 2: Overlapping Compute and Transfer (Intermediate)

### Objective
Pipeline data transfers with computation.

### Code

```python
import torch
import time

def overlapped_transfers():
    """Overlap H2D transfer with computation."""
    
    num_chunks = 4
    chunk_size = 10_000_000
    
    # Create pinned memory for faster transfers
    cpu_data = [torch.randn(chunk_size).pin_memory() for _ in range(num_chunks)]
    gpu_buffers = [torch.empty(chunk_size, device='cuda') for _ in range(num_chunks)]
    results = [torch.empty(chunk_size, device='cuda') for _ in range(num_chunks)]
    
    # Create streams
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    
    print("Pipelined Execution")
    print("=" * 40)
    
    # Sequential baseline
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(num_chunks):
        gpu_buffers[i].copy_(cpu_data[i])
        results[i] = gpu_buffers[i] * 2 + 1
    
    torch.cuda.synchronize()
    sequential_time = time.perf_counter() - start
    
    # Pipelined
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(num_chunks):
        # Transfer chunk i
        with torch.cuda.stream(transfer_stream):
            gpu_buffers[i].copy_(cpu_data[i])
        
        # Compute on chunk i-1 (if available)
        if i > 0:
            with torch.cuda.stream(compute_stream):
                # Wait for transfer of i-1 to complete
                compute_stream.wait_stream(transfer_stream)
                results[i-1] = gpu_buffers[i-1] * 2 + 1
    
    # Process last chunk
    with torch.cuda.stream(compute_stream):
        compute_stream.wait_stream(transfer_stream)
        results[-1] = gpu_buffers[-1] * 2 + 1
    
    torch.cuda.synchronize()
    pipelined_time = time.perf_counter() - start
    
    print(f"Sequential: {sequential_time*1000:.2f} ms")
    print(f"Pipelined:  {pipelined_time*1000:.2f} ms")
    print(f"Speedup:    {sequential_time/pipelined_time:.2f}x")

if __name__ == "__main__":
    overlapped_transfers()
```

---

## 🔬 Exercise 3: Multi-Stream Profiling (Advanced)

### Objective
Visualize stream execution timeline.

### Code

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_streams():
    """Profile multi-stream execution."""
    
    streams = [torch.cuda.Stream() for _ in range(4)]
    tensors = [torch.randn(2000, 2000, device='cuda') for _ in range(4)]
    
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        
        # Launch work on all streams
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                for _ in range(5):
                    tensors[i] = tensors[i] @ tensors[i]
        
        torch.cuda.synchronize()
    
    print("Stream Execution Profile")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    # Export for visualization
    prof.export_chrome_trace("streams_trace.json")
    print("\nTrace exported to streams_trace.json")
    print("Open in chrome://tracing to visualize")

if __name__ == "__main__":
    profile_streams()
```

---

## 🐛 Common Issues

### Issue: Operations not overlapping
**Cause:** Default stream synchronizes with all other streams
**Fix:** Use explicit non-default streams for all operations

### Issue: Race condition
**Fix:** Use `stream.wait_stream()` for dependencies

---

## 📚 Additional Resources
- [CUDA Streams Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)
