# Day 3: CUDA Memory Management - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Memory Allocation Patterns (Beginner)

### Objective
Compare different memory allocation strategies.

### Code

```python
import torch
import time

def compare_allocation_methods():
    """Compare different allocation strategies."""
    size = (1000, 1000)
    iterations = 100
    
    results = {}
    
    # Method 1: Regular allocation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        t = torch.zeros(size, device='cuda')
        del t
    torch.cuda.synchronize()
    results['zeros'] = (time.perf_counter() - start) / iterations * 1000
    
    # Method 2: Empty (uninitialized)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        t = torch.empty(size, device='cuda')
        del t
    torch.cuda.synchronize()
    results['empty'] = (time.perf_counter() - start) / iterations * 1000
    
    # Method 3: Pre-allocated buffer
    buffer = torch.empty(size, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        buffer.zero_()
    torch.cuda.synchronize()
    results['preallocated'] = (time.perf_counter() - start) / iterations * 1000
    
    print("Allocation Time Comparison (ms):")
    print("=" * 40)
    for method, time_ms in results.items():
        print(f"  {method}: {time_ms:.4f} ms")

if __name__ == "__main__":
    compare_allocation_methods()
```

### Expected Output
```
Allocation Time Comparison (ms):
========================================
  zeros: 0.0523 ms
  empty: 0.0089 ms
  preallocated: 0.0012 ms
```

---

## 🔬 Exercise 2: Pinned Memory Transfer (Intermediate)

### Objective
Measure the speedup from using pinned (page-locked) memory.

### Code

```python
import torch
import time

def benchmark_pinned_memory(size_mb=100):
    """Compare pinned vs pageable memory transfer speed."""
    size = size_mb * 1024 * 1024 // 4  # float32
    iterations = 10
    
    # Regular (pageable) memory
    cpu_pageable = torch.randn(size)
    
    # Pinned memory
    cpu_pinned = torch.randn(size).pin_memory()
    
    results = {}
    
    # Pageable -> GPU
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        gpu = cpu_pageable.cuda()
        torch.cuda.synchronize()
    results['pageable_H2D'] = size_mb / ((time.perf_counter() - start) / iterations)
    
    # Pinned -> GPU
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        gpu = cpu_pinned.cuda()
        torch.cuda.synchronize()
    results['pinned_H2D'] = size_mb / ((time.perf_counter() - start) / iterations)
    
    print(f"Memory Transfer Benchmark ({size_mb} MB)")
    print("=" * 40)
    print(f"  Pageable -> GPU: {results['pageable_H2D']:.2f} GB/s")
    print(f"  Pinned -> GPU:   {results['pinned_H2D']:.2f} GB/s")
    print(f"  Speedup: {results['pinned_H2D']/results['pageable_H2D']:.2f}x")

if __name__ == "__main__":
    benchmark_pinned_memory(500)
```

---

## 🔬 Exercise 3: Memory Pool Management (Advanced)

### Objective
Understand PyTorch's caching allocator behavior.

### Code

```python
import torch

def analyze_memory_allocator():
    """Analyze PyTorch CUDA memory allocator behavior."""
    
    print("CUDA Memory Allocator Analysis")
    print("=" * 50)
    
    # Initial state
    print("\n1. Initial State:")
    print(torch.cuda.memory_summary(abbreviated=True))
    
    # Allocate tensors
    tensors = []
    for i in range(5):
        t = torch.randn(1000, 1000, device='cuda')
        tensors.append(t)
    
    print("\n2. After allocating 5 tensors (1000x1000):")
    stats = torch.cuda.memory_stats()
    print(f"  Allocated: {stats['allocated_bytes.all.current'] / 1e6:.2f} MB")
    print(f"  Cached:    {stats['reserved_bytes.all.current'] / 1e6:.2f} MB")
    
    # Delete tensors
    del tensors
    
    print("\n3. After deleting tensors (before empty_cache):")
    stats = torch.cuda.memory_stats()
    print(f"  Allocated: {stats['allocated_bytes.all.current'] / 1e6:.2f} MB")
    print(f"  Cached:    {stats['reserved_bytes.all.current'] / 1e6:.2f} MB")
    
    # Empty cache
    torch.cuda.empty_cache()
    
    print("\n4. After empty_cache():")
    stats = torch.cuda.memory_stats()
    print(f"  Allocated: {stats['allocated_bytes.all.current'] / 1e6:.2f} MB")
    print(f"  Cached:    {stats['reserved_bytes.all.current'] / 1e6:.2f} MB")

if __name__ == "__main__":
    analyze_memory_allocator()
```

---

## 🐛 Common Errors & Fixes

### Error 1: CUDA out of memory
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Fix:** 
```python
torch.cuda.empty_cache()
# Or reduce batch size
# Or use gradient checkpointing
```

### Error 2: Memory leak
**Symptom:** Memory grows over training loop
**Fix:** Ensure you're not storing tensors in lists:
```python
# Bad
losses.append(loss)  # Stores computation graph!

# Good
losses.append(loss.item())  # Stores only the value
```

---

## 📚 Additional Resources
- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [CUDA Memory Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
