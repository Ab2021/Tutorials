# Day 3: CUDA Memory Management
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Master GPU memory allocation, transfer patterns, and optimization strategies.

---

## 🎯 Learning Objectives
1. **Implement** explicit memory management with cudaMalloc/cudaMemcpy
2. **Use** Unified Memory for simplified programming
3. **Optimize** with pinned (page-locked) memory
4. **Benchmark** memory transfer speeds

---

## 📖 Theoretical Foundation

### Memory Types and APIs

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Management APIs                    │
├─────────────────┬───────────────────────────────────────────┤
│ Operation       │ Function                                   │
├─────────────────┼───────────────────────────────────────────┤
│ Device Alloc    │ cudaMalloc(void** ptr, size_t size)       │
│ Device Free     │ cudaFree(void* ptr)                       │
│ Host→Device     │ cudaMemcpy(dst, src, size, H2D)           │
│ Device→Host     │ cudaMemcpy(dst, src, size, D2H)           │
│ Device→Device   │ cudaMemcpy(dst, src, size, D2D)           │
│ Unified Memory  │ cudaMallocManaged(void** ptr, size_t size)│
│ Pinned Host     │ cudaMallocHost(void** ptr, size_t size)   │
│ Async Copy      │ cudaMemcpyAsync(dst, src, size, kind, stream)│
└─────────────────┴───────────────────────────────────────────┘
```

### Pinned vs Pageable Memory

```
Pageable Memory (Default):
CPU RAM ──┬──▶ Staging Buffer ──▶ DMA ──▶ GPU Memory
          └── Page fault possible!

Pinned Memory:
CPU RAM ────────────────────▶ DMA ──▶ GPU Memory
         Direct transfer, no staging
```

**Benefits of Pinned Memory:**
- ~2x faster transfers
- Required for async transfers
- Can be mapped to GPU address space

**Drawbacks:**
- Limited system resource
- Cannot be swapped to disk
- Reduce available system memory

---

## 💻 Implementation

### Complete Memory Management Example

```cpp
// memory_management.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE (1 << 24)  // 16M elements
#define BYTES (SIZE * sizeof(float))

void benchmarkTransfer(const char* name, float* h_ptr, float* d_ptr, 
                       cudaMemcpyKind kind, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    cudaMemcpy(d_ptr, h_ptr, BYTES/4, kind);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_ptr, h_ptr, BYTES, kind);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float gb = (float)BYTES * iterations / 1e9;
    float bandwidth = gb / (ms / 1000.0f);
    
    printf("%s: %.2f GB/s (%.2f ms for %d iterations)\n", 
           name, bandwidth, ms, iterations);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    float *h_pageable, *h_pinned, *d_data;
    
    // Allocate pageable host memory (default)
    h_pageable = (float*)malloc(BYTES);
    
    // Allocate pinned host memory
    cudaMallocHost(&h_pinned, BYTES);
    
    // Allocate device memory
    cudaMalloc(&d_data, BYTES);
    
    // Initialize data
    for (int i = 0; i < SIZE; i++) {
        h_pageable[i] = h_pinned[i] = (float)i;
    }
    
    printf("=== Memory Transfer Benchmark (%.2f MB) ===\n", BYTES / 1e6);
    
    // Benchmark pageable memory
    benchmarkTransfer("Pageable H2D", h_pageable, d_data, 
                      cudaMemcpyHostToDevice, 10);
    
    // Benchmark pinned memory
    benchmarkTransfer("Pinned H2D", h_pinned, d_data, 
                      cudaMemcpyHostToDevice, 10);
    
    // Cleanup
    free(h_pageable);
    cudaFreeHost(h_pinned);
    cudaFree(d_data);
    
    return 0;
}
```

### Unified Memory Example

```python
#!/usr/bin/env python3
"""Day 3: Unified Memory with CuPy"""

import cupy as cp
import numpy as np

# Unified memory concept in Python/CuPy
# CuPy arrays automatically handle transfers

def unified_memory_example():
    N = 1 << 20
    
    # GPU array (automatically uses managed memory concepts)
    x_gpu = cp.arange(N, dtype=cp.float32)
    
    # Perform GPU computation
    y_gpu = cp.sin(x_gpu) ** 2 + cp.cos(x_gpu) ** 2
    
    # Access on CPU (implicit transfer)
    y_cpu = y_gpu.get()  # Explicit transfer
    
    # Verify (should be ~1.0 for all elements)
    print(f"Mean value: {np.mean(y_cpu):.6f} (expected: 1.0)")

def memory_pools():
    """CuPy memory pool management"""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    print(f"Used GPU memory: {mempool.used_bytes() / 1e6:.2f} MB")
    print(f"Total GPU memory: {mempool.total_bytes() / 1e6:.2f} MB")
    
    # Allocate some arrays
    arrays = [cp.zeros(1 << 20) for _ in range(10)]
    print(f"After allocation: {mempool.used_bytes() / 1e6:.2f} MB")
    
    # Free memory
    del arrays
    cp.get_default_memory_pool().free_all_blocks()
    print(f"After free: {mempool.used_bytes() / 1e6:.2f} MB")

if __name__ == "__main__":
    unified_memory_example()
    print()
    memory_pools()
```

---

## 🔬 Lab Exercise: "Memory Transfer Optimization"

### Task
1. Measure baseline transfer speed with pageable memory
2. Compare with pinned memory
3. Implement overlapped transfers with streams (preview of Day 8)

### Benchmark Script

```python
import cupy as cp
import numpy as np
import time

def benchmark_transfers(size_mb=100):
    size = size_mb * 1024 * 1024 // 4  # float32 elements
    
    # Pageable memory (NumPy)
    h_pageable = np.random.randn(size).astype(np.float32)
    
    # Transfer and time
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    d_array = cp.asarray(h_pageable)  # H2D
    end.record()
    end.synchronize()
    
    h2d_time = cp.cuda.get_elapsed_time(start, end)
    h2d_bw = (size * 4 / 1e9) / (h2d_time / 1000)
    
    print(f"Host→Device: {h2d_bw:.2f} GB/s ({h2d_time:.2f} ms)")
    
    start.record()
    h_result = d_array.get()  # D2H
    end.record()
    end.synchronize()
    
    d2h_time = cp.cuda.get_elapsed_time(start, end)
    d2h_bw = (size * 4 / 1e9) / (d2h_time / 1000)
    
    print(f"Device→Host: {d2h_bw:.2f} GB/s ({d2h_time:.2f} ms)")

benchmark_transfers(100)
```

---

## 📝 Daily Summary

### Key Takeaways
1. Use `cudaMalloc`/`cudaFree` for device memory
2. Pinned memory (`cudaMallocHost`) provides ~2x faster transfers
3. Unified Memory (`cudaMallocManaged`) simplifies programming but may have performance implications
4. Always consider memory transfer overhead in performance analysis
5. CuPy provides automatic memory management with pools

---

**Day 3 Complete** ✅
