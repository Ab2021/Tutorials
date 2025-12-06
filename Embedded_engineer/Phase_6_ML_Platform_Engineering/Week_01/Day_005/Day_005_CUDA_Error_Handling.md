# Day 5: CUDA Error Handling & Debugging
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Professional error handling, debugging tools, and common pitfalls.

---

## 🎯 Learning Objectives
1. **Implement** robust CUDA error checking
2. **Use** cuda-memcheck and compute-sanitizer
3. **Profile** with Nsight Systems
4. **Debug** common CUDA errors

---

## 📖 Theoretical Foundation

### Error Handling Strategy

```cpp
// CUDA_CHECK macro - Essential for production code
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA Error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, error,                          \
                    cudaGetErrorName(error),                            \
                    cudaGetErrorString(error));                         \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Check after kernel launch
kernelFunction<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());       // Check synchronous errors
CUDA_CHECK(cudaDeviceSynchronize());  // Force sync and check async errors
```

### Common Error Types

| Error | Meaning | Solution |
|-------|---------|----------|
| `cudaErrorInvalidConfiguration` | Invalid launch parameters | Check grid/block dimensions |
| `cudaErrorMemoryAllocation` | cudaMalloc failed | Check available memory |
| `cudaErrorLaunchOutOfResources` | Too many resources | Reduce block size/registers |
| `cudaErrorIllegalAddress` | Invalid memory access | Use compute-sanitizer |
| `cudaErrorAssert` | Device assertion failed | Check __assert in kernel |

---

## 💻 Implementation

### Robust Error Handling Framework

```python
#!/usr/bin/env python3
"""Day 5: CUDA Error Handling in Python"""

import cupy as cp
import traceback
from functools import wraps

class CUDADebugger:
    """Helper class for CUDA debugging"""
    
    @staticmethod
    def check_memory():
        """Print current GPU memory status"""
        mempool = cp.get_default_memory_pool()
        print(f"GPU Memory - Used: {mempool.used_bytes()/1e9:.2f} GB, "
              f"Total: {mempool.total_bytes()/1e9:.2f} GB")
    
    @staticmethod
    def safe_sync():
        """Synchronize and check for errors"""
        try:
            cp.cuda.Stream.null.synchronize()
            return True
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"CUDA Sync Error: {e}")
            return False
    
    @staticmethod
    def gpu_guard(func):
        """Decorator for catching GPU errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                cp.cuda.Stream.null.synchronize()
                return result
            except cp.cuda.runtime.CUDARuntimeError as e:
                print(f"CUDA Error in {func.__name__}: {e}")
                CUDADebugger.check_memory()
                traceback.print_exc()
                raise
        return wrapper

# Example usage
@CUDADebugger.gpu_guard
def risky_operation():
    # Intentionally allocate too much memory
    try:
        huge_array = cp.zeros((100000, 100000, 100), dtype=cp.float32)
    except cp.cuda.memory.OutOfMemoryError:
        print("Caught OutOfMemoryError - as expected")
        raise

if __name__ == "__main__":
    print("Testing CUDA error handling...")
    CUDADebugger.check_memory()
    
    try:
        risky_operation()
    except:
        print("Error handled gracefully")
```

### Debugging Kernel with Print

```cpp
__global__ void debugKernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Debug: Print first few thread indices
    if (tid < 5) {
        printf("Thread %d processing data[%d] = %f\n", 
               tid, tid, data[tid]);
    }
    
    // Assert: Check preconditions
    assert(N > 0 && "N must be positive");
    
    if (tid < N) {
        data[tid] *= 2.0f;
    }
}
```

---

## 🔬 Lab Exercise: "Debug Challenge"

### Task
Fix the following buggy kernel:

```cpp
// Bug 1: Race condition
// Bug 2: Out of bounds access
// Bug 3: Missing synchronization
__global__ void buggyKernel(float* data, int N) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[tid] = data[idx];  // Bug: No bounds check
    
    // Bug: Missing __syncthreads()
    
    if (tid > 0) {
        shared[tid] += shared[tid - 1];  // Race condition
    }
    
    data[idx] = shared[tid];  // Bug: No bounds check
}
```

---

## 🛠️ Debugging Tools

### Command-Line Tools
```bash
# Compile with debug symbols
nvcc -g -G -o program program.cu

# Memory checking
compute-sanitizer --tool memcheck ./program

# Race detection
compute-sanitizer --tool racecheck ./program

# Initialize detection
compute-sanitizer --tool initcheck ./program
```

---

## 📝 Daily Summary

### Key Takeaways
1. Always use CUDA_CHECK macro for all CUDA API calls
2. Check errors after kernel launches with `cudaGetLastError()`
3. Use `cudaDeviceSynchronize()` to catch asynchronous errors
4. compute-sanitizer is essential for memory error detection
5. Debug prints with `printf` work in kernels (but are slow)

---

**Day 5 Complete** ✅
