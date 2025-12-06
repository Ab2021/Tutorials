# Day 11: Dynamic Parallelism
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 2: Advanced CUDA Programming

---

> **🎯 Focus Area:** Master the art of launching kernels from within kernels for adaptive algorithms and recursive computations on the GPU.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** dynamic parallelism concepts and hardware requirements
2. **Implement** nested kernel launches from device code
3. **Manage** memory and synchronization in dynamic parallelism
4. **Apply** dynamic parallelism to adaptive algorithms
5. **Evaluate** when dynamic parallelism is beneficial vs alternatives
6. **Debug** common dynamic parallelism issues

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with compute capability 3.5 or higher
- Sufficient global memory for nested grids

### Software Environment
```bash
# Compile with dynamic parallelism support
nvcc -arch=sm_35 -rdc=true -lcudadevrt program.cu -o program

# For CuPy, dynamic parallelism has limited support
# Custom kernels in CUDA C++ recommended
```

### Prior Knowledge
- Day 2: CUDA Programming Model
- Day 4: Thread Synchronization

---

## 📖 Theoretical Foundation

### 1. What is Dynamic Parallelism?

Dynamic parallelism allows **GPU kernels to launch other GPU kernels** without returning to the CPU. This enables:

- **Recursive algorithms** on GPU
- **Adaptive algorithms** that adjust parallelism based on data
- **Reduced CPU-GPU communication** overhead

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRADITIONAL vs DYNAMIC PARALLELISM                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRADITIONAL (CPU orchestration):                                            │
│                                                                              │
│  CPU ──────────────────────────────────────────────────────────────────▶    │
│      │ launch      ↑ sync    │ launch      ↑ sync    │ launch    │          │
│      ▼             │         ▼             │         ▼           │          │
│  GPU ────kernel1───┴─────────kernel2───────┴─────────kernel3─────▶          │
│                                                                              │
│  High latency due to round-trips to CPU for each launch                     │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────          │
│                                                                              │
│  DYNAMIC PARALLELISM (GPU orchestration):                                    │
│                                                                              │
│  CPU ─────────────────────────────────────────────────────────────▶         │
│      │ launch                                   ↑ sync                       │
│      ▼                                          │                            │
│  GPU ────parent_kernel─────────────────────────────────────────────▶        │
│              │ launch         │ launch                                       │
│              ▼                ▼                                              │
│          child1─────────  child2──────────                                   │
│           │launch          │launch                                           │
│           ▼                ▼                                                 │
│         grandchild1     grandchild2                                          │
│                                                                              │
│  All orchestration happens on GPU - much lower latency                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Memory Model for Dynamic Parallelism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY VISIBILITY IN DYNAMIC PARALLELISM                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Global Memory: Visible to all threads (parent and child)                    │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  IMPORTANT: No automatic coherence between parent and child!│            │
│  │  Must use cudaDeviceSynchronize() in device code           │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
│  Shared Memory: NOT visible to child kernels (different block)              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Parent's shared memory is not accessible by children       │            │
│  │  Children have their own shared memory allocations          │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
│  Local Memory: Thread-private, not visible to children                       │
│                                                                              │
│  Constant Memory: Visible to all (same content)                              │
│                                                                              │
│  Texture Memory: Visible to all (same bindings)                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Synchronization Rules

```cpp
// In device code (__global__ or __device__ function)

// Launch child kernel
childKernel<<<grid, block>>>(args);

// CRITICAL: Synchronize to ensure child completes
// Before accessing results written by child
cudaDeviceSynchronize();  // Device-side synchronization

// Now safe to use results from child kernel
```

**Key Rules:**
1. Child kernels are asynchronous (return immediately)
2. Use `cudaDeviceSynchronize()` in device code to wait
3. Parent thread block can launch multiple children
4. All child grids must complete before parent completes

### 4. Nesting Depth and Resource Limits

```cpp
// Maximum nesting depth is device-dependent (typically 24)
// Can query with:
int maxDepth;
cudaDeviceGetAttribute(&maxDepth, 
    cudaDevAttrMaxNestedParallelismLevel, 0);

// Resource limits per grid level
// Each level consumes device memory for grid management
// Deep nesting can exhaust resources

// Best practice: limit depth, use iterative when possible
```

### 5. Streams in Dynamic Parallelism

```cpp
__global__ void parentKernel() {
    // Create stream in device code
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    // Launch child in specific stream
    childKernel<<<grid, block, 0, stream>>>(args);
    
    // Wait for specific stream
    cudaStreamSynchronize(stream);
    
    // Destroy stream
    cudaStreamDestroy(stream);
}
```

---

## 💻 Implementation

### 👨‍💻 Core Implementation

#### 📁 `dynamic_parallelism.cu` - Full CUDA Implementation
```cpp
/*
 * Day 11: Dynamic Parallelism
 * Phase 6: AI/ML Platform Engineering with GPU Programming
 *
 * Compile with: nvcc -arch=sm_60 -rdc=true -lcudadevrt dynamic_parallelism.cu -o dp_demo
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", \
                   cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)


// ============================================================================
// Example 1: Basic Nested Kernel Launch
// ============================================================================

__global__ void childKernel(int* data, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void parentKernel(int* data, int n, int numChildren) {
    // Each thread launches a child kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numChildren) {
        int chunkSize = n / numChildren;
        int offset = tid * chunkSize;
        int* childData = data + offset;
        
        // Launch child kernel
        int blockSize = 256;
        int gridSize = (chunkSize + blockSize - 1) / blockSize;
        
        childKernel<<<gridSize, blockSize>>>(childData, tid, chunkSize);
    }
    
    // Synchronize to ensure all children complete
    __syncthreads();  // First sync threads in block
    
    if (threadIdx.x == 0) {
        cudaDeviceSynchronize();  // Then wait for children
    }
    __syncthreads();  // Sync again after device sync
}


// ============================================================================
// Example 2: Recursive Quicksort (Classic Dynamic Parallelism Example)
// ============================================================================

__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

__global__ void quicksortKernel(int* arr, int low, int high, int depth) {
    // Limit recursion depth to avoid resource exhaustion
    const int MAX_DEPTH = 16;
    const int MIN_SIZE = 32;  // Below this, use sequential sort
    
    if (low >= high) return;
    
    int size = high - low + 1;
    
    // For small arrays, use sequential in-place sort
    if (size <= MIN_SIZE || depth >= MAX_DEPTH) {
        // Simple insertion sort for small arrays
        for (int i = low + 1; i <= high; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= low && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
        return;
    }
    
    // Partition the array
    int pivotIdx = partition(arr, low, high);
    
    // Launch child kernels for left and right partitions
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    
    // Sort left partition
    if (pivotIdx - 1 > low) {
        quicksortKernel<<<1, 1, 0, stream1>>>(arr, low, pivotIdx - 1, depth + 1);
    }
    
    // Sort right partition
    if (pivotIdx + 1 < high) {
        quicksortKernel<<<1, 1, 0, stream2>>>(arr, pivotIdx + 1, high, depth + 1);
    }
    
    // Wait for both partitions to complete
    cudaDeviceSynchronize();
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}


// ============================================================================
// Example 3: Adaptive Mesh Refinement (AMR) Pattern
// ============================================================================

struct Cell {
    float value;
    int level;       // Refinement level
    bool needsRefine;
};

__device__ bool checkRefinementCriteria(Cell* cells, int idx, int n) {
    // Check if cell needs refinement based on gradient
    if (idx <= 0 || idx >= n - 1) return false;
    
    float gradient = fabsf(cells[idx + 1].value - cells[idx - 1].value);
    return gradient > 0.5f && cells[idx].level < 3;  // Max 3 levels
}

__global__ void refineKernel(Cell* cells, int start, int end, int level);

__global__ void processAndRefineKernel(Cell* cells, int start, int end, int level) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < end) {
        cells[idx].level = level;
        
        // Check if this cell needs refinement
        cells[idx].needsRefine = checkRefinementCriteria(cells, idx, end);
    }
    
    __syncthreads();
    
    // Only one thread per block checks for refinement
    if (threadIdx.x == 0) {
        // Count cells needing refinement in this block
        int refineCount = 0;
        int blockStart = start + blockIdx.x * blockDim.x;
        int blockEnd = min(blockStart + blockDim.x, end);
        
        for (int i = blockStart; i < blockEnd; i++) {
            if (cells[i].needsRefine) {
                refineCount++;
            }
        }
        
        // If any cells need refinement, launch child kernel
        if (refineCount > 0 && level < 3) {
            int childBlocks = (blockEnd - blockStart + 255) / 256;
            refineKernel<<<childBlocks, 256>>>(cells, blockStart, blockEnd, level + 1);
            cudaDeviceSynchronize();
        }
    }
}

__global__ void refineKernel(Cell* cells, int start, int end, int level) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < end && cells[idx].needsRefine) {
        // Simulate refinement by updating values
        cells[idx].value = cells[idx].value * 0.5f;
        cells[idx].level = level;
        cells[idx].needsRefine = false;
    }
}


// ============================================================================
// Example 4: Octree Construction
// ============================================================================

struct Point3D {
    float x, y, z;
};

struct BoundingBox {
    float minX, maxX;
    float minY, maxY;
    float minZ, maxZ;
};

__device__ int getOctant(Point3D* point, BoundingBox* box) {
    float midX = (box->minX + box->maxX) / 2;
    float midY = (box->minY + box->maxY) / 2;
    float midZ = (box->minZ + box->maxZ) / 2;
    
    int octant = 0;
    if (point->x > midX) octant |= 1;
    if (point->y > midY) octant |= 2;
    if (point->z > midZ) octant |= 4;
    
    return octant;
}

__global__ void buildOctreeKernel(Point3D* points, int* indices, int numPoints,
                                   BoundingBox box, int depth, int maxDepth,
                                   int* nodeCount) {
    // Base case: too deep or too few points
    if (depth >= maxDepth || numPoints <= 8) {
        // This is a leaf node
        atomicAdd(nodeCount, 1);
        return;
    }
    
    // Count points in each octant
    __shared__ int octantCounts[8];
    __shared__ int octantOffsets[8];
    
    if (threadIdx.x < 8) {
        octantCounts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Count phase
    for (int i = threadIdx.x; i < numPoints; i += blockDim.x) {
        int octant = getOctant(&points[indices[i]], &box);
        atomicAdd(&octantCounts[octant], 1);
    }
    __syncthreads();
    
    // Only thread 0 launches children
    if (threadIdx.x == 0) {
        atomicAdd(nodeCount, 1);  // Count this node
        
        // Calculate offsets
        int offset = 0;
        for (int i = 0; i < 8; i++) {
            octantOffsets[i] = offset;
            offset += octantCounts[i];
        }
        
        // Launch child kernels for non-empty octants
        float midX = (box.minX + box.maxX) / 2;
        float midY = (box.minY + box.maxY) / 2;
        float midZ = (box.minZ + box.maxZ) / 2;
        
        for (int oct = 0; oct < 8; oct++) {
            if (octantCounts[oct] > 0) {
                BoundingBox childBox;
                childBox.minX = (oct & 1) ? midX : box.minX;
                childBox.maxX = (oct & 1) ? box.maxX : midX;
                childBox.minY = (oct & 2) ? midY : box.minY;
                childBox.maxY = (oct & 2) ? box.maxY : midY;
                childBox.minZ = (oct & 4) ? midZ : box.minZ;
                childBox.maxZ = (oct & 4) ? box.maxZ : midZ;
                
                // Launch child kernel
                buildOctreeKernel<<<1, 256>>>(
                    points, indices + octantOffsets[oct],
                    octantCounts[oct], childBox, depth + 1, maxDepth, nodeCount
                );
            }
        }
        
        cudaDeviceSynchronize();
    }
}


// ============================================================================
// Benchmarking: Dynamic Parallelism vs Iterative
// ============================================================================

__global__ void iterativeQuicksortKernel(int* arr, int n) {
    // Stack-based iterative quicksort
    // For comparison with recursive version
    __shared__ int stack[64];  // Stack for start/end pairs
    __shared__ int top;
    
    if (threadIdx.x == 0) {
        top = 0;
        stack[top++] = 0;
        stack[top++] = n - 1;
    }
    __syncthreads();
    
    while (top > 0) {
        __syncthreads();
        
        if (threadIdx.x == 0) {
            int high = stack[--top];
            int low = stack[--top];
            
            if (low < high) {
                int p = partition(arr, low, high);
                
                // Push larger partition first (to limit stack depth)
                if (p - 1 - low > high - p - 1) {
                    if (low < p - 1) {
                        stack[top++] = low;
                        stack[top++] = p - 1;
                    }
                    if (p + 1 < high) {
                        stack[top++] = p + 1;
                        stack[top++] = high;
                    }
                } else {
                    if (p + 1 < high) {
                        stack[top++] = p + 1;
                        stack[top++] = high;
                    }
                    if (low < p - 1) {
                        stack[top++] = low;
                        stack[top++] = p - 1;
                    }
                }
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// Main Function with Tests
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("DAY 11: DYNAMIC PARALLELISM\n");
    printf("Phase 6: AI/ML Platform Engineering with GPU Programming\n");
    printf("============================================================\n\n");
    
    // Check compute capability
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("GPU: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    
    if (props.major < 3 || (props.major == 3 && props.minor < 5)) {
        printf("ERROR: Dynamic parallelism requires compute capability 3.5+\n");
        return 1;
    }
    
    // Test 1: Basic nested kernel launch
    printf("\n--- Test 1: Basic Nested Kernel ---\n");
    {
        int n = 1024;
        int numChildren = 4;
        int* d_data;
        
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(int)));
        
        parentKernel<<<1, numChildren>>>(d_data, n, numChildren);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int* h_data = new int[n];
        CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
        
        printf("Values written by each child kernel:\n");
        for (int i = 0; i < numChildren; i++) {
            printf("  Child %d wrote: %d\n", i, h_data[i * (n / numChildren)]);
        }
        
        delete[] h_data;
        cudaFree(d_data);
    }
    
    // Test 2: Recursive Quicksort
    printf("\n--- Test 2: Recursive Quicksort ---\n");
    {
        int n = 1024;
        int* h_data = new int[n];
        int* d_data;
        
        // Initialize with random data
        srand(42);
        for (int i = 0; i < n; i++) {
            h_data[i] = rand() % 10000;
        }
        
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));
        
        // Time the sort
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        quicksortKernel<<<1, 1>>>(d_data, 0, n - 1, 0);
        cudaEventRecord(stop);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        // Verify sort
        CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
        
        bool sorted = true;
        for (int i = 1; i < n; i++) {
            if (h_data[i] < h_data[i-1]) {
                sorted = false;
                break;
            }
        }
        
        printf("Array size: %d\n", n);
        printf("Time: %.3f ms\n", ms);
        printf("Sorted correctly: %s\n", sorted ? "YES" : "NO");
        
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\n============================================================\n");
    printf("Day 11 Complete: Dynamic Parallelism\n");
    printf("============================================================\n");
    
    return 0;
}
```

#### 📁 `dynamic_parallelism_python.py` - Python Wrapper
```python
#!/usr/bin/env python3
"""
Day 11: Dynamic Parallelism (Python Interface)
Phase 6: AI/ML Platform Engineering with GPU Programming

Note: Dynamic parallelism requires CUDA C++. This module provides
Python wrappers and examples using subprocess or pre-compiled binaries.
"""

import subprocess
import tempfile
import os
import cupy as cp
import numpy as np


def check_dynamic_parallelism_support():
    """Check if GPU supports dynamic parallelism."""
    device = cp.cuda.Device()
    cc = device.compute_capability
    
    supported = cc[0] > 3 or (cc[0] == 3 and cc[1] >= 5)
    
    print(f"GPU: {device.name}")
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print(f"Dynamic Parallelism: {'Supported' if supported else 'Not Supported'}")
    
    return supported


def simulate_dynamic_parallelism():
    """
    Simulate dynamic parallelism behavior in Python.
    
    This demonstrates the concepts even though true DP requires CUDA C++.
    """
    print("\n" + "="*60)
    print("SIMULATED DYNAMIC PARALLELISM PATTERNS")
    print("="*60)
    
    # Pattern 1: Recursive-like divide and conquer
    print("\n1. Divide and Conquer Pattern:")
    
    def parallel_sum(data, threshold=1024):
        """Simulate recursive parallel sum."""
        n = len(data)
        
        if n <= threshold:
            # Base case: compute directly
            return float(cp.sum(data))
        
        # Divide: split into two halves
        mid = n // 2
        left = data[:mid]
        right = data[mid:]
        
        # Conquer: recursively sum (simulates child kernels)
        left_sum = parallel_sum(left, threshold)
        right_sum = parallel_sum(right, threshold)
        
        # Combine
        return left_sum + right_sum
    
    data = cp.random.randn(1_000_000, dtype=cp.float32)
    
    result = parallel_sum(data)
    expected = float(cp.sum(data))
    
    print(f"   Data size: {len(data):,}")
    print(f"   Result: {result:.6f}")
    print(f"   Expected: {expected:.6f}")
    print(f"   Error: {abs(result - expected):.2e}")
    
    # Pattern 2: Adaptive refinement
    print("\n2. Adaptive Refinement Pattern:")
    
    def adaptive_process(data, level=0, max_level=3):
        """Simulate adaptive mesh refinement."""
        if level >= max_level:
            return cp.mean(data)
        
        # Check if refinement needed
        variance = float(cp.var(data))
        
        if variance > 0.5:  # High variance - refine
            n = len(data)
            if n >= 4:
                # Split and process children
                chunk_size = n // 4
                results = []
                for i in range(4):
                    chunk = data[i*chunk_size:(i+1)*chunk_size]
                    results.append(adaptive_process(chunk, level + 1, max_level))
                return np.mean(results)
        
        return float(cp.mean(data))
    
    data = cp.random.randn(10000, dtype=cp.float32)
    result = adaptive_process(data)
    print(f"   Adaptive result: {result:.6f}")


def compile_and_run_cuda():
    """
    Compile and run the CUDA dynamic parallelism example.
    
    This requires nvcc to be installed.
    """
    print("\n" + "="*60)
    print("COMPILING CUDA DYNAMIC PARALLELISM CODE")
    print("="*60)
    
    cuda_code = '''
    #include <stdio.h>
    #include <cuda_runtime.h>
    
    __global__ void childKernel(int depth) {
        if (threadIdx.x == 0) {
            printf("  Child kernel at depth %d, block %d\\n", depth, blockIdx.x);
        }
    }
    
    __global__ void parentKernel(int maxDepth, int currentDepth) {
        if (threadIdx.x == 0) {
            printf("Parent kernel at depth %d, block %d\\n", currentDepth, blockIdx.x);
            
            if (currentDepth < maxDepth) {
                // Launch child kernels
                childKernel<<<2, 32>>>(currentDepth + 1);
                cudaDeviceSynchronize();
            }
        }
    }
    
    int main() {
        printf("Dynamic Parallelism Demo\\n");
        printf("========================\\n\\n");
        
        parentKernel<<<1, 1>>>(2, 0);
        cudaDeviceSynchronize();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\\n", cudaGetErrorString(err));
            return 1;
        }
        
        printf("\\nDone!\\n");
        return 0;
    }
    '''
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source file
            src_path = os.path.join(tmpdir, "dp_demo.cu")
            exe_path = os.path.join(tmpdir, "dp_demo")
            
            with open(src_path, 'w') as f:
                f.write(cuda_code)
            
            # Compile
            compile_cmd = [
                'nvcc',
                '-arch=sm_60',  # Adjust for your GPU
                '-rdc=true',
                '-lcudadevrt',
                src_path,
                '-o', exe_path
            ]
            
            print("Compiling...")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return
            
            # Run
            print("Running...\n")
            result = subprocess.run([exe_path], capture_output=True, text=True)
            print(result.stdout)
            
            if result.stderr:
                print(f"Errors: {result.stderr}")
                
    except FileNotFoundError:
        print("nvcc not found. Please install CUDA toolkit.")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main demonstration function."""
    print("="*60)
    print("DAY 11: DYNAMIC PARALLELISM")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*60)
    
    check_dynamic_parallelism_support()
    simulate_dynamic_parallelism()
    
    # Optionally compile and run CUDA code
    try:
        compile_and_run_cuda()
    except:
        print("\nSkipping CUDA compilation (nvcc not available)")
    
    print("\n" + "="*60)
    print("Day 11 Complete")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Adaptive Integration with Dynamic Parallelism"

### Lab Objectives
1. Implement adaptive numerical integration on GPU
2. Use dynamic parallelism for recursive refinement
3. Compare with non-adaptive approach

### Implementation Concept

```cpp
// Adaptive integration: refine where error is high
__global__ void adaptiveIntegrate(
    float* result, 
    float a, float b,  // Integration bounds
    float tolerance,
    int maxDepth
) {
    float midpoint = (a + b) / 2.0f;
    
    // Estimate integral
    float whole = (b - a) * f(midpoint);
    float left = (midpoint - a) * f((a + midpoint) / 2.0f);
    float right = (b - midpoint) * f((midpoint + b) / 2.0f);
    float refined = left + right;
    
    float error = fabsf(whole - refined);
    
    if (error < tolerance || maxDepth <= 0) {
        // Accept this approximation
        atomicAdd(result, refined);
    } else {
        // Subdivide and recurse
        adaptiveIntegrate<<<1, 1>>>(result, a, midpoint, tolerance/2, maxDepth-1);
        adaptiveIntegrate<<<1, 1>>>(result, midpoint, b, tolerance/2, maxDepth-1);
        cudaDeviceSynchronize();
    }
}
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### Issue 1: "cudaErrorLaunchFailure" in Child Kernels
**Symptom:** Parent completes but children fail silently
**Cause:** Insufficient device memory for nested grids

**Solution:**
```cpp
// Set device limit for pending kernel launches
size_t limit;
cudaDeviceGetLimit(&limit, cudaLimitDevRuntimePendingLaunchCount);
cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 8192);
```

#### Issue 2: Memory Corruption Between Parent and Child
**Symptom:** Race conditions, incorrect results
**Cause:** Missing synchronization

**Solution:**
```cpp
// ALWAYS synchronize before accessing child results
childKernel<<<grid, block>>>(d_data);
cudaDeviceSynchronize();  // Wait for child
// Now safe to use d_data
```

#### Issue 3: Stack Overflow in Deep Recursion
**Symptom:** Kernel hangs or crashes
**Cause:** Too deep nesting

**Solution:**
```cpp
// Limit recursion depth
if (depth >= MAX_DEPTH) {
    // Fall back to sequential processing
    sequentialProcess(data);
    return;
}
```

---

## ⚡ Performance Considerations

### When to Use Dynamic Parallelism
✅ **Good Use Cases:**
- Recursive algorithms (quicksort, tree traversal)
- Adaptive algorithms (AMR, ray tracing)
- Variable workload (sparse data)
- Complex control flow

❌ **Avoid When:**
- Fixed, regular workloads
- Low depth of nesting
- When CPU launch overhead is acceptable
- Memory constrained systems

### Overhead Comparison

| Approach | Launch Overhead | Memory | Complexity |
|----------|-----------------|--------|------------|
| CPU Launch | ~5-20 μs | Low | Simple |
| Dynamic Parallelism | ~1-5 μs | Higher | Complex |
| Persistent Threads | ~0.1 μs | Medium | Most Complex |

---

## 📝 Daily Summary

### Key Takeaways
1. **Dynamic parallelism** = kernels launching kernels on GPU
2. **Requires CC 3.5+** and `-rdc=true -lcudadevrt` compilation
3. **Synchronization is critical** - use `cudaDeviceSynchronize()` in device code
4. **Memory is not automatically coherent** between parent and child
5. **Limit nesting depth** to avoid resource exhaustion
6. **Best for adaptive algorithms** where workload varies by data

### Compilation Command
```bash
nvcc -arch=sm_60 -rdc=true -lcudadevrt program.cu -o program
```

---

**Day 11 Complete** ✅

*Next: Day 12 - Texture and Surface Memory - Specialized memory for image processing!*
