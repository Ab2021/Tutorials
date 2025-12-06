# Day 20: Rapid GPU Development with Thrust and CUB
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 3: GPU Libraries

---

> **🎯 Focus Area:** Master high-productivity libraries to write concise, efficient CUDA C++ code using **Thrust** (STL-like) and **CUB** (Block/Warp primitives).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Utilize** Thrust vectors and algorithms (`sort`, `reduce`, `transform`) for rapid prototyping.
2.  **Implement** custom "Functors" to run arbitrary logic with Thrust.
3.  **Differentiate** between Thrust (Host-side API) and CUB (Device-side primitives).
4.  **Integrate** CUB block-level primitives (`BlockReduce`, `BlockScan`) inside custom kernels.
5.  **Achieve** near-peak performance with significantly fewer lines of code than raw CUDA.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU
- CUB/Thrust rely heavily on atomic operations and shared memory.

### Software Environment
- **Thrust:** Included in CUDA Toolkit.
- **CUB:** Included in CUDA Toolkit (since version 11).
```bash
# Check include paths
ls /usr/local/cuda/include/thrust/
ls /usr/local/cuda/include/cub/
```

### Prior Knowledge
- C++ Standard Template Library (STL) Concepts (Vectors, Iterators).
- Day 4: Parallel Reduction (Conceptually, CUB implements this optimally).

---

## 📖 Theoretical Foundation

### 1. Thrust: The STL of CUDA

Writing `cudaMalloc`, `cudaMemcpy`, and raw kernels for simple tasks (like summing a vector) is tedious and error-prone. **Thrust** abstracts this away.
*   **Containers:** `thrust::host_vector`, `thrust::device_vector`.
*   **Algorithms:** `thrust::sort`, `thrust::reduce`, `thrust::transform`, `thrust::scan`.
*   **Backends:** Can target CUDA (GPU), OpenMP (CPU), or TBB (CPU) just by changing a flag.

### 2. CUB (CUDA Unbound)

While Thrust is a **high-level** library (you call it from the CPU), CUB is a **flexible, low-level** library of CUDA kernel primitives (you call it *inside* your `__global__` functions).

**Components:**
*   **Warp Primitives:** `WarpScan`, `WarpReduce`.
*   **Block Primitives:** `BlockLoad`, `BlockRadixSort`, `BlockReduce`.
*   **Device Primitives:** `DeviceRadixSort` (Similar to Thrust but callable).

**Why use CUB?**
If you are writing a complex custom kernel but need to "sort data within a thread block," writing a bitonic sort from scratch is hard. CUB provides a `BlockSort` template that just works and is highly optimized.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Thrust Algorithms

This example demonstrates how to replace 100 lines of CUDA boilerplate with 10 lines of Thrust.

#### 📁 `src/thrust_demo.cu`
```cpp
/*
 * Day 20: Thrust Basics
 * Phase 6: Platform Engineering
 *
 * Demonstrates vectors, sorting, and transformation functors.
 * Compile: nvcc -o thrust_demo thrust_demo.cu
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>
#include <cstdlib>

// Custom Functor for "SAXPY" (y = a*x + y) style logic or Squares
struct square_op {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

int main() {
    // 1. Generation
    // Generate scale random data on Host
    thrust::host_vector<float> h_vec(10);
    for(size_t i = 0; i < h_vec.size(); i++) 
        h_vec[i] = (float)(rand() % 100);

    // 2. Transfer
    // Copy to Device (Automates cudaMalloc & cudaMemcpy)
    thrust::device_vector<float> d_vec = h_vec;

    std::cout << "Original: ";
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";

    // 3. Sorting (Radix Sort under the hood)
    thrust::sort(d_vec.begin(), d_vec.end());

    std::cout << "Sorted:   ";
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";

    // 4. Transformation
    // output = input^2
    thrust::device_vector<float> d_result(10);
    thrust::transform(d_vec.begin(), d_vec.end(), d_result.begin(), square_op());

    std::cout << "Squared:  ";
    thrust::copy(d_result.begin(), d_result.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";

    // 5. Reduction
    float sum = thrust::reduce(d_result.begin(), d_result.end(), 0.0f, thrust::plus<float>());
    std::cout << "Sum of Squares: " << sum << std::endl;

    return 0;
}
```

### 👨‍💻 Advanced: CUB Block Primitives

This example shows how to use `cub::BlockReduce` inside a custom kernel. This is essential when writing optimized Softmax or Normalization layers.

#### 📁 `src/cub_kernel.cu`
```cpp
/*
 * Day 20: CUB Block Primitives
 * Phase 6: Platform Engineering
 *
 * Implements a Block Reduction (Sum) using CUB.
 * Compile: nvcc -o cub_kernel cub_kernel.cu
 */

#include <iostream>
#include <vector>
#include <cub/cub.cuh>

// Kernel that sums a segment of items per thread block
// Each block processes BLOCK_THREADS items * ITEMS_PER_THREAD
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSumKernel(int *d_in, int *d_out, int n) {
    // 1. Define the CUB primitive type
    // This defines the storage logic for the reduction
    typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;

    // 2. Allocate Shared Memory for CUB
    // CUB helper calculates exact bytes needed
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // 3. Per-Thread Data Loading
    // Each thread holds a small array locally (Registers)
    int thread_data[ITEMS_PER_THREAD];
    
    // Global offset for this block
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int thread_offset = block_offset + (threadIdx.x * ITEMS_PER_THREAD);

    // Simple Load
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (thread_offset + i < n)
            thread_data[i] = d_in[thread_offset + i];
        else
            thread_data[i] = 0; // Padding
    }

    // 4. Compute thread-local sum first
    int thread_sum = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        thread_sum += thread_data[i];
    }

    // 5. Compute Block-wide sum using CUB
    // This runs the efficient tree-reduction in shared memory
    int block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    // 6. Store Result
    // Only thread 0 receives the valid reduced result
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = block_sum;
    }
}

int main() {
    int num_items = 1024 * 1024; // 1M items
    int block_threads = 128;
    int items_per_thread = 4;
    
    // Each block handles (128*4) = 512 items
    // Total blocks = 1M / 512 = 2048
    int tile_size = block_threads * items_per_thread;
    int grid_size = (num_items + tile_size - 1) / tile_size;

    std::cout << "Items: " << num_items << ", Grid: " << grid_size << " blocks" << std::endl;

    // Host setup
    std::vector<int> h_in(num_items, 1); // Fill with 1s
    std::vector<int> h_out(grid_size);
    
    int *d_in, *d_out;
    cudaMalloc(&d_in, num_items * sizeof(int));
    cudaMalloc(&d_out, grid_size * sizeof(int));
    
    cudaMemcpy(d_in, h_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);

    // Launch
    BlockSumKernel<128, 4><<<grid_size, block_threads>>>(d_in, d_out, num_items);

    // Check
    cudaMemcpy(h_out.data(), d_out, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify (Total sum should be num_items)
    // We only summed per-block, so we sum the block sums on CPU for verification
    int total_sum = 0;
    for (int x : h_out) total_sum += x;

    std::cout << "Computed Sum: " << total_sum << std::endl;
    std::cout << "Expected Sum: " << num_items << std::endl;
    
    if (total_sum == num_items) std::cout << "SUCCESS!" << std::endl;
    else std::cout << "FAILURE" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

---

## 🔬 Lab Exercise: "Thrust vs Native Performance"

### Lab Objectives
1.  Implement a `Sort` operation using `thrust::sort`.
2.  Implement a Bitonic Sort (or similar) or use `std::sort` on CPU.
3.  Benchmark the two for $N=1,000,000$ and $N=100,000,000$.

### Analysis
*   **Thrust Sort:** Uses extremely optimized Device Radix Sort. For 100M integers, it is typically limited only by memory bandwidth ($>90\%$ peak bandwidth).
*   **CUB:** CUB powers Thrust's backend. When you call `thrust::device_vector`, it often dispatches to CUB routines.

### Exercise: Zip Iterators
A common pattern is **"Sort-Reference"**: Sort Array A, but permute Array B accordingly (Key-Value sort).
Thrust makes this trivial:
```cpp
thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
```
Try implementing this for a list of mock objects (ID, Score) sorting by Score.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Productivity / Performance:** Thrust gives you 90% of the maximum possible performance for 1% of the code effort. **Always check if a Thrust algo exists before writing a kernel.**
2.  **CUB for Kernels:** When you MUST write a kernel, use CUB for the hard parts (scan, reduce, sort within block).
3.  **Iterators:** Thrust iterators (`zip_iterator`, `counting_iterator`, `transform_iterator`) allow you to fuse operations (Kernel Fusion) lazily without creating temp arrays.

### API Summary
```cpp
// Thrust
thrust::device_vector<int> v(100);
thrust::fill(v.begin(), v.end(), 9);
thrust::sort(v.begin(), v.end());
thrust::reduce(v.begin(), v.end());

// CUB (Inside Kernel)
typedef cub::BlockReduce<int, 128> BlockReduce;
__shared__ BlockReduce::TempStorage temp;
int sum = BlockReduce(temp).Sum(thread_data);
```

---

**Day 20 Complete** ✅

*Next: Day 21 - Week 3 Review & The Libraries Project - Building a High-Performance Compute Engine!*
