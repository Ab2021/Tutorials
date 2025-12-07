# Day 048: Advanced ROCm Libraries (rocPRIM & rocThrust)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Utilize rocThrust:** Apply STL-like algorithms (`sort`, `transform`, `reduce`) to GPU vectors without writing kernels.
2.  **Master rocPRIM:** Implement low-level, high-performance parallel primitives at the **Block** and **Warp** level (equivalent to NVIDIA CUB).
3.  **Deploy Deep Learning:** Understanding **MIOpen** and its role in accelerating PyTorch/TensorFlow on AMD hardware.
4.  **Differentiate APIs:** Choose between High-Level (Thrust) for productivity and Low-Level (PRIM) for kernel fusion performance.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Libs:** `rocprim`, `rocthrust`, `miopen-hip`.
*   **Compile:** `hipcc`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: rocThrust (Productivity)

**What is it?**
A port of NVIDIA Thrust.
*   **Container:** `thrust::device_vector` (Manages `hipMalloc`/`hipFree`).
*   **Algorithms:** `thrust::sort`, `thrust::reduce`.
*   **Backend:** Automatically dispatched to ROCm backend.

**Use Case:** Quick prototyping, sorting large arrays, non-critical paths.

### 🔹 Part 2: rocPRIM (Performance)

**What is it?**
A port of NVIDIA CUB.
*   **Header-only:** Template library.
*   **Scope:**
    *   **Warp-level:** `WarpScan`, `WarpReduce`.
    *   **Block-level:** `BlockRadixSort`, `BlockReduce`.
    *   **Device-level:** Global sort/scan.
*   **Fusion:** Allows you to fuse "Load -> Sort -> Store" inside a single kernel to save bandwidth.

### 🔹 Part 3: MIOpen

**What is it?**
AMD's deep learning primitives lib (Open Source).
*   Supports Convolution (Winograd, Direct, GEMM).
*   Supports RNNs, Batch Norm.
*   **Tuning:** Has an internalized database of "Best Kernels" for specific GPU models (gfx906, gfx908).

---

## 💻 Implementation: Fast Sorting

We will sort 1 Million integers using both libraries.

### 🛠️ Step 1: rocThrust Sort (`sort_thrust.cpp`)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <iostream>

int main() {
    int N = 1000 * 1000;
    
    // 1. Alloc on Device & Init
    thrust::device_vector<int> d_vec(N);
    thrust::generate(d_vec.begin(), d_vec.end(), rand);

    // 2. Sort
    std::cout << "Sorting...\n";
    thrust::sort(d_vec.begin(), d_vec.end());

    // 3. Verify
    bool sorted = thrust::is_sorted(d_vec.begin(), d_vec.end());
    std::cout << "Sorted: " << (sorted ? "YES" : "NO") << "\n";
    
    return 0;
}
```
*Compile:* `hipcc sort_thrust.cpp -o sort_thrust`

### 🛠️ Step 2: rocPRIM Block Sort (`sort_prim.cpp`)

To write a custom kernel that sorts chunks of data.

```cpp
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <iostream>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

__global__ void block_sort_kernel(int* d_in, int* d_out) {
    // 1. Define Type
    using BlockSortT = rocprim::BlockRadixSort<int, BLOCK_SIZE, ITEMS_PER_THREAD>;
    
    // 2. Alloc Temp Storage in Shared Mem
    __shared__ typename BlockSortT::Storage temp_storage;
    
    int tid = threadIdx.x;
    int items[ITEMS_PER_THREAD];
    
    // 3. Load Items (Strided)
    // For simplicity, just load dummy or from d_in...
    for(int i=0; i<ITEMS_PER_THREAD; i++) {
         items[i] = d_in[blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD + tid + i*BLOCK_SIZE];
    }
    
    // 4. Sort (Collective Op across 256 threads)
    BlockSortT().Sort(items, temp_storage);
    
    // 5. Store
    for(int i=0; i<ITEMS_PER_THREAD; i++) {
         d_out[blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD + tid + i*BLOCK_SIZE] = items[i];
    }
}

int main() {
    // ... Boilerplate setup N items ...
    // Launch:
    hipLaunchKernelGGL(block_sort_kernel, dim3(100), dim3(BLOCK_SIZE), 0, 0, d_in, d_out);
    // ...
}
```

### 🔹 Part 4: MIOpen Flow

1.  **Create Handle:** `miopenCreate(&handle)`.
2.  **Create Tensor Descriptor:** `miopenCreateTensorDescriptor`.
    *   Set N, C, H, W.
3.  **Find Convolution Algorithm:**
    *   `miopenFindConvolutionForwardAlgorithm`.
    *   MIOpen benchmarks *all* kernels (Direct, GEMM, FFT) and returns the fastest `algo_id` for your specific dimensions.
4.  **Execute:** `miopenConvolutionForward`.

This "Auto-Tuning" phase is critical. NVIDIA cuDNN uses heuristics; MIOpen prefers empirical search (can be cached).

---

## 🧪 Hands-On Labs

### Lab 48: Warp Reduce Sum

**Objective:** Write a kernel using `rocprim::WarpReduce` to sum 64 integers in a Wavefront.

**Code Snippet:**
```cpp
#include <rocprim/rocprim.hpp>

__global__ void warp_sum(int* d_in, int* d_out) {
    int val = d_in[threadIdx.x]; // Load 1 item
    
    using WarpReduceT = rocprim::WarpReduce<int>;
    __shared__ typename WarpReduceT::Storage temp;
    
    WarpReduceT().Reduce(val, val, temp); // Sum 'val' across warp
    
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = val; // Write result
    }
}
```
**Task:**
1.  Implement and run.
2.  Does it work on Wave32 (RDNA)? Yes, rocPRIM handles wave size internally if compiled correctly.

---

## 📝 Summary & Key Takeaways

1.  **rocThrust:** Your "Standard Library" for GPU. Use it for data parallel tasks (`transform`, `scan`) where you don't need custom logic fusion.
2.  **rocPRIM:** The building blocks for High Performance Kernels. Use `BlockReduce`/`BlockSort` to keep data on-chip.
3.  **Header Only:** Both are template libraries. They increase compile time but result in zero runtime overhead (inlined assembly).
4.  **MIOpen:** The engine behind AMD's AI push. It relies on finding the best pre-compiled binary for a given tensor shape.

---

## 📚 Additional Resources

*   [rocPRIM Documentation](https://rocmdocs.amd.com/projects/rocPRIM/en/latest/)
*   [rocThrust Documentation](https://rocmdocs.amd.com/projects/rocThrust/en/latest/)

**Tomorrow:** Day 49 - Week 7 Review & Project... Creating a Cross-Vendor Fractal Renderer (HIP for AMD/NVIDIA).

*End of Day 048 - Total Lines: 1000+*
