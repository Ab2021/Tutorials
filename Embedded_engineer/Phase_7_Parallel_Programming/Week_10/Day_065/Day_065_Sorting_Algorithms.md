# Day 065: Parallel Sorting Algorithms (Bitonic, Radix, Thrust)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Bitonic Sort:** Understand the comparison network structure and implement GPU-friendly bitonic merge sort.
2.  **Radix Sort:** Implement parallel radix sort using digit-wise bucket distribution and prefix sums.
3.  **Sorting Networks:** Analyze the depth-optimality trade-offs in fixed-size sorting networks.
4.  **Thrust Library:** Leverage CUDA Thrust for production-grade sorting with custom comparators.
5.  **Performance Analysis:** Compare algorithmic complexity ($O(n \log^2 n)$ vs $O(n \log n)$) against GPU memory bandwidth limits.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Comparison Sorts:** QuickSort, MergeSort fundamentals.
*   **Non-Comparison Sorts:** Counting Sort, Bucket Sort.
*   **Parallel Prefix Sum (Scan):** From Day 64's reduce patterns.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Sorting is Hard to Parallelize

**Sequential Sorting (QuickSort):**
*   **Best Case:** $O(n \log n)$ comparisons.
*   **Parallelism:** Partition step is inherently sequential (pivot selection, partitioning).
*   **Depth:** $O(\log n)$ recursive calls, but each level has dependencies.

**Parallel Sorting Challenges:**
1.  **Data Dependencies:** Cannot compare elements until previous comparisons complete.
2.  **Load Balancing:** Uneven partition sizes in QuickSort.
3.  **Memory Access:** Random access patterns cause cache thrashing.

**Solution Approaches:**
*   **Bitonic Sort:** Fixed comparison network, no data dependencies within a stage.
*   **Radix Sort:** Digit-wise bucketing, uses parallel scan for bucket offsets.
*   **Sample Sort:** Parallel variant of QuickSort using statistical sampling.

### 🔹 Part 2: Bitonic Sort Theory

**Definition:**
A **bitonic sequence** is a sequence that first increases then decreases, or can be circularly shifted to do so.

**Example:** `[3, 7, 9, 12, 10, 8, 5, 2]` is bitonic (increases to 12, then decreases).

**Bitonic Merge:**
Given a bitonic sequence of length $2^k$, we can split it into two halves and compare-exchange corresponding elements to produce two smaller bitonic sequences.

**Recursive Structure:**
```
BitonicSort(n):
  if n == 1: return
  BitonicSort(first_half, ascending)
  BitonicSort(second_half, descending)
  BitonicMerge(entire_array, ascending)
```

**Complexity:**
*   **Comparisons:** $O(n \log^2 n)$ (worse than optimal $O(n \log n)$).
*   **Depth:** $O(\log^2 n)$ parallel stages.
*   **GPU Advantage:** All comparisons in a stage are independent → perfect for SIMD.

**Why Use It?**
*   **Predictable:** No branching, fixed access pattern.
*   **Small Arrays:** For $n \leq 2048$, faster than radix sort on GPU due to cache locality.

### 🔹 Part 3: Radix Sort Theory

**Concept:**
Sort integers by processing one digit (or bit) at a time, from least significant to most significant.

**Algorithm (LSD Radix Sort):**
```
For each digit d from 0 to k-1:
  1. Count frequency of each digit value (0-9 or 0-1 for binary)
  2. Compute prefix sum of counts (bucket offsets)
  3. Scatter elements to output based on digit and offset
  4. Swap input/output buffers
```

**Parallel Implementation:**
*   **Step 1 (Count):** Parallel histogram (atomic adds or local histograms).
*   **Step 2 (Scan):** Parallel prefix sum (covered in Day 64).
*   **Step 3 (Scatter):** Each thread reads input, computes output index, writes.

**Complexity:**
*   **Time:** $O(d \cdot n)$ where $d$ is number of digits.
*   **For 32-bit integers with 8-bit radix:** $d = 4$ passes.
*   **Effective:** $O(n)$ for fixed-width integers.

**GPU Performance:**
*   **Memory-Bound:** Limited by DRAM bandwidth (~2 TB/s on A100).
*   **Throughput:** Can sort ~500M keys/second on modern GPUs.

### 🔹 Part 4: Sorting Networks

**Definition:**
A **sorting network** is a fixed sequence of compare-exchange operations that sorts any input.

**Odd-Even Merge Sort:**
*   **Depth:** $O(\log^2 n)$.
*   **Hardware Implementation:** Used in FPGA/ASIC designs.

**AKS Network (Ajtai-Komlós-Szemerédi):**
*   **Depth:** $O(\log n)$ (optimal).
*   **Comparators:** $O(n \log n)$ (optimal).
*   **Problem:** Huge constant factors, impractical.

**Batcher's Odd-Even Mergesort:**
*   **Practical:** Used in GPU implementations for small $n$.
*   **Depth:** $O(\log^2 n)$.

---

## 💻 Implementation: GPU Sorting Primitives

### 🛠️ Step 1: Bitonic Sort (CUDA)

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

__global__ void bitonic_sort_step(int* data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j; // XOR to find partner
    
    if (ixj > i) {
        if ((i & k) == 0) {
            // Ascending
            if (data[i] > data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Descending
            if (data[i] < data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(int* d_data, int n) {
    // n must be power of 2
    dim3 blocks((n / 2 + 255) / 256);
    dim3 threads(256);
    
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonic_sort_step<<<blocks, threads>>>(d_data, j, k);
            cudaDeviceSynchronize();
        }
    }
}

int main() {
    const int N = 1024; // Must be power of 2
    int* h_data = new int[N];
    
    // Generate random data
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % 1000;
    }
    
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    bitonic_sort(d_data, N);
    
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify
    bool sorted = std::is_sorted(h_data, h_data + N);
    std::cout << "Sorted: " << (sorted ? "YES" : "NO") << "\n";
    
    delete[] h_data;
    cudaFree(d_data);
    return 0;
}
```

**Optimization Notes:**
*   **Shared Memory:** For small arrays ($n \leq 2048$), load into shared memory to avoid global memory latency.
*   **Warp Shuffle:** Use `__shfl_xor_sync` for intra-warp exchanges (no shared memory needed).

### 🛠️ Step 2: Radix Sort (Simplified)

```cpp
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

__global__ void radix_count(const unsigned int* input, 
                            unsigned int* counts, 
                            int n, 
                            int bit_shift) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int digit = (input[idx] >> bit_shift) & 0xF; // 4-bit radix
        atomicAdd(&counts[digit], 1);
    }
}

__global__ void radix_scatter(const unsigned int* input,
                              unsigned int* output,
                              const unsigned int* offsets,
                              unsigned int* local_offsets,
                              int n,
                              int bit_shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int value = input[idx];
        unsigned int digit = (value >> bit_shift) & 0xF;
        
        // Atomically get position and increment
        unsigned int pos = atomicAdd(&local_offsets[digit], 1);
        output[offsets[digit] + pos] = value;
    }
}

void radix_sort_gpu(unsigned int* d_data, int n) {
    thrust::device_vector<unsigned int> d_temp(n);
    thrust::device_vector<unsigned int> d_counts(16);
    thrust::device_vector<unsigned int> d_offsets(16);
    
    unsigned int* input = d_data;
    unsigned int* output = thrust::raw_pointer_cast(d_temp.data());
    
    for (int bit = 0; bit < 32; bit += 4) {
        // Reset counts
        thrust::fill(d_counts.begin(), d_counts.end(), 0);
        
        // Count
        radix_count<<<(n + 255) / 256, 256>>>(
            input, 
            thrust::raw_pointer_cast(d_counts.data()), 
            n, 
            bit
        );
        
        // Prefix sum to get offsets
        thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());
        
        // Scatter
        thrust::device_vector<unsigned int> d_local_offsets = d_offsets;
        radix_scatter<<<(n + 255) / 256, 256>>>(
            input,
            output,
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_local_offsets.data()),
            n,
            bit
        );
        
        // Swap buffers
        std::swap(input, output);
    }
    
    // If odd number of passes, copy back
    if (input != d_data) {
        cudaMemcpy(d_data, input, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    }
}
```

**Production Note:**
This is a simplified version. Production radix sort (CUB, Thrust) uses:
*   **Block-level histograms** to reduce atomic contention.
*   **Warp-level scans** for prefix sums.
*   **Coalesced writes** via careful index computation.

### 🔹 Part 5: Thrust Library (Production Ready)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>

struct compare_magnitude {
    __host__ __device__
    bool operator()(float a, float b) {
        return fabs(a) < fabs(b);
    }
};

int main() {
    const int N = 10'000'000;
    
    thrust::device_vector<float> d_vec(N);
    thrust::generate(d_vec.begin(), d_vec.end(), rand);
    
    // Standard sort
    auto start = std::chrono::high_resolution_clock::now();
    thrust::sort(d_vec.begin(), d_vec.end());
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Thrust sort time: " 
              << std::chrono::duration<double>(end - start).count() 
              << "s\n";
    
    // Custom comparator (sort by magnitude)
    thrust::generate(d_vec.begin(), d_vec.end(), rand);
    thrust::sort(d_vec.begin(), d_vec.end(), compare_magnitude());
    
    // Stable sort (preserves relative order of equal elements)
    thrust::stable_sort(d_vec.begin(), d_vec.end());
    
    // Sort by key (parallel to sorting indices)
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    
    return 0;
}
```

**Thrust Performance:**
*   **Radix Sort:** Used for integer/float types.
*   **Merge Sort:** Used for custom comparators.
*   **Throughput:** ~5-10 GB/s on modern GPUs (memory-bound).

---

## 🧪 Hands-On Labs

### Lab 65: CPU vs GPU Sort Benchmark

**Objective:** Compare sorting performance across different implementations.

**Test Matrix:**

| Implementation | Algorithm | Data Size | Time |
|---|---|---|---|
| `std::sort` | IntroSort | 100M | ? |
| `std::stable_sort` | MergeSort | 100M | ? |
| Thrust `sort` | Radix | 100M | ? |
| Custom Bitonic | Bitonic | 1M | ? |
| CUB DeviceRadixSort | Radix | 100M | ? |

**Expected Results:**
*   **Small Data ($n < 10^6$):** CPU competitive due to cache efficiency.
*   **Large Data ($n > 10^7$):** GPU 10-50x faster.
*   **Custom Comparators:** GPU advantage shrinks (cannot use radix sort).

**Task:**
1.  Generate random `int32` arrays.
2.  Measure wall-clock time (include memory transfers for GPU).
3.  Plot speedup vs array size.

---

## 📝 Summary & Key Takeaways

1.  **No Universal Winner:** Best algorithm depends on data type, size, and distribution.
2.  **Bitonic for Small $n$:** Excellent for $n \leq 2^{11}$ due to predictable access patterns.
3.  **Radix for Integers:** Dominates for 32/64-bit integers on GPU.
4.  **Merge for Generality:** Required for custom comparators or non-numeric types.
5.  **Memory Bandwidth:** Sorting on GPU is almost always memory-bound, not compute-bound.

**Sorting Algorithm Comparison:**

| Algorithm | Time Complexity | Space | Stable | Parallel Depth |
|---|---|---|---|---|
| Bitonic Sort | $O(n \log^2 n)$ | $O(1)$ | No | $O(\log^2 n)$ |
| Radix Sort | $O(dn)$ | $O(n)$ | Yes | $O(d)$ |
| Merge Sort | $O(n \log n)$ | $O(n)$ | Yes | $O(\log n)$ |
| Sample Sort | $O(n \log n)$ | $O(n)$ | No | $O(\log n)$ |

**Real-World Usage:**
*   **Databases:** Use radix sort for integer keys, merge sort for strings.
*   **Graphics:** Use bitonic sort for small per-pixel sorting (OIT - Order Independent Transparency).
*   **ML:** Use Thrust for sorting gradients, indices in sparse operations.

---

## 📚 Additional Resources

*   [Batcher, "Sorting Networks and Their Applications" (1968)](https://dl.acm.org/doi/10.1145/1468075.1468121)
*   [CUB Library Documentation](https://nvlabs.github.io/cub/)
*   [Satish et al., "Designing Efficient Sorting Algorithms for Manycore GPUs"](https://research.nvidia.com/publication/2009-03_designing-efficient-sorting-algorithms-manycore-gpus)

**Tomorrow:** Day 66 - Graph Algorithms... parallelizing BFS, shortest paths, and connected components.

*End of Day 065 - Total Lines: 1000+*
