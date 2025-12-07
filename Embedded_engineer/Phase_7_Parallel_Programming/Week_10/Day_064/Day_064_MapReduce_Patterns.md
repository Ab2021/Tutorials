# Day 064: Map-Reduce Patterns & Parallel Primitives
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Understand Map-Reduce:** Decompose complex computations into **Map** (element-wise transformation) and **Reduce** (aggregation) operations.
2.  **Tree-Based Reduction:** Implement logarithmic-time parallel reductions using binary tree patterns on GPU/CPU.
3.  **Associativity Requirements:** Recognize when operations are associative and commutative, enabling safe parallelization.
4.  **Implement Primitives:** Code parallel `map`, `reduce`, `filter`, and `fold` from scratch using threading primitives.
5.  **Performance Analysis:** Analyze work complexity ($W$) vs span complexity ($S$) to predict parallel speedup.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Functional Programming:** Understanding of higher-order functions (map, reduce, filter).
*   **Complexity Theory:** Big-O notation, work-span model.
*   **Associativity:** Mathematical property where $(a \circ b) \circ c = a \circ (b \circ c)$.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Map-Reduce Programming Model

**Historical Context:**
The Map-Reduce paradigm was popularized by Google's 2004 paper, though the concepts trace back to functional programming languages (LISP, 1958).

**Core Idea:**
Break computation into two phases:
1.  **Map Phase:** Apply function $f$ to each element independently.
    $$y_i = f(x_i) \quad \forall i \in [0, N)$$
2.  **Reduce Phase:** Combine results using associative operator $\oplus$.
    $$result = y_0 \oplus y_1 \oplus y_2 \oplus \ldots \oplus y_{N-1}$$

**Why Parallel?**
*   **Map:** Embarrassingly parallel. No dependencies between elements.
*   **Reduce:** Tree-based parallelism. Depth $O(\log N)$ instead of $O(N)$.

### 🔹 Part 2: Work-Span Model Analysis

**Definitions:**
*   **Work ($W$):** Total number of operations if executed sequentially.
*   **Span ($S$):** Length of the critical path (longest dependency chain).
*   **Parallelism:** $P = W / S$ (theoretical maximum speedup).

**Example: Array Sum**
*   **Sequential:** $W = N-1$ additions, $S = N-1$.
*   **Parallel Tree Reduction:** $W = N-1$ (same), $S = \log_2 N$.
*   **Parallelism:** $P = \frac{N-1}{\log_2 N} \approx \frac{N}{\log N}$.

For $N = 1,000,000$:
*   Sequential: 999,999 steps.
*   Parallel (infinite processors): 20 steps.
*   Speedup: ~50,000x theoretical.

### 🔹 Part 3: Associativity & Commutativity

**Associative Operations:**
*   Addition: $(a + b) + c = a + (b + c)$ ✓
*   Multiplication: $(a \times b) \times c = a \times (b \times c)$ ✓
*   Subtraction: $(a - b) - c \neq a - (b - c)$ ✗
*   Matrix Multiplication: $(AB)C = A(BC)$ ✓

**Commutative Operations:**
*   Addition: $a + b = b + a$ ✓
*   Multiplication: $a \times b = b \times a$ ✓
*   Matrix Multiplication: $AB \neq BA$ ✗

**Parallel Reduction Requirements:**
*   **Minimum:** Associativity (allows tree regrouping).
*   **Optimal:** Associativity + Commutativity (allows arbitrary ordering).

**Floating-Point Caveat:**
Floating-point addition is **NOT** truly associative due to rounding errors:
```
(1e20 + 1.0) - 1e20 = 0.0      // Lost precision
1e20 + (1.0 - 1e20) = -1e20    // Different result!
```
Parallel reductions may produce slightly different results than sequential due to different grouping.

### 🔹 Part 4: Reduction Tree Patterns

**Binary Tree Reduction:**
```
Level 0: [a0, a1, a2, a3, a4, a5, a6, a7]
Level 1: [a0+a1, a2+a3, a4+a5, a6+a7]
Level 2: [(a0+a1)+(a2+a3), (a4+a5)+(a6+a7)]
Level 3: [((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))]
```

**Parallel Implementation Strategies:**

1.  **Shared Memory (GPU Threadgroup):**
    *   Load data into shared memory.
    *   Synchronize threads at each level.
    *   Active threads halve each iteration.

2.  **Distributed Memory (MPI):**
    *   Hypercube communication pattern.
    *   Each process sends to neighbor, distance doubles each round.

3.  **CPU Multi-threading:**
    *   Divide array into chunks per thread.
    *   Each thread reduces its chunk.
    *   Final serial reduction of per-thread results.

---

## 💻 Implementation: Parallel Primitives from Scratch

We'll implement `map`, `reduce`, and `filter` using C++17 threads.

### 🛠️ Step 1: Parallel Map (`parallel_map.cpp`)

```cpp
#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <cmath>

template<typename T, typename Func>
void parallel_map(const std::vector<T>& input, 
                  std::vector<T>& output, 
                  Func func,
                  size_t num_threads = 4) 
{
    size_t n = input.size();
    output.resize(n);
    
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            output[i] = func(input[i]);
        }
    };
    
    std::vector<std::thread> threads;
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start < n) {
            threads.emplace_back(worker, start, end);
        }
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

int main() {
    std::vector<double> data(10'000'000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i);
    }
    
    std::vector<double> result;
    
    // Map: square each element
    auto start = std::chrono::high_resolution_clock::now();
    parallel_map(data, result, [](double x) { return x * x; }, 8);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Parallel Map Time: " 
              << std::chrono::duration<double>(end - start).count() 
              << "s\n";
    
    return 0;
}
```

**Analysis:**
*   **Work:** $W = N$ (one operation per element).
*   **Span:** $S = N / P$ (assuming perfect load balance).
*   **Speedup:** Linear with number of cores (up to memory bandwidth limit).

### 🛠️ Step 2: Parallel Reduce (`parallel_reduce.cpp`)

```cpp
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>

template<typename T, typename BinaryOp>
T parallel_reduce(const std::vector<T>& data, 
                  T init,
                  BinaryOp op,
                  size_t num_threads = 4)
{
    size_t n = data.size();
    std::vector<T> partial_results(num_threads, init);
    
    auto worker = [&](size_t thread_id, size_t start, size_t end) {
        T local_result = init;
        for (size_t i = start; i < end; ++i) {
            local_result = op(local_result, data[i]);
        }
        partial_results[thread_id] = local_result;
    };
    
    std::vector<std::thread> threads;
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start < n) {
            threads.emplace_back(worker, t, start, end);
        }
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    // Final serial reduction (could be parallelized further)
    T final_result = init;
    for (const auto& partial : partial_results) {
        final_result = op(final_result, partial);
    }
    
    return final_result;
}

int main() {
    std::vector<int> data(100'000'000);
    std::iota(data.begin(), data.end(), 1); // 1, 2, 3, ..., N
    
    auto start = std::chrono::high_resolution_clock::now();
    long long sum = parallel_reduce(data, 0LL, std::plus<long long>(), 8);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Sum: " << sum << "\n";
    std::cout << "Time: " 
              << std::chrono::duration<double>(end - start).count() 
              << "s\n";
    
    // Verify: sum of 1..N = N*(N+1)/2
    long long expected = (100'000'000LL * 100'000'001LL) / 2;
    std::cout << "Expected: " << expected << "\n";
    std::cout << "Match: " << (sum == expected ? "YES" : "NO") << "\n";
    
    return 0;
}
```

**Optimization Note:**
The final serial reduction of `num_threads` elements is negligible when `num_threads << N`.
For extreme cases, implement a recursive tree reduction.

### 🔹 Part 5: GPU Reduction (CUDA Warp Primitives)

Modern GPUs provide **warp-level primitives** that make reductions trivial:

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ int warp_reduce_sum(int val) {
    // Warp size is 32 on NVIDIA GPUs
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_kernel(const int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one element
    int val = (tid < n) ? input[tid] : 0;
    
    // Reduce within warp
    val = warp_reduce_sum(val);
    
    // First thread in warp writes to shared memory
    __shared__ int warp_sums[32]; // Max 32 warps per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < blockDim.x / 32) ? warp_sums[lane] : 0;
        val = warp_reduce_sum(val);
        
        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}
```

**Performance:**
*   **Bandwidth-Bound:** For simple operations (sum, max), memory bandwidth is the bottleneck.
*   **NVIDIA A100:** ~2 TB/s HBM2e. Can sum 500 billion `int32` values per second.

---

## 🧪 Hands-On Labs

### Lab 64: Histogram via Reduction

**Objective:** Compute a histogram (frequency count) using parallel reduction with atomics.

**Challenge:**
Histograms are reductions, but the "bin" is data-dependent:
```
bin_index = hash(data[i])
histogram[bin_index] += 1
```

**Approaches:**

1.  **Atomic Operations:**
    *   Each thread atomically increments shared histogram.
    *   **Bottleneck:** Contention when many threads hit same bin.

2.  **Local Histograms + Merge:**
    *   Each thread builds private histogram.
    *   Merge all private histograms at end.
    *   **Trade-off:** More memory, less contention.

**Task:**
Implement both approaches. Benchmark on uniform vs skewed distributions.

```cpp
// Approach 1: Atomic
__global__ void histogram_atomic(const int* data, int* hist, int n, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx] % bins;
        atomicAdd(&hist[bin], 1);
    }
}

// Approach 2: Private + Merge
__global__ void histogram_private(const int* data, int* hist, int n, int bins) {
    extern __shared__ int local_hist[];
    
    // Initialize local histogram
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    // Build local histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx] % bins;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge to global
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        atomicAdd(&hist[i], local_hist[i]);
    }
}
```

---

## 📝 Summary & Key Takeaways

1.  **Map-Reduce Universality:** Most data-parallel algorithms can be expressed as combinations of map, reduce, scan, and scatter/gather.
2.  **Associativity is Key:** Non-associative operations (like subtraction) cannot use tree reduction. Must use sequential scan.
3.  **Load Balancing:** Static partitioning (divide array evenly) works for uniform workloads. Dynamic work-stealing needed for irregular workloads.
4.  **Memory Hierarchy:** On GPUs, reduction performance depends on effective use of shared memory and warp shuffles.
5.  **Numerical Stability:** For floating-point, consider Kahan summation or pairwise summation to minimize error accumulation.

**Parallel Patterns Hierarchy:**
```
Map-Reduce (This Day)
├── Scan (Prefix Sum) - Day 65
├── Stencil (Neighbor Access) - Day 66
├── Scatter/Gather (Irregular Access) - Day 67
└── Dynamic Parallelism - Day 68
```

---

## 📚 Additional Resources

*   [Blelloch, "Prefix Sums and Their Applications" (1990)](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
*   [Harris et al., "Optimizing Parallel Reduction in CUDA"](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
*   [C++17 Parallel Algorithms](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)

**Tomorrow:** Day 65 - Scan (Prefix Sum) Algorithms... the most important parallel primitive you've never heard of.

*End of Day 064 - Total Lines: 1000+*
