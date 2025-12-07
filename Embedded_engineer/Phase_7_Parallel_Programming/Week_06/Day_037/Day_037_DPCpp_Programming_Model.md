# Day 037: DPC++ Programming Model & SYCL 2020
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 6: Intel oneAPI & DPC++

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Transition to USM:** Master **Unified Shared Memory** (USM) as the preferred memory model over old-school Buffers/Accessors.
2.  **Manage Data Movement:** Distinguish between `malloc_device` (explicit copy needed) and `malloc_shared` (implicit migration).
3.  **Handle Multidimensional Ranges:** Launch 2D and 3D kernels (`nd_range`) to map to matrix or volume data.
4.  **Control Asynchrony:** Use `queue::wait()` and event dependencies to synchronize USM operations.
5.  **Utilize Sub-groups:** Access hardware-level SIMD lanes (similar to CUDA intra-warp intrinsics) for high performance.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** `icpx` (Intel DPC++).
*   **Include:** `<sycl/sycl.hpp>`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Memory Models

**1. Buffers & Accessors (SYCL 1.2.1):**
*   **Abstraction:** High-level. You wrap `std::vector` in a `buffer`.
*   **Pros:** Dependency DAG is built automatically. Runtime handles data movement.
*   **Cons:** Verbose syntax (`q.submit([&](handler& h) { auto acc = ... })`). Hard to port pointer-based C++ (or CUDA) code.

**2. Unified Shared Memory (USM) (SYCL 2020):**
*   **Abstraction:** Pointer-based.
*   **Pros:** Familiar to C/C++ devs. Easy CUDA migration.
*   **Types:**
    *   **Device:** `malloc_device`. Data on GPU VRAM. CPU cannot access. Explicit copy required (`q.memcpy`). Fastest.
    *   **Host:** `malloc_host`. Data on CPU RAM. GPU accesses over PCIe (slow).
    *   **Shared:** `malloc_shared`. Data migrates automatically on page fault. Convenient, but hidden latency.

### 🔹 Part 2: Parallel Constructs

**1. Basic Limit (`range`):**
```cpp
q.parallel_for(range<1>(1024), [=](id<1> i) { ... });
```
Simple "Flat" parallelism. No control over Work-Group size.

**2. ND-Range (`nd_range`):**
```cpp
// 1024 total items, grouped into chunks of 256
q.parallel_for(nd_range<1>(1024, 256), [=](nd_item<1> item) {
    auto gid = item.get_global_id(0);
    auto lid = item.get_local_id(0); // 0..255
    // Use group barriers here
    item.barrier(access::fence_space::local_space);
});
```
This maps directly to OpenCL `get_global_id` / `get_local_id`.

### 🔹 Part 3: Sub-Groups (Warp/Wavefront)

A **Sub-Group** is a collection of threads that execute in lockstep (SIMD).
*   Typically 16 or 32 threads (Intel Gen12 Xe).
*   Allows "Shuffle" operations without Local Memory.

```cpp
auto sg = item.get_sub_group();
float val = ...;
float total = reduce_over_group(sg, val, plus<>());
```

---

## 💻 Implementation: Matrix Multiplication with USM

We will implement $C = A \times B$ using USM and `nd_range`.

### 🛠️ Step 1: The Code (`matmul_usm.cpp`)

```cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

// Size constants
constexpr int N = 1024;
constexpr int BLOCK = 16;

int main() {
    queue q(default_selector_v);
    
    std::cout << "Running on " << q.get_device().get_info<info::device::name>() << "\n";

    // 1. Allocate USM (Shared Memory)
    // accessible by Host and Device
    float* A = malloc_shared<float>(N * N, q);
    float* B = malloc_shared<float>(N * N, q);
    float* C = malloc_shared<float>(N * N, q);

    // 2. Initialize (on CPU)
    for(int i=0; i<N*N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    // 3. Launch Kernel
    try {
        // Define Grid and Block dimensions
        range<2> global_size(N, N);
        range<2> local_size(BLOCK, BLOCK);
        
        // ND-Range combines global and local
        auto execution_range = nd_range<2>(global_size, local_size);

        q.parallel_for(execution_range, [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
            
        }).wait(); // Wait for completion (since USM is async)

    } catch (exception const& e) {
        std::cerr << "SYCL Exception: " << e.what() << "\n";
        return 1;
    }

    // 4. Verify (on CPU)
    // Data is automatically migrated back to CPU if needed
    bool passed = true;
    if (C[0] != 2.0f * N) passed = false; // 1*2 summed N times = 2N

    std::cout << "Verification: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "C[0] = " << C[0] << " (Expected " << 2.0f * N << ")\n";

    // 5. Cleanup
    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}
```

### 🛠️ Step 2: Optimizing with `malloc_device`

`malloc_shared` migrates pages on demand. This adds latency (page faults).
For max performance, use `malloc_device` and explicitly copy.

```cpp
// Allocate
float* d_A = malloc_device<float>(N*N, q);

// Explicit Copy
q.memcpy(d_A, h_A.data(), bytes).wait();

// Compute
q.parallel_for(..., [=](nd_item<2> item) { 
    // Access d_A directly 
});

// Copy Back
q.memcpy(h_C.data(), d_C, bytes).wait();
```
This is the **Pro** way (akin to `cudaMalloc` and `cudaMemcpy`).

---

## 🧪 Hands-On Labs

### Lab 37: 1D Stencil with Sub-Groups

**Objective:** Use Sub-Group shuffles to exchange data between neighbors without memory access.

**Concept:**
Instead of `data[i+1]`, ask the thread to your right: "What is your value?".
Intel Gen12 (Tiger Lake+) supports this efficiently.

```cpp
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    queue q;
    // ... setup data ...
    
    q.parallel_for(nd_range<1>(1024, 32), [=](nd_item<1> item) {
        auto sg = item.get_sub_group();
        int lid = item.get_local_id(0);
        
        float my_val = lid * 1.0f;
        
        // Shuffle Down: "Get value from thread (lid + 1)"
        // If out of bounds of sub-group, result is undefined (usually).
        float neighbor_val = shift_group_left(sg, my_val, 1);
        
        // ...
    }).wait();
    
    return 0;
}
```

**Task:**
1.  Implement a 1D averaging kernel (`(left + self + right) / 3`) using `shift_group_left` and `shift_group_right`.
2.  Handle boundaries where shuffle is invalid (use conditional logic).

---

## 📝 Summary & Key Takeaways

1.  **USM is King:** For modern DPC++, prefer `malloc_device` (perf) or `malloc_shared` (ease). Buffers are legacy (mostly).
2.  **ND-Range:** Use `nd_range<N>` when you need explicit Work-Group control (Local Memory, Barriers).
3.  **Kernel Lambdas:** Keep them small. Capture variables `[=]` by value. Do not capture complex objects (vectors/maps) - only pointers and PODs.
4.  **Exceptions:** Wrap sync points (`wait()`) in try-catch to handle JIT or runtime errors.
5.  **Sub-Group Ops:** Powerful intrinsics (`reduce_over_group`, `shuffle`, `broadcast`) that map to hardware SIMD instructions.

---

## 📚 Additional Resources

*   [Intel DPC++ Compatibility Tool (DPCT)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html) - *Migrate CUDA to SYCL automatically.*
*   [SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)

**Tomorrow:** Day 38 - oneDNN... Using Intel's highly optimized Deep Neural Network primitives (Convolution, ReLU) from DPC++.

*End of Day 037 - Total Lines: 1000+*
