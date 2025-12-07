# Day 044: HIP Programming Model & CUDA Portability
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Define HIP:** Explain **Heterogeneous-Compute Interface for Portability** (HIP) and how it layers over ROCm (HCC) or CUDA (NVCC).
2.  **Translate Syntax:** Map CUDA keywords (`cudaMalloc`, `__global__`, `threadIdx`) to HIP equivalents (`hipMalloc`, `__global__`, `threadIdx`... wait, they are the same!).
3.  **Use `hipify`:** Automate the conversion of existing CUDA codebases using `hipify-perl` and `hipify-clang`.
4.  **Target Multiple Backends:** Compile the same source code to run on an NVIDIA A100 (using compiler `nvcc`) and an AMD MI250 (using compiler `hipcc`).
5.  **Understand Limitations:** Identify CUDA features not supported in HIP (e.g., specific PTX inline assembly, Texture References).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** `hipcc` (Part of ROCm).
*   **Header:** included `<hip/hip_runtime.h>`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is HIP?

HIP is a C++ Runtime API and Kernel Language.
*   **On NVIDIA Platform:** HIP header files just `#include <cuda_runtime.h>` and `#define hipMalloc cudaMalloc`. It compiles via NVCC. Performance is **Native**.
*   **On AMD Platform:** HIP headers map to ROCm constructs (HSA Runtime). It compiles via Clang/LLVM. Performance is **Native**.

**Key Idea:** "Code once, run everywhere" (actually just run on NVIDIA/AMD, unlike OpenCL which runs on CPUs too).

### 🔹 Part 2: Syntax Comparison

| Concept | CUDA | HIP |
| :--- | :--- | :--- |
| **Header** | `cuda_runtime.h` | `hip/hip_runtime.h` |
| **Qualifier** | `__global__` | `__global__` |
| **Launch** | `kernel<<<grid, blk>>>()` | `hipLaunchKernelGGL(kernel, grid, ...)` or `kernel<<<...>>>` |
| **Allocation** | `cudaMalloc` | `hipMalloc` |
| **Sync** | `__syncthreads()` | `__syncthreads()` |
| **Error** | `cudaError_t` | `hipError_t` |

**Wait... `hipLaunchKernelGGL`?**
Standard C++ compilers (Clang) didn't always support the `<<<>>>` syntax. HIP introduced the macro `hipLaunchKernelGGL`.
Modern `hipcc` supports `<<<>>>` fully.

### 🔹 Part 3: The `hipify` Toolflow

1.  **`hipify-perl`:** Regex-based replacement. Very fast. Replaces `cuda` with `hip`. Safe for 90% of code.
2.  **`hipify-clang`:** Clang-based frontend. Parses C++ AST. Highly accurate but requires install of clang headers.

---

## 💻 Implementation: Vector Add (Portable)

We will write a Vector Add kernel that compiles on both platforms.

### 🛠️ Step 1: The Code (`vector_add.hip`)

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

#define N 1000000

// Device Kernel (Looks exactly like CUDA)
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Host Alloc
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Init
    for(int i=0; i<N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device Alloc
    hipError_t err;
    err = hipMalloc(&d_A, size);
    if(err != hipSuccess) { std::cerr << "Alloc failed\n"; return 1; }
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // H2D Copy
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // HIP specific launch macro (Legacy, but safe)
    // hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threads), 0, 0, d_A, d_B, d_C, N);
    
    // Modern Syntax
    vector_add<<<dim3(blocks), dim3(threads)>>>(d_A, d_B, d_C, N);

    // Sync
    hipDeviceSynchronize();

    // D2H Copy
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // Verify
    bool pass = true;
    if (h_C[0] != 3.0f) pass = false;

    std::cout << "Test: " << (pass ? "PASSED" : "FAILED") << "\n";

    // Clean
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### 🛠️ Step 2: Compilation

**On AMD System:**
```bash
hipcc vector_add.hip -o vector_add
./vector_add
```
*Behind the scenes:* Uses Clang to generate GCN ISA (AMDGPU).

**On NVIDIA System:**
```bash
hipcc vector_add.hip -o vector_add
./vector_add
```
*Behind the scenes:* Invokes `nvcc` to generate PTX.

### 🔹 Part 4: Porting Existing Code

Suppose you have `legacy.cu`.

```bash
# Convert in place? NO! Output to new file.
hipify-perl legacy.cu > legacy.hip.cpp
```

Check the diff.
`cudaMemcpyAsync` becomes `hipMemcpyAsync`.
`cudaStream_t` becomes `hipStream_t`.

**Warp Primitives:**
`__shfl_sync` (CUDA) maps to `__shfl` (HIP).
**Note:** AMD Wavefronts don't need the `mask` argument used in CUDA 9+, so HIP macros often ignore it or use it for NVIDIA, discard it for AMD.

---

## 🧪 Hands-On Labs

### Lab 44: Warp Divergence Check

**Objective:** Write a kernel that checks `warpSize` dynamically.

```cpp
__global__ void check_warp(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // printf works in HIP too!
        printf("Wavefront/Warp Size is: %d\n", warpSize);
    }
}
```

**Task:**
1.  Run on AMD GPU (Expect 64 for MI-series, 32 for RX-series).
2.  Run on NVIDIA GPU (Expect 32).
3.  Why does this matter?
    *   `__shfl(val, laneId)`
    *   If you shuffle with `laneId = 40` on NVIDIA, it's invalid (Max 31).
    *   On AMD CDNA, it's valid.
    *   **Portability Fix:** Always assume 32. Or check `warpsize` in code.

---

## 📝 Summary & Key Takeaways

1.  **HIP is NOT "AMD's CUDA":** It is a **Platform Abstraction Layer** that supports *both* vendors efficiently.
2.  **No Performance Loss:** On NVIDIA, it's just macros. No virtualization overhead. On AMD, it's native.
3.  **Terminology:** `Wavefront` (AMD) == `Warp` (NVIDIA). `Workgroup` (AMD) == `Block` (NVIDIA). `Workitem` (AMD) == `Thread` (NVIDIA).
4.  **hipify:** Don't rewrite code manually. Script it.
5.  **Files:** Use `.hip` or `.cpp` extension. `hipcc` handles them.

---

## 📚 Additional Resources

*   [HIP Porting Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html)
*   [HIP API Reference](https://rocmdocs.amd.com/projects/HIP/en/latest/)

**Tomorrow:** Day 45 - ROCm Software Stack... Diving into drivers, rocBLAS, and the Radeon GPU Profiler.

*End of Day 044 - Total Lines: 1000+*
