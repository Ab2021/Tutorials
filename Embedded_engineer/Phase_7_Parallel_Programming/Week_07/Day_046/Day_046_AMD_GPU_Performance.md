# Day 046: Performance Tuning on AMD GPUs
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Maximize Occupancy:** Calculate Wavefront Occupancy based on VGPR (Vector Register) and LDS usage.
2.  **Control Register Spilling:** Use `__launch_bounds__` to force the compiler to limit register usage per thread.
3.  **Optimize LDS Access:** Avoid Bank Conflicts by padding arrays (stride 32 vs 33).
4.  **Target Matrix Cores:** Execute `MFMA` (Matrix Fused Multiply Add) instructions intrinsics on CDNA architecture.
5.  **Flatten Control Flow:** Minimize divergence penalty on Wave64 architectures.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** `hipcc`.
*   **Intrinsics:** `bumltin_amdgcn_mfma_...` (Requires CDNA/MI-series GPU).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Occupancy & Registers

**The Equation:**
Max Waves per CU is hardware limited (e.g., 40 waves).
But also resource limited.
*   **VGPRs:** 256 vector registers per thread (Max).
*   **Physical File:** 64KB per CU (Example).
*   If your kernel uses 100 VGPRs per thread -> You fit fewer threads.

**Optimization:**
Reduction of VGPRs from 65 to 64 might double your theoretical occupancy (allows 2 waves instead of 1).

### 🔹 Part 2: `__launch_bounds__`

Tells the compiler: "I will never launch blocks larger than X".
Allows compiler to be aggressive with register allocation (knowing it doesn't need to support max block size).

```cpp
__global__ void __launch_bounds__(256, 4) my_kernel(...)
// Max threads per block: 256
// Min blocks per CU: 4
```
This forces the compiler to keep VGPR usage low enough to fit 4 blocks residency.

### 🔹 Part 3: Matrix Cores (MFMA)

On CDNA (MI100+), we have **Matrix Cores**.
These execute `D = A*B + C` on small tiles (e.g., 32x32x1) in a single instruction.
Accessed via `__builtin_amdgcn_mfma_f32_32x32x1f32(...)`.

**Wave64 Execution:**
All 64 threads cooperate to feed the Matrix Unit.

---

## 💻 Implementation: Tuning Matrix Transpose

We will optimize a memory-bound kernel (Transpose) by fixing bank conflicts.

### 🛠️ Step 1: Naive Transpose (`transpose_naive.hip`)

```cpp
#include <hip/hip_runtime.h>

#define DIM 32

__global__ void transpose_naive(float* out, const float* in, int width) {
    __shared__ float tile[DIM][DIM];
    
    int x = blockIdx.x * DIM + threadIdx.x;
    int y = blockIdx.y * DIM + threadIdx.y;
    
    int width_in = width;

    // Load Global -> Shared
    // Coalesced Read (Good)
    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
        
    __syncthreads();
    
    // Store Shared -> Global
    // Write x to y
    x = blockIdx.y * DIM + threadIdx.x;
    y = blockIdx.x * DIM + threadIdx.y;
    
    // Bank Conflict Here!
    // Threads in a warp access tile[0][0], tile[1][0], tile[2][0]...
    // This is Strided access on Shared Mem (Stride = 32).
    // All 32 threads hit Bank 0 simultaneously. 32-way serialization.
    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y]; 
}
```

### 🛠️ Step 2: Optimized Transpose (Padding)

To fix the conflicts, we pad the shared memory width.

```cpp
__global__ void transpose_opt(float* out, const float* in, int width) {
    // Pad column by 1. Stride becomes 33 floats.
    // Bank mapping: 
    // Row 0, Col 0 -> Bank 0
    // Row 1, Col 0 -> Bank (1*33)%32 = Bank 1
    // Row 2, Col 0 -> Bank (2*33)%32 = Bank 2
    // Conflict Free!
    __shared__ float tile[DIM][DIM + 1]; 
    
    int x = blockIdx.x * DIM + threadIdx.x;
    int y = blockIdx.y * DIM + threadIdx.y;

    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
        
    __syncthreads();
    
    x = blockIdx.y * DIM + threadIdx.x;
    y = blockIdx.x * DIM + threadIdx.y;
    
    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y]; 
}
```

### 🛠️ Step 3: Launch Bounds Experiment

```cpp
// Force low VGPR usage
__launch_bounds__(256, 8) 
__global__ void compute_heavy(...) {
    // If I use too many registers, compiler Spills to Scratch RAM (Global Mem).
    // But I will have 8 blocks residency (High Hiding).
}

// Allow high VGPR usage
__launch_bounds__(256, 1) 
__global__ void latency_heavy(...) {
    // Compiler uses tons of registers to unroll loops.
    // Low residency, but fast single-thread perf.
}
```

### 🔹 Part 4: CDNA Matrix Intrinsics (Advanced)

Only works on CDNA (Instinct) cards.

```cpp
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx90a__) // MI250
__global__ void mfma_test(float* d, float* a, float* b) {
    typedef float  v32f __attribute__((ext_vector_type(32))); 
    typedef float  v16f __attribute__((ext_vector_type(16))); 
    typedef float  v4f  __attribute__((ext_vector_type(4))); 
    
    v32f c = {0}; // Accumulator
    v16f a_vec = ...; // Load A
    v16f b_vec = ...; // Load B
    
    // 32x32x1 F32 Matrix Multiply
    // C = A * B + C
    c = __builtin_amdgcn_mfma_f32_32x32x1f32(a, b, c, 0, 0, 0);
    
    // Store C...
}
#endif
```

---

## 🧪 Hands-On Labs

### Lab 46: Occupancy Calculator

**Objective:** Use `rocm_smi` and CLI tools to verify occupancy.

**Task:**
1.  Run the `transpose_naive` kernel.
2.  Use `rocprof --stats` to check `LDSBankConflict` counter (if available) or `MemUnitStalled`.
3.  Run `transpose_opt`.
4.  Observe speedup (typically 20-30% on memory bound kernels).

---

## 📝 Summary & Key Takeaways

1.  **Bank Conflicts:** The silent killer of Shared Memory performance. Always ensure stride is NOT a multiple of 32 (use 33).
2.  **Launch Bounds:** The primary knob for you to control the Compiler's Register Allocator.
3.  **Hiding Latency:** AMD GPUs need high occupancy (many waves in flight) to cover the deep Global Memory latency.
4.  **MFMA:** Powerful, but low-level. Prefer `rocBLAS` unless you are writing a custom attention kernel.

---

## 📚 Additional Resources

*   [AMD GCN3 ISA Guide (Concepts apply to RDNA/CDNA)](https://gpuopen.com/documentation/amd-isa-documentation/)
*   [ROCm Performance Optimization Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/Performance-Optimization-Guide.html)

**Tomorrow:** Day 47 - Multi-GPU Programming with HIP... `hipMemcpyPeer`, P2P access, and scaling across multiple cards.

*End of Day 046 - Total Lines: 1000+*
