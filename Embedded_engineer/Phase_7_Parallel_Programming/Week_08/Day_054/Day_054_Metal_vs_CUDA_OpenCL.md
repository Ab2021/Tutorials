# Day 054: Metal vs CUDA vs OpenCL (Porting Guide)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 8: Apple Metal & GPU Compute

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Map Terminology:** Translate CUDA concepts (Grid, Block, Warp) to Metal (Grid, Threadgroup, SIMD-group).
2.  **Refactor Host Code:** Understand the philosophical differences between CUDA's "Hidden Global State" (Contexts) and Metal's "Explicit Object OO" (Device/CommandQueue).
3.  **Handle Coordinate Systems:** Fix common bugs related to Texture Coordinates (0..1 vs 0..Width) and Z-Clip space.
4.  **Manage Compilation:** Compare JIT (OpenCL) vs AOT (Metal `.metallib`) vs Fatbin (CUDA).
5.  **Strategy:** Decide between Native Port (Rewriting in MSL) vs Translation Layers (MoltenVK).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Mindset:** Open to unlearning "CUDA-isms" (like Global Contexts).
*   **Reference:** A CUDA kernel to translate.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Rosetta Stone

| Feature | CUDA (NVIDIA) | OpenCL (Standard) | Metal (Apple) |
| :--- | :--- | :--- | :--- |
| **Worker** | Thread | Work-item | Thread |
| **Group** | Block | Work-group | Threadgroup |
| **Vector Unit** | Warp (32) | Sub-group | SIMD-group (32) |
| **Fast Mem** | Shared Memory | Local Memory | Threadgroup Memory |
| **RAM** | Global Memory | Global Memory | Device Memory |
| **Index** | `threadIdx` | `get_local_id` | `thread_position_in_threadgroup` |
| **Global Id**| `blockIdx*Dim+...` | `get_global_id` | `thread_position_in_grid` |

### 🔹 Part 2: Host API Philosophy

**CUDA:**
*   **Procedural C-style.**
*   Implicit Context stack (`cudaSetDevice`).
*   Launch syntax: `kernel<<<...>>>`.
*   Synchronization: `cudaDeviceSynchronize()` (Blocking).

**Metal:**
*   **Object-Oriented (Obj-C++).**
*   No implicit state. You must pass `device`, `queue`, `pipeline` explicitly.
*   Command Buffer logic: "Record -> Commit".
*   Synchronization: `[buffer waitUntilCompleted]` or completion handlers (Async callbacks).

### 🔹 Part 3: Memory Models

**CUDA (Discrete):**
*   Host (CPU) vs Device (GPU) are separate.
*   Must `cudaMemcpy`.

**Metal (Unified - Apple Silicon):**
*   `Shared` Mode: CPU writes, GPU reads instantly.
*   Coherency is hardware managed (snooping L2).
*   **Porting Trap:** If you port `cudaMemcpy` logic blindly, you are wasting performance. Remove copies! Just pass the pointer.

---

## 💻 Implementation: Porting SAXPY

We will take a classic "Single-Precision A*X + Y" from CUDA and port it to Metal.

### 🛠️ Step 1: The Original CUDA

```cpp
// CUDA
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*4);
    // ... Copy ...
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
}
```

### 🛠️ Step 2: The Metal Port (Shader)

**File: `saxpy.metal`**

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(constant int& n   [[ buffer(0) ]],
                  constant float& a [[ buffer(1) ]],
                  device float* x   [[ buffer(2) ]],
                  device float* y   [[ buffer(3) ]],
                  uint id [[ thread_position_in_grid ]])
{
    if (id < n) {
        y[id] = a * x[id] + y[id];
    }
}
```
**Changes:**
1.  Arguments need `[[ buffer(n) ]]` attributes.
2.  Scalars (`n`, `a`) must be passed in a buffer (or via `setBytes`), they cannot be direct arguments in the signature like CUDA.
3.  `id` is injected by the hardware via `[[ ... ]]` attribute.

### 🛠️ Step 3: The Metal Port (Host C++)

```cpp
// ... Setup Device & Pipeline ...

// 1. Arguments
int n = 100000;
float a = 2.0f;

// 2. Buffers
MTL::Buffer* bufX = ...;
MTL::Buffer* bufY = ...; // Shared Mode

// 3. Encode
MTL::CommandBuffer* cmd = queue->commandBuffer();
MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

enc->setComputePipelineState(pso);
// Scalars: send immediate bytes (Inline constants)
enc->setBytes(&n, sizeof(int), 0);
enc->setBytes(&a, sizeof(float), 1);
enc->setBuffer(bufX, 0, 2);
enc->setBuffer(bufY, 0, 3);

// 4. Dispatch
MTL::Size gridSize(n, 1, 1);
MTL::Size groupSize(256, 1, 1);
enc->dispatchThreads(gridSize, groupSize);

enc->endEncoding();
cmd->commit();
```

### 🔹 Part 4: Texture Coordinates

*   **CUDA/OpenGL:**
    *   Unnormalized (Surface Object): `read(x, y)` where x is 0..Width.
    *   Normalized (Texture Object): 0.0 .. 1.0.
*   **Metal:**
    *   Defaults to Normalized (0..1) usually, but check `normalizedCoordinates` flag on Sampler.
    *   **Wait:** Compute Kernels usually use `texture.read(uint2(x,y))` which is Unnormalized (integer coords).
    *   **Trap:** Metal Pixel Center convention is (0.5, 0.5) for SAMPLING, but (0,0) for READING integers.

---

## 🧪 Hands-On Labs

### Lab 54: The "Coalescing" Myth

**Objective:** Test memory stride patterns.

**Context:**
In CUDA, coalescing (accessing global memory in a contiguous 128-byte line) is critical.
On Apple Silicon, the GPU reads 128-byte cache lines from LPDDR, similar to a CPU.
**Experiment:**
1.  Kernel A: `out[i] = in[i]` (Sequential).
2.  Kernel B: `out[i] = in[i * 128]` (Strided).
**Result:**
Both patterns punish bandwidth, but Apple's huge Cache Hierarchy (System Level Cache) often masks the stride penalty better than older Discrete GPUs, *if* the data prefetcher detects the stream. However, sequential is still 10x faster. **The rule "Coalesce your reads" remains true.**

---

## 📝 Summary & Key Takeaways

1.  **Refactoring Cost:** Porting logic is easy (MSL $\approx$ CUDA C++). Porting Host Code is hard (State Machine vs Object Graph).
2.  **Buffers for Scalars:** In Metal, you can't just pass `int n`. It must go into a `setBytes` call (which essentially creates a tiny constant buffer).
3.  **Compilation:** Metal compiles source to `.air` (Apple IR) then to `.metallib` at build time. At runtime, it compiles `.metallib` to machine code (PSO creation). This "PSO creation" step can take 100ms+ per kernel, so do it async or at splash screen.
4.  **No `printf`:** Use GPU Frame Capture to debug. `printf` support exists in Metal 3 but is tricky to setup compared to CUDA.

---

## 📚 Additional Resources

*   [Porting CUDA to Metal (Apple Guide)](https://developer.apple.com/documentation/metal/porting_your_code_to_metal)
*   [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

**Tomorrow:** Day 55 - Metal Debugging & Profiling... Instruments, GPU Trace, and performance counters.

*End of Day 054 - Total Lines: 1000+*
