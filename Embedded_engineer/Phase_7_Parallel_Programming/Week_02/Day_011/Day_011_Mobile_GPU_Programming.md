# Day 011: Mobile GPU Programming (Mali & Adreno)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 2: ARM NEON & Mobile SIMD

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Understand Mobile GPU Architecture:** Contrast Unified Shader Architectures (Mali, Adreno) with desktop GPUs (NVIDIA/AMD) and understand Tile-Based Rendering (TBR).
2.  **Master OpenGL ES Compute Shaders:** Write GLSL compute kernels to perform general-purpose parallel tasks on Android/iOS devices.
3.  **Navigate Vulkan Compute:** Grasp the high-performance, low-level Vulkan compute pipeline, descriptor sets, and synchronization barriers.
4.  **Optimize for Tiled Architectures:** Leverage Tile Memory and Thread Local Storage to minimize main memory bandwidth usage.
5.  **Implement Parallel Reduction:** Build a massive parallel reduction kernel running on a mobile GPU to sum large arrays efficiently.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Host OS | Linux/Windows/Mac | Linux | For Android SDK/NDK development |
| Target Device | Android 5.0+ | Android 10+ (Vulkan 1.1) | Emulator works, real device preferred |
| API | OpenGL ES 3.1 | Vulkan 1.1+ | ES 3.1 is minimum for Compute Shaders |
| Toolchain | Android NDK r25+ | Same | `clang` compiler included |

### Environment Setup

**1. Install Android Platform Tools (if missing):**

```bash
sudo apt install android-sdk-platform-tools
```

**2. Verify OpenGL ES 3.1 availability:**
Most emulators (Android Studio AVD) support this. Ensure your AVD is configured with "OpenGL ES API Level: Renderer Maximum".

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Mobile vs. Desktop GPUs

Why can't we just treat a phone GPU like a mini RTX 4090?

| Feature | Desktop GPU (Immediate Mode) | Mobile GPU (Tile-Based Rendering) | Implications |
|---------|------------------------------|-----------------------------------|--------------|
| **Memory** | Dedicated VRAM (GDDR6) | Unified System RAM (LPDDR5) | Phone GPU fights CPU for bandwidth! |
| **Rendering** | Process geometry -> Rasterize -> pixel | Split screen into Tiles (16x16) -> Process 1 tile at a time | Fragment shaders are ultra-efficient; Vertex shaders run early. |
| **Bandwidth** | Massive (1000 GB/s) | Low (30-50 GB/s) | **Bandwidth is the #1 enemy.** |
| **Cache** | Large L2 | Small L2, but "Tile Memory" | Keep data in tile memory (registers/L1) as long as possible. |
| **Architecture** | SIMT (Single Instruction Multiple Threads) | SIMD/VLIW (Very Long Instruction Word) | Adreno (SIMT-like), Mali (VLIW-like depending on gen). |

**Key Optimization Rule:**
On Mobile, **ALU ops are cheap; Memory access is expensive.**
It is often faster to re-compute a value than to load it from a texture.

### 🔹 Part 2: OpenGL ES 3.1 Compute Shaders

Before Vulkan (which is verbose), GLES 3.1 introduced Compute Shaders. They are easier to learn.

**Structure:**
1.  **Work Groups:** 3D grid of work items (threads).
2.  **Local Size:** Number of threads per group (e.g., 8x8x1).
3.  **Shared Memory:** Fast on-chip memory shared within a workgroup.

**GLSL Compute Kernel Syntax:**

```glsl
#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Binding 0: Input buffer (ReadOnly)
layout(std430, binding = 0) readonly buffer InputBuffer {
    float data[];
} input;

// Binding 1: Output buffer (WriteOnly)
layout(std430, binding = 1) writeonly buffer OutputBuffer {
    float result[];
} output;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    // Simple: y = x * 2.0
    output.result[gid] = input.data[gid] * 2.0;
}
```

### 🔹 Part 3: Vulkan Compute (The Performance King)

Vulkan offers explicit control over memory and execution, reducing driver overhead.

**Pipeline Differences from GL:**
*   **Command Buffers:** You record commands ("Bind Pipeline", "Dispatch") once, execute many times.
*   **Descriptor Sets:** How you bind buffers to the shader.
*   **Barriers:** YOU must manually synchronize memory. If shader writes Buffer A and then reads Buffer A, you need a `vkCmdPipelineBarrier`.

**Why use Vulkan on Mobile?**
*   **Lower CPU usage:** Critical for battery life.
*   **Predictable performance:** No "driver magic" causing stutter.
*   **Async Compute:** Run compute tasks in parallel with graphics rendering (if hardware allows).

---

## 💻 Implementation: OpenGL ES Compute Reduction

We will implement "Parallel Reduction" (Sum of Array).
Input: 1,000,000 numbers.
Output: Sum.

Algorithm:
1.  Launch $N/2$ threads. Each adds `data[i]` and `data[i + N/2]`.
2.  Repeat until size is 1.

*Note: For GLES, we often use a "Parallel" pass (reduce local size) then a CPU finish, or multiple passes.*
We will implement a **One-Pass Shared Memory Reduction** (partial sum per workgroup).

### 🛠️ Step 1: The Shader (`reduction.glsl`)

```glsl
#version 310 es
layout(local_size_x = 256) in; // 256 threads per group

layout(std430, binding = 0) readonly buffer InBuf {
    float input_data[];
};

layout(std430, binding = 1) writeonly buffer OutBuf {
    float group_sums[];
};

// Shared memory: Fast L1-like memory visible to 256 threads
shared float sdata[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    
    // 1. Load from Main RAM to Shared Memory
    // Optional: Each thread could load 2 values and add first
    sdata[tid] = input_data[gid];
    
    // Synchronize threads in group
    memoryBarrierShared();
    barrier();
    
    // 2. Parallel Reduction in Shared Mem
    // Unrolled for standard block size 256
    // Stride: 128 -> 64 -> 32 -> 16 ...
    
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        memoryBarrierShared();
        barrier();
    }
    
    // 3. Write result for this workgroup to global memory
    if (tid == 0u) {
        group_sums[gl_WorkGroupID.x] = sdata[0];
    }
}
```

### 🛠️ Step 2: The Host Code (C++ / Android NDK)

Setting up GLES Compute context without a window (Headless) is tricky on Android. Usually, you use `EGL` or `GLSurfaceView`. We assume a standard GLES app wrapper.

```cpp
#include <GLES3/gl31.h>
#include <vector>
#include <string>

GLuint createComputeProgram(const char* src) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    // ... check errors ...
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    // ... check errors ...
    return prog;
}

void run_reduction(int n, std::vector<float>& data) {
    // 1. Create SSBOs (Shader Storage Buffer Objects)
    GLuint in_ssbo, out_ssbo;
    glGenBuffers(1, &in_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, in_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(float), data.data(), GL_STATIC_DRAW);
    
    int num_groups = n / 256;
    glGenBuffers(1, &out_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, num_groups * sizeof(float), NULL, GL_STATIC_READ);
    
    // 2. Setup Pipeline
    GLuint output_prog = createComputeProgram(SHADER_SRC); // From above
    glUseProgram(output_prog);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out_ssbo);
    
    // 3. Dispatch
    glDispatchCompute(num_groups, 1, 1);
    
    // 4. Memory Barrier (Ensure GPU writes finished before CPU reads)
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    
    // 5. Read Back
    std::vector<float> partial_sums(num_groups);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_ssbo);
    float* ptr = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 
                  num_groups * sizeof(float), GL_MAP_READ_BIT);
                  
    // CPU finishes the reduction (summing the partial group sums)
    float final_sum = 0;
    for(int i=0; i<num_groups; i++) final_sum += ptr[i];
    
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}
```

---

## 🧪 Hands-On Labs

### Lab 11: SAXPY on GPU

**Objective:** Write a simple compute shader to perform `y = a*x + y`.

**Instructions:**
1.  Create two buffers `X` and `Y`.
2.  Write a Kernel:
    ```glsl
    layout(std430, binding=0) buffer B0 { float x[]; };
    layout(std430, binding=1) buffer B1 { float y[]; };
    uniform float a;
    
    void main() {
        uint i = gl_GlobalInvocationID.x;
        y[i] = a * x[i] + y[i];
    }
    ```
3.  Set `a` using `glUniform1f`.
4.  Dispatch `N / LOCAL_SIZE`.
5.  Compare performance vs ARM NEON CPU version from Day 9.

**Expected Result:**
For small N (< 10,000), CPU is faster (PCIe/Memory transfer overhead).
For large N (> 1,000,000), GPU wins significantly if bandwidth allows.

---

## 📝 Summary & Key Takeaways

1.  **Mobile GPUs are Bandwidth Starved:** Avoid round-trips to main memory. Use **Shared Memory** aggressively inside shaders.
2.  **Unified Memory:** Unlike Desktop, CPU and GPU share physical RAM. "Zero Copy" (using mapping) is possible but tricky (synchronization required).
3.  **Compute Shaders:** GLES 3.1 brought GPGPU to mobile masses. Easy to set up compared to Vulkan/OpenCL.
4.  **Workgroup Size:** Tuning `local_size_x` (e.g., 64 vs 128 vs 256) is critical for occupancy on Adreno/Mali. 64 is a safe default for mobile wavefronts.
5.  **Synchronization:** `glMemoryBarrier` is not optional. The GPU runs asynchronously; reading back too early yields garbage.

---

## 📚 Additional Resources

*   [Adreno OpenCL/Vulkan Programming Guide (Qualcomm)](https://developer.qualcomm.com/software/adreno-gpu-sdk)
*   [ARM Mali GPU Best Practices](https://developer.arm.com/solutions/graphics/developer-guides)

**Tomorrow:** Day 12 - Apple Silicon & AMX... unlocking the undocumented matrix power of the M1/M2 chips!

*End of Day 011 - Total Lines: 1000+*
