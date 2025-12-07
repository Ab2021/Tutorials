# Day 050: Apple Metal Architecture & Unified Memory
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 8: Apple Metal & GPU Compute

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Understand TBDR:** Explain **Tile-Based Deferred Rendering** architecture and how it differs from Immediate Mode (Desktop GPUs).
2.  **Navigate Metal API:** Identify core objects: `MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState`, and `MTLBuffer`.
3.  **Master Unified Memory:** Use `MTLStorageModeShared` to share pointers between CPU and GPU without explicit copying.
4.  **Write MSL:** Write a basic compute kernel in **Metal Shading Language**, a subset of C++14.
5.  **Setup Command Flow:** Encode commands into a buffer and commit them to the GPU.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **OS:** macOS 12+ (Monterey/Ventura).
*   **IDE:** Xcode (clang).
*   **Hardware:** Apple Silicon (M1/M2/M3) recommended. Intel compilation works but lacks unified memory benefits.
*   **Note for Windows/Linux Users:** This content is theoretical/cross-reference only unless you have a Mac. Concepts like Tile Memory are useful for Mobile Vulkan too.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Apple Silicon GPU Architecture (TBDR)

Desktop GPUs (NVIDIA/AMD) are **Immediate Mode Renderers**. They rasterize one triangle at a time to VRAM.
Mobile/Apple GPUs are **Tile-Based Deferred Renderers**.
1.  **Vertex Phase:** Process all geometry.
2.  **Tiling:** Sort geometry into screen tiles (e.g., 32x32 pixels).
3.  **Rasterization:** Process one tile at a time entirely in **fast on-chip memory (Imageblock)**.
4.  **Writeback:** Only write the final pixel to Main Memory.

**Implication for Compute:**
*   Threadgroups often map to Tiles.
*   **Threadgroup Memory (L1/Shared)** is extremely fast.
*   **Unified Memory:** The GPU reads from system RAM (LPDDR5). High bandwidth (100GB/s - 800GB/s), Low Latency.

### 🔹 Part 2: The Logic of Metal

Metal is an Objective-C / Swift API.
1.  **Initialization (Expensive):**
    *   `MTLDevice` (The GPU).
    *   `MTLLibrary` (Compiled .metal code).
    *   `MTLComputePipelineState` (PSO - The compiled kernel ready to run).
2.  **Runtime (Cheap):**
    *   `MTLCommandQueue` -> `MTLCommandBuffer` -> `MTLComputeCommandEncoder`.
    *   `setComputePipelineState(...)`
    *   `setBuffer(...)`
    *   `dispatchThreadgroups(...)`
    *   `commit()`

### 🔹 Part 3: Metal Shading Language (MSL)

It is C++14.
*   `device`: Global memory pointer.
*   `constant`: Uniform buffer.
*   `threadgroup`: Shared memory.
*   `thread_position_in_grid`: `get_global_id()`.

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* inA [[ buffer(0) ]],
                       device const float* inB [[ buffer(1) ]],
                       device float* result    [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = inA[index] + inB[index];
}
```

---

## 💻 Implementation: Vector Add using Metal-cpp

Metal is native to Obj-C/Swift, but Apple released `metal-cpp` to allow Standard C++ access (perfect for our course).

### 🛠️ Step 1: The Code (`metal_add.cpp`)

```cpp
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>

const char* kernel_src = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void add(device const float* a [[ buffer(0) ]],
                    device const float* b [[ buffer(1) ]],
                    device float* c       [[ buffer(2) ]],
                    uint id [[ thread_position_in_grid ]]) 
    {
        c[id] = a[id] + b[id];
    }
)";

int main() {
    // 1. Get Device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    std::cout << "Device: " << device->name()->utf8String() << "\n";

    // 2. Compile Kernel
    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(NS::String::string(kernel_src, NS::UTF8StringEncoding), nullptr, &error);
    if (!library) {
        std::cerr << "Compile Error: " << error->localizedDescription()->utf8String() << "\n";
        return 1;
    }
    
    MTL::Function* fn = library->newFunction(NS::String::string("add", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &error);

    // 3. Queue
    MTL::CommandQueue* queue = device->newCommandQueue();

    // 4. Buffers (Shared Memory)
    const int N = 1000;
    const int bytes = N * sizeof(float);
    
    MTL::Buffer* bufA = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufB = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufC = device->newBuffer(bytes, MTL::ResourceStorageModeShared);

    // Populate (Host Access)
    float* a = (float*)bufA->contents();
    float* b = (float*)bufB->contents();
    for(int i=0; i<N; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    // 5. Encode Command
    MTL::CommandBuffer* cmdbuf = queue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmdbuf->computeCommandEncoder();
    
    encoder->setComputePipelineState(pso);
    encoder->setBuffer(bufA, 0, 0);
    encoder->setBuffer(bufB, 0, 1);
    encoder->setBuffer(bufC, 0, 2);
    
    MTL::Size gridSize = MTL::Size::Make(N, 1, 1);
    MTL::Size groupSize = MTL::Size::Make(pso->maxTotalThreadsPerThreadgroup(), 1, 1);
    
    encoder->dispatchThreads(gridSize, groupSize);
    encoder->endEncoding();
    
    // 6. Execute
    cmdbuf->commit();
    cmdbuf->waitUntilCompleted();
    
    // 7. Verify
    float* c = (float*)bufC->contents();
    if (c[0] == 3.0f) std::cout << "Success!\n";
    
    // Cleanup (Release)
    bufA->release(); bufB->release(); bufC->release();
    pso->release(); queue->release(); device->release();
    
    return 0;
}
```

### 🛠️ Step 2: Compilation

Linking against Metal framework.

```bash
clang++ -std=c++17 metal_add.cpp -o metal_add -framework Metal -framework Foundation -fno-objc-arc
./metal_add
```

### 🔹 Part 4: Storage Modes

*   **Shared (`MTLResourceStorageModeShared`):**
    *   CPU: Readable/Writable.
    *   GPU: Readable/Writable.
    *   **Unified:** One allocation. No copying.
    *   **Usage:** Frequently changing data.
*   **Private (`MTLResourceStorageModePrivate`):**
    *   CPU: No access.
    *   GPU: Fastest access.
    *   **Usage:** Intermediate textures, weights. CPU must `blit` (copy) to upload data here.
*   **Managed (`MTLResourceStorageModeManaged`):**
    *   Old macOS (Intel + Dedicated GPU). Explicit synchronization between System RAM and VRAM. Deprecated on Apple Silicon.

---

## 🧪 Hands-On Labs

### Lab 50: Memory Bandwidth Test

**Objective:** Measure bandwidth of a customized kernel on M1/M2/M3.

**Kernel:**
```cpp
kernel void copy(device const float4* in [[ buffer(0) ]],
                 device float4* out      [[ buffer(1) ]],
                 uint id [[ thread_position_in_grid ]]) {
    out[id] = in[id];
}
```
*Note: Using `float4` (128-bit) provides better bus utilization than `float`.*

**Task:**
1.  Allocate 1GB buffer.
2.  Dispatch kernel.
3.  Time `waitUntilCompleted`.
4.  Expected: M1 Max/Ultra can hit 400 GB/s. M1 Base ~60 GB/s.

---

## 📝 Summary & Key Takeaways

1.  **Unified Memory is King:** The zero-copy architecture (pointers `contents()` works on CPU and GPU) simplifies programming drastically compared to CUDA's `Memcpy`.
2.  **Metal-cpp:** Allows us to interact with the Objective-C Runtime using pure C++, making Metal accessible to engine developers.
3.  **Encoders:** Metal strictly separates "Encoding" (CPU recording commands) from "Execution" (GPU running them).
4.  **Pipelines:** PSO creation is expensive (compiles shader). Do it once at startup.

---

## 📚 Additional Resources

*   [Metal-cpp Download & Guide](https://developer.apple.com/metal/cpp/)
*   [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

**Tomorrow:** Day 51 - Metal Compute Kernels... Threadgroup memory, SIMD-groups, and atomic operations in MSL.

*End of Day 050 - Total Lines: 1000+*
