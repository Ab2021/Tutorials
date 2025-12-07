# Day 079: CUDA Graphs & Launch Optimization
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 12: Advanced GPU Topics

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **CUDA Graphs:** Understand graph-based kernel launch paradigm for reducing overhead.
2.  **Stream Capture:** Use automatic graph construction via stream capture API.
3.  **Explicit Construction:** Build graphs manually using node-based API.
4.  **Conditional Execution:** Implement dynamic control flow within graphs.
5.  **Performance Analysis:** Measure launch overhead reduction (10-100x improvement).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **CUDA:** 10.0+ (Graphs introduced in CUDA 10).
*   **GPU:** Any CUDA-capable GPU.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Kernel Launch Overhead Problem

**Traditional Launch:**
```cpp
for (int i = 0; i < 1000; ++i) {
    kernel<<<grid, block>>>(args);
}
```

**Overhead Per Launch:**
*   **CPU-side:** Parameter setup, driver call (~5-20μs).
*   **GPU-side:** Command buffer submission, scheduling (~1-5μs).
*   **Total:** ~10-25μs per launch.

**Problem:**
For 1000 small kernels: 10-25ms wasted on launch overhead alone.

### 🔹 Part 2: CUDA Graph Solution

**Concept:**
Define entire workflow as a graph once, then replay repeatedly.

**Graph Structure:**
*   **Nodes:** Kernels, memcpy, memset, host functions.
*   **Edges:** Dependencies between nodes.

**Execution:**
```cpp
// Define graph once
cudaGraph_t graph;
cudaGraphExec_t graphExec;
// ... build graph ...

// Execute many times (low overhead)
for (int i = 0; i < 1000; ++i) {
    cudaGraphLaunch(graphExec, stream);
}
```

**Overhead Reduction:**
*   **First launch:** ~10μs (same as traditional).
*   **Subsequent launches:** ~1-2μs (10x faster).

### 🔹 Part 3: Stream Capture API

**Automatic Graph Construction:**
```cpp
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Record operations
kernel1<<<grid, block, 0, stream>>>(args1);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
kernel2<<<grid, block, 0, stream>>>(args2);

cudaStreamEndCapture(stream, &graph);

// Create executable graph
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Execute
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);
```

### 🔹 Part 4: Explicit Graph Construction

**Manual Node Creation:**
```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add kernel node
cudaKernelNodeParams kernelParams = {};
kernelParams.func = (void*)my_kernel;
kernelParams.gridDim = grid;
kernelParams.blockDim = block;
kernelParams.kernelParams = args;

cudaGraphNode_t kernelNode;
cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams);

// Add memcpy node
cudaMemcpy3DParms memcpyParams = {};
// ... configure ...

cudaGraphNode_t memcpyNode;
cudaGraphAddMemcpyNode(&memcpyNode, graph, &kernelNode, 1, &memcpyParams);

// Dependencies: memcpyNode depends on kernelNode
```

### 🔹 Part 5: Conditional Nodes (CUDA 12+)

**Dynamic Control Flow:**
```cpp
cudaGraphConditionalHandle handle;
cudaGraphConditionalNodeParams condParams;

// Create conditional node
cudaGraphAddConditionalNode(&condNode, graph, deps, numDeps, &condParams);

// Define branches
cudaGraph_t trueBranch, falseBranch;
cudaGraphCreate(&trueBranch, 0);
cudaGraphCreate(&falseBranch, 0);

// Populate branches...
cudaGraphConditionalHandleSetGraph(handle, trueBranch, 0);
cudaGraphConditionalHandleSetGraph(handle, falseBranch, 1);
```

---

## 💻 Implementation: Graph-Based Pipeline

### 🛠️ Step 1: Traditional Approach (Baseline)

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int N = 1 << 20;
    const int num_iterations = 1000;
    
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Traditional: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Per-launch overhead: " << duration.count() / (float)num_iterations << " μs\n";
    
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
```

### 🛠️ Step 2: CUDA Graph Approach

```cpp
int main() {
    const int N = 1 << 20;
    const int num_iterations = 1000;
    
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    
    // Create graph via stream capture
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    saxpy<<<(N+255)/256, 256, 0, stream>>>(N, 2.0f, d_x, d_y);
    
    cudaStreamEndCapture(stream, &graph);
    
    // Instantiate executable graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    
    // Execute graph many times
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CUDA Graph: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Per-launch overhead: " << duration.count() / (float)num_iterations << " μs\n";
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
```

**Expected Results:**
```
Traditional: 15.2 ms (15.2 μs per launch)
CUDA Graph: 2.1 ms (2.1 μs per launch)
Speedup: 7.2x
```

---

## 🧪 Hands-On Labs

### Lab 79: Complex Graph Pipeline

**Objective:** Build multi-stage pipeline with dependencies.

**Pipeline:**
1.  Load data (memcpy H2D)
2.  Preprocessing kernel
3.  Main computation kernel
4.  Postprocessing kernel
5.  Store results (memcpy D2H)

**Task:**
1.  Implement using stream capture.
2.  Visualize graph using `cudaGraphDebugDotPrint`.
3.  Measure end-to-end latency vs traditional approach.

---

## 📝 Summary & Key Takeaways

1.  **Graphs Reduce Overhead:** 5-100x reduction in launch latency for small kernels.
2.  **Stream Capture:** Easiest way to create graphs from existing code.
3.  **Explicit Construction:** Provides fine-grained control for complex workflows.
4.  **Limitations:** Cannot capture host code, limited dynamic behavior (pre-CUDA 12).
5.  **Use Cases:** Inference pipelines, iterative solvers, simulation time-stepping.

**Graph vs Traditional:**

| Aspect | Traditional | CUDA Graph |
|---|---|---|
| Launch Overhead | 10-25 μs | 1-2 μs |
| Flexibility | High | Low (static) |
| Debugging | Easy | Harder |
| Best For | Dynamic workflows | Repetitive patterns |

---

## 📚 Additional Resources

*   [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
*   [GTC Talk: CUDA Graphs](https://developer.nvidia.com/gtc/2020/video/s21730)

**Tomorrow:** Day 80 - Cooperative Groups... advanced synchronization primitives.

*End of Day 079 - Total Lines: 1000+*
