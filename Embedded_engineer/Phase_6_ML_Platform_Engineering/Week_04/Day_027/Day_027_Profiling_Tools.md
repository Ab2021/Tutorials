# Day 27: Deep Profiling with Nsight Systems & Nsight Compute
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Stop guessing and start measuring. Master the art of performance analysis using NVIDIA's premier profiling tools: **Nsight Systems** (for timeline/application scaling) and **Nsight Compute** (for kernel optimization).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Capture** application-wide traces using `nsys` (Nsight Systems).
2.  **Analyze** the timeline to identify CPU bottlenecks, memory transfer delays, and stream concurrency.
3.  **Profile** specific kernels using `ncu` (Nsight Compute) to inspect Warp efficiency and Memory throughput.
4.  **Interpret** the "Roofline Model" to determine if a kernel is Compute-Bound or Memory-Bound.
5.  **Use** NVTX (NVIDIA Tools Extension) to annotate code for clearer visualization.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU.
- Local machine with Nsight Systems/Compute GUI installed (to view reports from remote GPU).

### Software Environment
```bash
# Verify installation (usually compliant with CUDA Toolkit)
nsys --version
ncu --version

# Python NVTX Bindings
pip install nvtx
```

---

## 📖 Theoretical Foundation

### 1. The Hierarchy of Profiling

1.  **Level 0: `nvidia-smi`**: 
    *   *What it shows:* "Is the GPU on?" "How much VRAM?"
    *   *Limit:* 1Hz sample rate. Useless for kernels (microseconds).
2.  **Level 1: Nsight Systems (`nsys`)**:
    *   *What it shows:* **The Timeline**. When did the CPU launch the kernel? When did the memory copy finish? Are streams overlapping?
    *   *Use Case:* System tuning, removing bubbles, optimizing pipelines (DALI vs DataLoader).
3.  **Level 2: Nsight Compute (`ncu`)**:
    *   *What it shows:* **The Microscope**. Why did `my_kernel` take 50us? Was it L1 cache misses? Low warp occupancy?
    *   *Use Case:* Writing custom CUDA kernels (Weeks 1-2).

### 2. The Roofline Model

A visual graph plotting **Arithmetic Intensity** (FLOPS / Byte) vs **Performance** (GFLOPS).
*   **Memory Bound:** Under the slanted roof. Speed limited by VRAM Bandwidth.
*   **Compute Bound:** Under the flat roof. Speed limited by SM Clocks/Tensor Cores.
*   **Optimization Strategy:**
    *   *Memory Bound?* Reduce reads, improve caching, use Shared Memory.
    *   *Compute Bound?* Use Tensor Cores, faster math (FP16).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Code with NVTX Annotations

To make sense of the timeline, we must "name" our logical blocks (e.g., "Data Load", "Inference").

#### 📁 `src/profile_target.py`
```python
#!/usr/bin/env python3
"""
Day 27: Profiling Target with NVTX
Phase 6: Platform Engineering
"""

import torch
import nvtx
import time

# Create a naive kernel (Memory Bound)
# Adding a scalar to a large vector
def heavy_memory_op(N=50_000_000):
    t = torch.randn(N, device='cuda')
    return t + 1.0

# Create a compute heavy op (Compute Bound)
# Large Matrix Multiplication
def heavy_compute_op(N=4096):
    a = torch.randn(N, N, device='cuda')
    b = torch.randn(N, N, device='cuda')
    return torch.matmul(a, b)

def main():
    print("Warming up...")
    heavy_memory_op(1000)
    
    # 1. Nsight Systems Trace
    # We use NVTX ranges to label sections on the timeline
    
    # Range: "Memory Bound Phase"
    with nvtx.annotate("Memory Bound Ops", color="blue"):
        for i in range(5):
            heavy_memory_op()
            # Sync to force separation in simple view (Usually bad for perf, good for learning visualization)
            torch.cuda.synchronize()

    # Range: "Compute Bound Phase"
    with nvtx.annotate("Compute Bound Ops", color="red"):
        for i in range(5):
            heavy_compute_op()
            torch.cuda.synchronize()

    # Range: "CPU Bottleneck Simulation"
    with nvtx.annotate("CPU Sleep", color="yellow"):
        time.sleep(0.1) # The GPU is idle here! Look for a gap in the timeline.

if __name__ == "__main__":
    main()
```

### 👨‍💻 Profiling Workflow: Nsight Systems

**Command:**
```bash
# -t cuda,nvtx,osrt : Trace CUDA calls, NVTX markers, and OS Runtime (sleeps)
# --stats=true : Print summary text at end
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=report_day27 \
    --force-overwrite=true \
    --stats=true \
    python src/profile_target.py
```

**Analysis:**
1.  Open `report_day27.nsys-rep` in Nsight Systems GUI.
2.  Find the "NVTX" row. You will see Blue ("Memory Bound Ops") and Red ("Compute Bound Ops") bars.
3.  Look at the "CUDA HW" row below.
    *   Under Blue: You see many short `elementwise_kernel`. Gaps between them? (Kernel launch overhead).
    *   Under Red: Long `gemm` blocks. High utilization.
    *   Under Yellow: **Empty space**. CPU slept, GPU did nothing.

### 👨‍💻 Profiling Workflow: Nsight Compute

**Command:**
```bash
# --set full : Collect detailed metrics (Slow!)
# --kernel-name-base function : Filter specific kernels
ncu --set full \
    --output report_kernel \
    --force-overwrite \
    python src/profile_target.py
```

**Analysis:**
1.  Open `report_kernel.ncu-rep`.
2.  **Speed of Light (SOL):**
    *   *Memory SOL:* If 80%+, you are maximizing VRAM bandwidth.
    *   *SM SOL:* If 80%+, you are maximizing Compute.
3.  **Warp Stall Reasons:**
    *   *Long Scoreboard:* Waiting for Global Memory.
    *   *Math Pipe Throttle:* The ALUs are busy.

---

## 🔬 Lab Exercise: "The Bad Kernel"

### Lab Objectives
1.  Write a "Bad" Matrix Transpose (Naive Row/Col access causing uncoalesced reads).
2.  Write a "Good" Matrix Transpose (Using Shared Memory tiles).
3.  Profile both with `ncu` and compare **Global Memory Throughput**.

#### 📁 `src/transpose_profile.cu`
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define N 4096
#define TILE 32

// Bad: Stride issues. Reads coalesced (row), writes uncoalesced (col)
__global__ void transpose_naive(float *out, float *in) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    if (x < width && y < width) {
        out[x * width + y] = in[y * width + x];
    }
}

// Good: Coalesced read into Shared Mem, Coalesced write out
__global__ void transpose_shared(float *out, float *in) {
    __shared__ float tile[TILE][TILE+1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    // Load to shared (Coalesced)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // Transpose in shared
    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    // Write from shared (Coalesced)
    if (x < width && y < width) {
        out[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    float *d_in, *d_out;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    dim3 block(TILE, TILE);
    dim3 grid(N/TILE, N/TILE);

    // Profile me!
    transpose_naive<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();

    transpose_shared<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();
    
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

**Exercise:** Compile with `nvcc` and run `ncu`.
*Expectation:* `transpose_shared` should show 5-10x higher DRAM Throughput.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Always Profile First:** Do not optimize code until `nsys` proves it is the bottleneck. If the timeline shows the GPU is idle 50% of the time, optimizing the kernel is useless. Fix the valid pipeline first.
2.  **NVTX is Your Friend:** Annotating python/C++ code with user-readable labels ("Layer 1", "Loss Calc") makes the timeline instantly understandable.
3.  **Memory is Usually the Bottleneck:** For most unoptimized kernels, you will be Memory Bound. Check coalescing and shared memory usage using `ncu`.

### API Summary
```python
# NVTX Annotations
import nvtx
with nvtx.annotate("My Description", color="green"):
    do_work()
```
```bash
# Nsight Systems CLI
nsys profile --trace=cuda,nvtx --output=report ./app
# Nsight Compute CLI
ncu --set full ./app
```

---

**Day 27 Complete** ✅

*Next: Day 28 - Week 4 Review & Project - Optimizing a complete MaskRCNN pipeline using everything we learned.*
