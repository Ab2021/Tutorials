# Day 138: CUDA for Robotics (Parallel Perception)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> Pixels are parallel.
> - **Focus:** GPU Architecture (Streaming Multiprocessors, Warps, Threads), CUDA Programming Model, Numba for Python speedup, and why Lidar/Image processing belongs on the GPU.
> - **Code:** A Python script `cuda_lidar.py` utilizing `numba.cuda` to filter a 1,000,000 point cloud in parallel (Ground Plane Removal) vs a CPU implementation.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Grid-Block-Thread hierarchy in CUDA.
2.  **Identify** memory bottlenecks: Global Memory (Slow) vs Shared Memory (Fast).
3.  **Write** a Custom CUDA Kernel in Python using Numba.
4.  **Accelerate** Lidar processing by 100x compared to Python For-Loops.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Simulated/Mocked if not available, but code requires `numba` + `cudatoolkit`).
- If no GPU: Reads can follow the logic, but runtime verification fails.

### Software Environment
```bash
pip install numba numpy
```

### Prior Knowledge
- Parallelism.
- Point Clouds (XYZ data).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The GPU is a Throughput Beast

*   **CPU:** 16 Cores, complex logic (Branch Prediction), low latency.
*   **GPU:** 4096 Cores, simple logic, massive throughput.
*   **Warp:** 32 threads executing the *same instruction* at the same time (SIMT). If one thread has an `if/else` divergence, the other 31 threads wait.

### 🔹 Part 2: CUDA Hierarchy

1.  **Kernel:** The function running on the GPU.
2.  **Thread:** One execution instance.
3.  **Block:** A group of threads (share L1 Cache/Shared Mem).
4.  **Grid:** A group of blocks.

### 🔹 Part 3: Memory Matters

*   **Global Memory:** Big DRAM (VRAM). Slow access (High Latency).
*   **Shared Memory:** User-managed L1 Cache. Very fast.
*   **Coalescing:** Threads should access consecutive memory addresses $(i, i+1, ...)$ to maximize bandwidth.

---

## 💻 Implementation: GPU Lidar Filter

We filter points below $Z < -1.0$ (Remove floor).

### 🛠️ Project Structure
```text
day138_cuda/
├── src/
│   ├── cuda_lidar.py
└── output/
    ├── benchmark_cuda.png
```

### 👨‍💻 Numba CUDA Kernel (`src/cuda_lidar.py`)

```python
import numpy as np
from numba import cuda, float32
import math
import time
import matplotlib.pyplot as plt

# 1. The Kernel (Runs on GPU)
@cuda.jit
def remove_ground_kernel(points, out_mask, z_threshold):
    # Calculate global thread ID
    # ix = blockIdx.x * blockDim.x + threadIdx.x
    ix = cuda.grid(1)

    # Boundary check prevents reading garbage
    if ix < points.shape[0]:
        z_val = points[ix, 2] # Read Z
        
        # Simple Filter Logic
        if z_val > z_threshold:
            out_mask[ix] = 1
        else:
            out_mask[ix] = 0

def run_cpu(points, z_thresh):
    start = time.time()
    # NumPy Masking (Already optimized C, but single threaded often)
    mask = points[:, 2] > z_thresh
    end = time.time()
    return (end - start) * 1000 # ms

def run_gpu(points, z_thresh):
    n = points.shape[0]
    
    # Configure Grid
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    
    start = time.time()
    
    # 1. Copy to Device (Costly!)
    d_points = cuda.to_device(points)
    d_mask = cuda.device_array(n, dtype=np.uint8)
    
    # 2. Launch Kernel
    remove_ground_kernel[blocks_per_grid, threads_per_block](d_points, d_mask, z_thresh)
    cuda.synchronize() # Wait for finish
    
    # 3. Copy back
    h_mask = d_mask.copy_to_host()
    
    end = time.time()
    return (end - start) * 1000 # ms

def main():
    # Check GPU
    if not cuda.is_available():
        print("No GPU detected. Skipping benchmark.")
        return

    print("Generating Point Cloud...")
    N = 10_000_000 # 10 Million points
    points = np.random.randn(N, 3).astype(np.float32) # Random XYZ
    z_thresh = -0.5
    
    print(f"Benchmarking Ground Removal on {N} points...")
    
    # 1. CPU
    t_cpu = run_cpu(points, z_thresh)
    print(f"CPU Time: {t_cpu:.2f} ms")
    
    # 2. GPU (Cold Start - Includes compilation)
    run_gpu(points[:1000], z_thresh) # Warmup
    
    t_gpu = run_gpu(points, z_thresh)
    print(f"GPU Time: {t_gpu:.2f} ms")
    
    print(f"Speedup: {t_cpu / t_gpu:.1f}x")
    
    # Plot
    plt.bar(['CPU (NumPy)', 'GPU (Numba)'], [t_cpu, t_gpu], color=['red', 'green'])
    plt.ylabel('Time (ms)')
    plt.title(f'Filtering {N} Points')
    plt.savefig("output/benchmark_cuda.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Overhead"

### 1. Lab Objectives
- **Run:** The Benchmark.
- **Observe:**
    *   For $N=1000$, GPU might be *slower*.
    *   For $N=10,000,000$, GPU wins huge.
- **Why?** PCIe Transfer time. Copying data to GPU takes time. Kernel execution is instant.
- **Lesson:** Only use GPU for massive data or compute-heavy tasks (Convolutions, Matrix Mult). Memory transfer is the enemy.

---

## 🚀 Project: "GPU Euclidean Clustering"

**Goal:** Group points into objects using GPU.
1.  **Challenge:** Clustering is searching neighbors.
2.  **Grid:** Build a Voxel Grid in GPU Memory.
3.  **Kernel:** Each thread checks 26 neighbors in the grid.
4.  **Result:** Fast segmentation of cars/pedestrians from Lidar.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Kernel Panic / Segfault"
*   **Cause:** Determining Array bounds. `if ix < n` is mandatory.
*   **Fix:** Always guard against out-of-bounds access.

#### 2. "Wrong Answer on GPU"
*   **Cause:** Race Conditions. Threads writing to same index simultaneously.
*   **Fix:** Use `cuda.atomic.add` for reduction operations (Sum, Count).

---

## ⚡ Optimization: Zero-Copy (Unified Memory)

Jetson devices (Orin/Xavier) share RAM between CPU and GPU.
*   **Standard:** `cudaMemcpy` (CPU RAM $\to$ GPU VRAM).
*   **Unified:** `cudaMallocManaged`. Pointer works on both. No copy needed (Hardware handles paging).
*   **Result:** Huge speedup for Robotics pipelines.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Warp Divergence"?
    *   **A:** When `if (condition)` causes half the warp to go `true` and half `false`. The warp executes *both* branches sequentially, masking off threads. Performance halves.
2.  **Q:** Why is NumPy fast on CPU?
    *   **A:** It uses SIMD (AVX/SSE) instructions and BLAS libraries (C/Fortran). But GPU is SIMT (Single Instruction Multiple Threads) on steroids.
3.  **Q:** Kernel Launch Overhead?
    *   **A:** ~5-10 $\mu$s. If your kernel runs for 1 $\mu$s, you are wasting 90% time launching it. Fuse kernels!

### Challenge Task
> **Task:** Matrix Multiplication.
> 1. Write `matmul_kernel`.
> 2. Use Shared Memory tiles to reduce Global Memory traffic.
> 3. Compare with `np.dot`.

---

## 📚 Further Reading
- **Numba Docs:** CUDA for Python.
- **NVIDIA:** "CUDA C++ Programming Guide" (The Bible).

---

**Day 138 Complete**
