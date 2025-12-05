# Day 33: CUDA Programming (GPU)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 5: Edge AI & Optimization

---

> **📝 Content Creator Instructions:**
> PyTorch hides the GPU. To truly optimize robotics (e.g., Particle Filters, Point Cloud Processing), we must write Kernels.
> - **Focus:** CUDA Architecture, Thread Hierarchy, and Memory Management.
> - **Code:** Writing a custom CUDA kernel for Lidar Point Processing.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Map** algorithms to the CUDA Grid-Block-Thread hierarchy.
2.  **Write** a `.cu` kernel to process array data in parallel ($O(1)$ time).
3.  **Manage** Host-to-Device data transfer (and minimize it).
4.  **Optimize** memory access using Shared Memory to avoid global memory latency.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU.

### Software Environment
```bash
sudo apt install nvidia-cuda-toolkit
pip install pycuda # For python orchestration
```

### Prior Knowledge
- C pointers.
- Parallel Processing concepts.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SIMT (Single Instruction Multiple Threads)

*   **CPU:** SAS (Single-thread optimized). 4 Cores. Huge Cache. Complex Logic.
*   **GPU:** Throughput optimized. 4000 Cores. Tiny Cache.
*   **Kernel:** A function that runs on *every core* simultaneously.
    *   `idx = blockIdx.x * blockDim.x + threadIdx.x`
    *   Each thread processes `data[idx]`.

### 🔹 Part 2: Memory Hierarchy

1.  **Global Memory (VRAM):** Huge (16GB), Slow (High Latency).
2.  **Shared Memory:** Tiny (48KB per block), Ultra-Fast (L1 speed). User programmable cache.
3.  **Registers:** Instant.
4.  **Strategy:** Load chunk from Global -> Shared. Threads sync. Compute using Shared. Write result to Global.

### 🔹 Part 3: Coalesced Access

GPU reads memory in 32-byte chunks (Transactions).
*   **Good:** Thread 0 reads `addr`, Thread 1 reads `addr+1`. (1 Transaction).
*   **Bad:** Thread 0 reads `addr`, Thread 1 reads `addr+100`. (32 Transactions).

---

## 💻 Implementation: Vector Addition (Hello World)

We will use PyCUDA for the host code (easier than C++ boilerplate).

### 🛠️ Project Structure
```text
day33_cuda/
├── kernels/
│   └── simple_math.cu
└── run_kernel.py
```

### 👨‍💻 Kernel (`kernels/simple_math.cu`)

```cpp
extern "C" {

// __global__ means called from CPU, runs on GPU
__global__ void add_vectors(float* A, float* B, float* C, int N) {
    // 1. Calculate Global Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Boundary Check
    if (tid < N) {
        // 3. Compute
        C[tid] = A[tid] + B[tid];
        
        // Example Complex Math: Sqrt, Sin, Exp are hardware accelerated
        // C[tid] = __sinf(A[tid]); 
    }
}

}
```

### 👨‍💻 Host Code (`run_kernel.py`)

```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# 1. Prepare Data
N = 1_000_000
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_c = np.zeros_like(h_a)

# 2. Allocate on Device
d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
d_c = cuda.mem_alloc(h_c.nbytes)

# 3. Copy Host -> Device
cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)

# 4. Compile Kernel
with open("kernels/simple_math.cu", "r") as f:
    mod = SourceModule(f.read())

func = mod.get_function("add_vectors")

# 5. Launch
# Block Size: 256 threads (standard)
# Grid Size: Ceil(N / 256)
block_dim = (256, 1, 1)
grid_dim = (int((N + 255) // 256), 1)

func(d_a, d_b, d_c, np.int32(N), block=block_dim, grid=grid_dim)

# 6. Copy Device -> Host
cuda.memcpy_dtoh(h_c, d_c)

# Verify
assert np.allclose(h_c, h_a + h_b)
print("CUDA Success!")
```

---

## 🔬 Lab Exercise: RGB to Grayscale

### 1. Lab Objectives
- Input: 4K Image (3840x2160x3). Flattened array.
- Formula: $Y = 0.299 R + 0.587 G + 0.114 B$.
- **Compare Speed:**
    - Python Loop: 5 seconds.
    - NumPy: 50 ms.
    - CUDA: 2 ms.

### 2. Kernel Hint
```cpp
__global__ void rgb2gray(unsigned char* img, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3; # RGB stride
        float r = img[idx];
        float g = img[idx+1];
        float b = img[idx+2];
        
        gray[y * width + x] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}
```

---

## 🚀 Project: "Lidar Filter"

**Goal:** Filter a PointCloud (100k points). Remove points where $z > 0.5$ (Ceiling) or $r < 0.5$ (Self-reflection).
1.  **Naive:** `for p in points: if check(p): new_list.append(p)`. (Slow CPU).
2.  **CUDA Approach:**
    *   Kernel `filter_mask` runs on all points. Writes `1` or `0` to a mask array.
    *   **Prefix Sum (Scan):** Use `Thrust` library (CUDA STL) to compact the array based on the mask.
3.  **Result:** Filter 1M points in < 1ms. Essential for 30Hz Lidar processing.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Launch Failure" (Error 700)
*   **Cause:** Segfault on GPU. Accessing array out of bounds (`tid >= N`).
*   **Fix:** Always include `if (tid < N)` check.
*   **Tool:** `cuda-memcheck ./my_program`.

#### 2. "Watchdog Timeout"
*   **Cause:** Kernel took too long (> 2s). Windows/Linux kills the driver assuming GPU hung.
*   **Fix:** Optimize code or split into smaller kernels.

---

## ⚡ Optimization: Unified Memory

Newer CUDA (6.0+) supports `cudaMallocManaged`.
*   Data is accessible by BOTH CPU and GPU pointer.
*   Driver automatically migrates pages over PCIe on demand.
*   **Pros:** Easy coding (No explicit memcpy).
*   **Cons:** Performance hit if page faults occur frequently.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Warp"?
    *   **A:** A group of 32 threads that execute the *exact same instruction* at the same time.
2.  **Q:** What happens if `if (tid < 16)` inside a Warp?
    *   **A:** **Warp Divergence.** All 32 threads execute the `if` path (masked off threads wait), then all 32 execute the `else` path. Performance halves. Avoid branching!
3.  **Q:** Why prefer shared memory?
    *   **A:** Latency. Global memory = 400 cycles. Shared = 20 cycles.

### Challenge Task
> **Task:** Matrix Multiplication.
> 1. Naive: Row * Col.
> 2. Tiled: Load 16x16 tile into Shared Memory. Compute partial product.
> 3. Compare performance for N=4096.

---

## 📚 Further Reading
- **CUDA Programming Guide:** NVIDIA.
- **Thrust Library:** High-Performance Algorithms (Sort, Scan, Reduce).
- **Numba:** JIT compile Python to CUDA.

---

**Day 33 Complete**
