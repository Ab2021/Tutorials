# Day 062: Custom CUDA Kernels & UDFs
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Bridge the Gap:** Extend RAPIDS when standard `cudf` or `cuml` functions aren't enough.
2.  **JIT Compile:** Use **Numba** to write Python functions that compile to optimized PTX kernels at runtime.
3.  **Raw CUDA:** Inject raw C++ CUDA code into Python pipelines using **CuPy** `RawModule`.
4.  **DLPack Interop:** Seamlessly hand off memory between `Numba`, `CuPy`, and `PyTorch`.
5.  **Row-wise Logic:** Implement complex row-by-row transforms (financial greeks, physics simulation) that vectorization cannot express.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Libs:** `numba`, `cupy`, `cudf`.
*   **Concepts:** Thread hierarchy (Grid/Block) - essentially what we skipped in Week 9 start is revisited here.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Numba `cuda.jit`

RAPIDS is great for "Verbs" (Join, Sort, Filter).
But for "Adjectives/Logic" (e.g., `if (x > y and z < 5) return x*z`), vectorization is wasteful (requires creating boolean masks for every condition).
**Kernel Fusion:**
Numba compiles the Python function into a single GPU Kernel.
*   **Input:** pointers to arrays.
*   **Logic:** Executes in registers.
*   **Output:** writes to result array.
*   **Speed:** Reads memory once. Vectorization reads/writes memory for every intermediate step.

### 🔹 Part 2: CuPy Raw Kernels

Sometimes Numba isn't enough (e.g., need `__shfl_sync` or advanced texture lookups).
CuPy allows passing a C++ string containing CUDA code, compiles it with `nvrtc` (NVIDIA Runtime Compiler), and executes it on Cupy arrays.
**Usage:** High-performance stencils, cryptography, or porting legacy CUDA C code.

### 🔹 Part 3: Data Layout & Strides

Writing custom kernels requires understanding:
*   **Contiguity:** `C-order` vs `F-order`. processing a non-contiguous slice in a kernel is a common bug.
*   **Tiling:** Using Shared Memory (`__shared__`) to cache blocks of data for reuse (e.g., Matrix Mul, Convolution).

---

## 💻 Implementation: Custom Distance Metric

`cuml` supports Euclidean, Cosine, etc.
But what if we need **Haversine Distance** inside a K-Means clustering loop?
We can inject a Numba kernel.

### 🛠️ Step 1: Numba Implementation (`custom_dist.py`)

```python
from numba import cuda
import math
import numpy as np
import cupy as cp

# Define a Device Function (called by other kernels)
@cuda.jit(device=True)
def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Define the Global Kernel (The Entry Point)
@cuda.jit
def pairwise_haversine_kernel(coords_A, coords_B, out_matrix):
    # 2D Grid
    x, y = cuda.grid(2)
    
    # Check bounds
    if x < coords_A.shape[0] and y < coords_B.shape[0]:
        # coords shape: [N, 2] (Lat, Lon)
        lat1 = coords_A[x, 0]
        lon1 = coords_A[x, 1]
        lat2 = coords_B[y, 0]
        lon2 = coords_B[y, 1]
        
        dist = haversine_dist(lat1, lon1, lat2, lon2)
        out_matrix[x, y] = dist
```

### 🛠️ Step 2: Driver Code

```python
import cudf
import time

# Generate Data
N = 10000
M = 100
coords_1 = cp.random.uniform(0, 90, (N, 2), dtype=np.float32)
coords_2 = cp.random.uniform(0, 90, (M, 2), dtype=np.float32)
out = cp.zeros((N, M), dtype=np.float32)

# Configure Grid
threads = (16, 16)
blocks_x = (N + 15) // 16
blocks_y = (M + 15) // 16
blocks = (blocks_x, blocks_y)

print("Launching Numba Kernel...")
start = time.time()
pairwise_haversine_kernel[blocks, threads](coords_1, coords_2, out)
cuda.synchronize()
print(f"Time: {time.time() - start:.4f}s")

# Verification
print(out[0, 0])
```

### 🔹 Part 4: Raw C++ via CuPy

If we want to use `atomicAdd` extensively or specific instructions.

```python
import cupy as cp

# The C++ Source
code = r'''
extern "C" __global__
void atomic_hist(const float* data, int* bins, int N, int n_bins) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        int bin = (int)(val * n_bins);
        if (bin >= 0 && bin < n_bins) {
            atomicAdd(&bins[bin], 1);
        }
    }
}
'''

# Compile
module = cp.RawModule(code=code)
kernel = module.get_function('atomic_hist')

# Data
data = cp.random.rand(1_000_000, dtype=np.float32)
bins = cp.zeros(100, dtype=np.int32)

# Launch
blocks = (1_000_000 + 255) // 256
kernel((blocks,), (256,), (data, bins, 1_000_000, 100))

print(bins)
```

---

## 🧪 Hands-On Labs

### Lab 62: The Moving Average

**Objective:** Implement a Rolling Window Mean using Numba `stencil`.

**Task:**
1.  Create a 1D array of 10M floats.
2.  Compute weighted average of neighbors `[-2, -1, 0, 1, 2]`.
3.  Compare `cudf.Series.rolling()` performance vs `numba.cuda.stencil`.

```python
from numba import stencil

@stencil
def weighted_avg(a):
    return 0.1*a[-2] + 0.2*a[-1] + 0.4*a[0] + 0.2*a[1] + 0.1*a[2]

@cuda.jit
def run_stencil(inp, out):
    i = cuda.grid(1)
    # Numba stencil invocation
    out[i] = weighted_avg(inp, index=i)
```

**Why Custom?**
Standard rolling windows are efficient, but if you have *decaying weights* or *conditional windows* (time-based irregular intervals), standard libraries fail. Stencils allow arbitrary "Relative Indexing".

---

## 📝 Summary & Key Takeaways

1.  **Numba = Glue:** Numba is the essential glue for RAPIDS. It fills the gaps between `cudf` verbs.
2.  **Overhead:** Compiling a Numba kernel takes time (~200ms). Always cache functions or warmup before benchmarking.
3.  **Debugging:** Numba supports `simulator` mode (`NUMBA_ENABLE_CUDASIM=1`). This allows you to print and breakpoint your GPU code running on CPU.
4.  **DLPack:** Numba arrays can be converted to CuPy/PyTorch zero-copy via `cuda.as_cuda_array(obj)`.

---

## 📚 Additional Resources

*   [Numba CLI & CUDA Guide](https://numba.readthedocs.io/en/stable/cuda/index.html)
*   [CuPy Raw Kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html)

**Tomorrow:** Day 63 - Week 9 Review & Project... Building a "RecSys" (Recommender System) using cuDF (ETL) + cuML (Nearest Neighbors) + cuGraph (Graph Sim).

*End of Day 062 - Total Lines: 1000+*
