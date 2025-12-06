# Day 29: Introduction to DL Compilers & Apache TVM
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Move beyond vendor-locked libraries (cuDNN) and explore the world of **Deep Learning Compilers**. Learn how **Apache TVM** generates optimized machine code for any hardware from high-level model definitions.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the difference between Library-based Backends (PyTorch/cuDNN) and Compiler-based Backends (TVM/XLA).
2.  **Understand** the TVM Stack: Relay (High-Level), TE (Tensor Expression), TIR (Low-Level), and Target.
3.  **Write** a Matrix Multiplication using TVM's `te` (Tensor Expression) API.
4.  **Schedule** the computation manually (Tile, Pack, Reorder) to improve performance.
5.  **Compile** the scheduled kernel to CUDA PTX and execute it.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Supported target for TVM).
- CPU (LLVM target).

### Software Environment
```bash
# Installing TVM is complex. Recommended method is Pre-built/Conda
pip install apache-tvm-cu12 -f https://tlcpack.ai/wheels
# OR via Conda
conda install -c tlcpack -c conda-forge tvm-cuda-12.0
```

### Prior Knowledge
- Loop Blocking/Tiling (Day 4: Shared Memory).
- CUDA Grid/Block hierarchy.

---

## 📖 Theoretical Foundation

### 1. Libraries vs. Compilers

*   **The Library Approach (PyTorch + cuDNN):**
    *   PyTorch breaks the graph into ops (`Conv2D`, `ReLU`).
    *   It calls `cudnnConv2d()` and `relu_kernel()`.
    *   *Pros:* Extremely fast for standard layers.
    *   *Cons:* "Op Fusion" is hard. (Conv+ReLU+Add requires writing a new kernel). New hardware requires waiting for NVIDIA to update cuDNN.
*   **The Compiler Approach (TVM/XLA):**
    *   Represents the graph as math expressions.
    *   "Lowers" expressions into loops.
    *   Optimizes loops (Fuse, Tile, Unroll).
    *   Generates machine code (LLVM/PTX).
    *   *Pros:* Automated fusion. Portable to any hardware (ARM, RISC-V, GPU).

### 2. The TVM Workflow

1.  **Relay/Relax:** High-level IR. Represents Neural Network graphs (DAGs).
2.  **TE (Tensor Expression):** Domain Specific Language to describe *logic* (e.g., `C[i,j] = sum(A[i,k]*B[k,j])`).
3.  **Scheduler:** Transforming the loops *without* changing the logic (e.g., `split(i, factor=32)`).
4.  **TIR:** Low-level IR (Loops, pointers, allocations).
5.  **Codegen:** Generates CUDA C++/PTX or LLVM Assembly.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Matrix Mul with TVM TE

We will implement MatMul and manually optimize the schedule to see how TVM maps loops to CUDA hardware.

#### 📁 `src/tvm_matmul.py`
```python
#!/usr/bin/env python3
"""
Day 29: Matrix Multiplication with Apache TVM
Phase 6: DL Compiler Stack
"""

import tvm
from tvm import te
import numpy as np

def run_tvm_demo():
    # 1. Define the Computation (Algorithm)
    # -------------------------------------
    M, N, K = 1024, 1024, 1024
    
    # Define Tensors (Symbolic placeholders)
    # A(M, K), B(K, N)
    A = te.placeholder((M, K), name='A', dtype='float32')
    B = te.placeholder((K, N), name='B', dtype='float32')
    
    # Reduction Axis (The 'k' loop in C code)
    k = te.reduce_axis((0, K), name='k')
    
    # Compute Definition: C[i, j] = sum(A[i, k] * B[k, j])
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name='C'
    )
    
    # 2. Create the Schedule (Optimization)
    # -------------------------------------
    s = te.create_schedule(C.op)
    
    # Baseline: Nested loops on CPU
    # print(tvm.lower(s, [A, B, C], simple_mode=True))
    
    # 3. Target: CUDA GPU
    # -------------------
    # To run on GPU, we must map loops to Block/Thread
    
    # block_x, thread_x, etc.
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_y = te.thread_axis("threadIdx.y")
    
    # Split the workload
    # i loop (rows/M) -> block_y, thread_y
    # j loop (cols/N) -> block_x, thread_x
    
    i, j = s[C].op.axis
    
    # Tiling factors (Tune these!)
    bn = 32 # Block Size
    
    # Split i axis into (i_outer, i_inner) by factor 32
    i_outer, i_inner = s[C].split(i, factor=bn)
    # Split j axis
    j_outer, j_inner = s[C].split(j, factor=bn)
    
    # Bind to CUDA axes
    s[C].bind(i_outer, block_y)
    s[C].bind(j_outer, block_x)
    s[C].bind(i_inner, thread_y)
    s[C].bind(j_inner, thread_x)
    
    # Inspect the Generated Lowered IR (TIR)
    print("\n[Generated TIR Code]")
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    
    # 4. Compile
    # ----------
    target = "cuda"
    dev = tvm.cuda(0)
    
    # Build kernel
    func = tvm.build(s, [A, B, C], target=target)
    
    # 5. Execute
    # ----------
    # Allocate memory
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    c_np = np.zeros((M, N), dtype=np.float32)
    
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)
    
    # Run
    func(a_tvm, b_tvm, c_tvm)
    
    # 6. Verify and Profile
    # ---------------------
    # Verification
    np.testing.assert_allclose(c_tvm.numpy(), np.dot(a_np, b_np), rtol=1e-5)
    print("\nVerification Passed!")
    
    # Profiling
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(f"Algorithm Time: {evaluator(a_tvm, b_tvm, c_tvm).mean * 1e3:.3f} ms")

if __name__ == "__main__":
    run_tvm_demo()
```

### 👨‍💻 Understanding the Schedule

The magic happens in `te.schedule`:
*   **Base:** 3 nested loops (M, N, K).
*   **Tiling:** We split M and N to fit into GPU Blocks.
*   **Binding:** We map logic `i_outer` to hardware `blockIdx.y`.

If you don't bind to hardware threads, TVM generates CPU code (sequential loops). If you bind incorrectly (too many threads), CUDA launch fails.

---

## 🔬 Lab Exercise: "Vector Add Schedule"

### Lab Objectives
1.  Implement Vector Add `C = A + B` using `te`.
2.  Inspect the default schedule (Serial loop).
3.  Apply `split` to break vector of size N=1024 into logic suitable for threads.
4.  Bind to `blockIdx.x` and `threadIdx.x`.

### Key Code snippet
```python
n = te.var("n")
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
C = te.compute(A.shape, lambda i: A[i] + B[i], name='C')

s = te.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=256) # 256 threads per block
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decouple Algorithm from Schedule:** In native CUDA, you write the algorithm (math) and schedule (tiling/threads) mixed in the C++ kernel. In TVM, you define Math (`te.compute`) once, and can apply 50 different Schedules (CPUs, GPUs, TPUs) to it.
2.  **TIR (Tensor IR):** The intermediate representation that looks like "C with macros". It handles pointers, allocation, and loops.
3.  **Correctness:** TVM ensures that as long as the Schedule primitives are valid (split, reorder), the mathematical result remains correct (though performance varies wildly).

### API Summary
```python
# define
C = te.compute((M,), lambda i: A[i] * 2)

# schedule
s = te.create_schedule(C.op)
xo, xi = s[C].split(axis, factor=32)
s[C].bind(xo, te.thread_axis("blockIdx.x"))

# build
func = tvm.build(s, [A, C], target="cuda")
```

---

**Day 29 Complete** ✅

*Next: Day 30 - AutoTVM & Ansor - Why write schedules by hand when AI can write them for you?*
