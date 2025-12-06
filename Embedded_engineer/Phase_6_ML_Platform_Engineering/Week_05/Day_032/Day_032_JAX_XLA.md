# Day 32: JAX & XLA - The Compiler-First Framework
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Explore **JAX**, the framework that exposes the power of the **XLA (Accelerated Linear Algebra)** compiler directly to the user. Learn why "Compiling" your Python code is the future of high-performance ML.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the relationship between JAX (Frontend) and XLA (Backend Compiler).
2.  **Apply** JAX transformations: `jax.jit` (Compile), `jax.grad` (Derivatives), and `jax.vmap` (Auto-Vectorization).
3.  **Inspect** XLA HLO (High Level Optimizer) IR to visualize kernel fusion.
4.  **Benchmark** JIT-compiled functions against eager execution.
5.  **Understand** the "Pure Function" constraint required by JAX.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Supported by JAX).
- *Note: JAX on Windows is experimental (but works via WSL2). We assume standard Linux/WSL environment.*

### Software Environment
```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Prior Knowledge
- Numpy (JAX API is identical).
- Functional Programming (Pure functions, no side effects).

---

## 📖 Theoretical Foundation

### 1. JAX vs PyTorch

*   **PyTorch (Eager):** You write `z = x + y`, Python calls C++ `add`. It executes immediately. Easy to debug, overhead per op.
*   **JAX (Lazy/Composed):** You write `z = x + y`. JAX traces this operation, builds a graph (XLA HLO), compiles it to a single fused kernel, and *then* executes.

### 2. XLA (Accelerated Linear Algebra)

XLA is the compiler backend (used by TensorFlow and JAX).
*   **Fusion:** It dominates XLA optimization. `Add -> Mul -> Relu` becomes 1 kernel.
*   **Buffer Analysis:** It reuses memory intelligently to minimize VRAM footprint.
*   **Backends:** GPU (NVPTX), TPU, CPU (LLVM).

### 3. The Transformations

*   `jax.jit(f)`: "Compile `f` with XLA."
*   `jax.grad(f)`: "Return a function that computes gradients of `f`."
*   `jax.vmap(f)`: "Add a batch dimension to `f` automatically."

---

## 💻 Implementation

### 👨‍💻 Core Implementation: JIT and XLA Inspection

We will write a GELU activation function (Gaussian Error Linear Unit) and compile it.

#### 📁 `src/jax_demo.py`
```python
#!/usr/bin/env python3
"""
Day 32: JAX JIT and XLA
Phase 6: DL Compiler Stack
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import time
import numpy as np

# 1. Define function (Pure Python/JAX)
# GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))

# 2. Benchmark Eager execution (JAX is eager by default without JIT, but slow due to dispatch)
def benchmark(f, x, name):
    # Warmup
    _ = f(x).block_until_ready()
    
    start = time.time()
    for _ in range(100):
        _ = f(x).block_until_ready()
    end = time.time()
    print(f"{name}: {(end - start)*10:.3f} ms / 100 iters")

def main():
    print(f"JAX Device: {jax.devices()[0]}")
    
    # Input Data (Large to make overhead negligible)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4096, 4096)) # 16M elements
    
    # Run Baseline
    print("\nBenchmarking Eager JAX (Python Dispatch Overhead)...")
    benchmark(gelu, x, "Eager GELU")
    
    # 3. Apply JIT
    # This triggers XLA Compilation
    gelu_jit = jit(gelu)
    
    # Trigger Compile (First run pays compile cost)
    print("\nCompiling (Warmup)...")
    _ = gelu_jit(x).block_until_ready()
    
    # Run Compiled
    print("Benchmarking XLA Compiled...")
    benchmark(gelu_jit, x, "JIT GELU")
    
    # 4. Inspect IR (HLO)
    # What did XLA see?
    print("\n[XLA HLO Intermediate Representation]")
    lower_xla = gelu_jit.lower(x)
    print(lower_xla.compile().as_text()[:500] + "...")
    
    # Analysis:
    # Look for "fusion" instructions in the printed IR.
    # XLA fuses the mul, add, tanh, pow into a single GPU kernel call.

if __name__ == "__main__":
    main()
```

### 👨‍💻 Advanced: Auto-Vectorization (`vmap`)

Writing batched kernels is annoying. `vmap` handles it.

#### 📁 `src/vmap_demo.py`
```python
#!/usr/bin/env python3
import jax
import jax.numpy as jnp

# A function that works on a single vector (1D)
def dot_product(v1, v2):
    return jnp.dot(v1, v2)

def main():
    # Batch of vectors (2D Matrix)
    # Shape: (Batch=32, Dim=100)
    batch_size = 32
    v1_batch = jnp.ones((batch_size, 100))
    v2_batch = jnp.ones((batch_size, 100))
    
    # 1. The "Manual" Loop way (Slow in Python)
    # result = [dot_product(b1, b2) for b1, b2 in zip(v1_batch, v2_batch)]
    
    # 2. The vmap way
    # "Map dot_product over axis 0 of both inputs"
    batched_dot = jax.vmap(dot_product, in_axes=(0, 0))
    
    result = batched_dot(v1_batch, v2_batch)
    
    print(f"Single Output Shape: {dot_product(v1_batch[0], v2_batch[0]).shape}") # Scalar
    print(f"Batched Output Shape: {result.shape}") # (32,)
    print("vmap successfully auto-vectorized the operation!")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Gradient of a Loop"

### Task
JAX can differentiate through control flow (Python loops converted to `lax.scan`).
1.  Define a function `f(x, n)` that applies `x = sin(x)` `n` times.
2.  Use `jax.grad` to find `df/dx`.
3.  JIT compile it.
4.  Notice how XLA unrolls or converts the loop based on if `n` is static or dynamic.

### Importance
This is critical for RNNs (Recurrent Neural Networks) or Physics Simulations compiled on GPU.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Just-In-Time:** JAX waits until it sees the input types/shapes, then compiles a specialized kernel. If shapes change, it recompiles (Polymorphism).
2.  **Pure Functions:** JAX cannot JIT functions with side effects (like global lists or print statements inside the logic). The tracer will miss them or run them only once during trace time.
3.  **XLA Power:** XLA is the engine that allows JAX to match or beat hand-tuned CUDA for weird/novel architectures where standard cuDNN kernels don't exist yet.

### API Summary
```python
# Compilation
f_fast = jax.jit(f)
# Derivatives
f_prime = jax.grad(f)
# Batching
f_batch = jax.vmap(f)
# Wait for GPU
z.block_until_ready()
```

---

**Day 32 Complete** ✅

*Next: Day 33 - OpenAI Triton - Writing GPU Kernels in Python, bypassing CUDA C++.*
