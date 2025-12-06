# Day 33: OpenAI Triton - GPU Kernels in Python
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Master **OpenAI Triton** (the Language), the revolutionary compiler that allows you to write high-performance GPU kernels in standard Python, bypassing the complexity of CUDA C++.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between NVIDIA Triton Inference Server and OpenAI Triton Language.
2.  **Understand** Triton's "Block-Based" Programming Model (vs CUDA's SIMT Thread-Based model).
3.  **Write** a Vector Addition kernel using `@triton.jit`.
4.  **Implement** a Fused Softmax kernel that outperforms PyTorch native routines on large inputs.
5.  **Use** `tl.load` and `tl.store` with masking to handle boundary conditions safely.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Triton produces PTX).
- *Note: Newer AMD GPU support is improving, but we focus on NV.*

### Software Environment
```bash
# Install Triton (Linux/WSL only usually)
pip install triton torch
```

### Prior Knowledge
- CUDA Memory Coalescing (Triton handles this, but understanding it helps).
- Softmax Math: $e^{x_i} / \sum e^{x_j}$.

---

## 📖 Theoretical Foundation

### 1. The "Other" Triton

*   **NVIDIA Triton Inference Server:** A Deployment tool (Week 4).
*   **OpenAI Triton Language:** A Loop-based Language for writing Kernels (Week 5).
    *   *Why?* CUDA C++ is hard. You manage registers, shared memory banks, and barriers manually.
    *   *Triton:* You write loops over **Blocks** (Tiles). The compiler figures out how to map "Load a 32x32 block" to "Threads, Registers, Shared Memory, and Coalesced Reads".

### 2. Block-Based Programming

In CUDA:
```cpp
// You control one thread
int idx = threadIdx.x;
float val = data[idx]; // Single scalar load
```

In Triton:
```python
# You control a BLOCK of logical threads
# offsets is a vector [0, 1, ... 127]
offsets = program_id(0) * BLOCK_SIZE + arange(0, BLOCK_SIZE)
# Load a whole vector at once
val_vector = tl.load(ptr + offsets) 
```

Triton generates the optimized CUDA code to load that vector efficiently using the underlying hardware (L2 Cache, Shared Mem).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Vector Addition

Hello World in Triton.

#### 📁 `src/triton_vecadd.py`
```python
#!/usr/bin/env python3
"""
Day 33: Triton Vector Addition
Phase 6: DL Compiler Stack
"""

import torch
import triton
import triton.language as tl

# 1. Define Kernel
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to X (Input)
    y_ptr,  # Pointer to Y (Input)
    output_ptr, # Pointer to Output
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr, # Number of elements per block
):
    # Get Program ID (Like blockIdx.x)
    pid = tl.program_id(axis=0)
    
    # Calculate Offsets
    # block_start = pid * BLOCK_SIZE
    # offsets = [block_start, block_start+1, ... ]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask: Important! Check boundary to avoid out-of-bounds Read/Write
    mask = offsets < n_elements
    
    # Load data from GPU memory
    # If mask is False, load 0.0 (though for reading output it matters less if we don't write it back)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def run_vecadd():
    torch.manual_seed(0)
    size = 98432 # Arbitrary size
    
    # Input Tensors (Must be on GPU/CUDA)
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)
    
    # Grid Calculation
    # How many blocks do we need?
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    # Launch Kernel
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Verify
    expected = x + y
    print(f"Max Error: {torch.max(torch.abs(output - expected))}")
    assert torch.allclose(output, expected)
    print("Triton Vector Add Verified!")

if __name__ == "__main__":
    run_vecadd()
```

### 👨‍💻 Advanced: Fused Softmax

Softmax requires a `reduction` (Sum/Max). In CUDA, reducing across a block requires `__syncthreads` and warp shoveling. in Triton, `tl.sum` handles it.

#### 📁 `src/triton_softmax.py`
```python
#!/usr/bin/env python3
"""
Day 33: Triton Fused Softmax
"""

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # One block per row input (Simplification: assuming n_cols < BLOCK_SIZE)
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Offsets [0, 1, ... n_cols-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load Row
    # Note: mask=mask ensures we don't read garbage beyond n_cols
    # other=float("-inf") ensures padding doesn't affect Max calc
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float("-inf"))
    
    # 1. Compute Max (for numerical stability)
    row_minus_max = row - tl.max(row, axis=0)
    
    # 2. Compute Numerator (Exp)
    # Fast exp implementation inside Triton
    numerator = tl.exp(row_minus_max)
    
    # 3. Compute Denominator (Sum)
    denominator = tl.sum(numerator, axis=0)
    
    # 4. Compute Softmax
    softmax_output = numerator / denominator
    
    # Store
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)

def benchmark_softmax():
    # Setup
    rows, cols = 4096, 1024 # Cols must fit in BLOCK_SIZE for this simple kernel
    x = torch.randn(rows, cols, device='cuda')
    y_tri = torch.empty_like(x)
    
    # Launch
    # grid = (rows, ) -> One block per row
    BLOCK_SIZE = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](
        x, y_tri,
        x.stride(0), y_tri.stride(0),
        cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Verify
    y_torch = torch.softmax(x, dim=1)
    print(f"Max Error: {torch.max(torch.abs(y_tri - y_torch))}")
    assert torch.allclose(y_tri, y_torch, atol=1e-2, rtol=1e-2)
    print("Triton Softmax Verified!")
    
    # Benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'], x_vals=[1024, 2048, 4096, 8192], 
            line_arg='provider', line_vals=['triton', 'torch'],
            line_names=["Triton", "Torch"],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s', plot_name='softmax-bench', args={'M': 4096}
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        if provider == 'torch':
            return triton.testing.do_bench(lambda: torch.softmax(x, axis=1))
        if provider == 'triton':
            # Recompile for specific N
            BLOCK = triton.next_power_of_2(N)
            y = torch.empty_like(x)
            return triton.testing.do_bench(lambda: softmax_kernel[(M,)](
                x, y, x.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK))

    benchmark.run(show_plots=False, print_data=True)

if __name__ == "__main__":
    benchmark_softmax()
```

---

## 🔬 Lab Exercise: "Triton GELU"

### Task
Implement a GELU kernel in Triton.
Formula: `0.5 * x * (1 + tanh(...))`
1.  Port the math from Day 32 JAX example to Triton `tl.*` ops.
2.  Launch it element-wise (Like vector add).
3.  Benchmark against `torch.nn.functional.gelu`.

### Why?
Triton shines here. For bandwidth-bound ops like GELU, Triton matches manual CUDA performance with 1/10th the code lines.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Tile-Centric:** Triton forces you to think in "Tiles" (Blocks). You load a Tile, process a Tile, store a Tile. The compiler handles the messy "SIMT" details (threads, warps, shared memory bank conflicts).
2.  **Pointer Arithmetic:** You behave like a C programmer (`ptr + offset`), but the operations work on vectors.
3.  **Metaprogramming:** Python acts as the metaprogramming language. `BLOCK_SIZE` is passed as a `constexpr` (constant expression), allowing the JIT to generate heavily optimized code for *specific* block sizes ($1024$ vs $128$).

### API Summary
```python
@triton.jit
def kernel(ptr, ...):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(ptr + offs, mask=mask)
    res = tl.exp(val)
    tl.store(ptr + offs, res, mask=mask)
```

---

**Day 33 Complete** ✅

*Next: Day 34 - Graph Optimization & Fusion - How compilers like Relay (TVM) and XLA decide *what* to fuse.*
