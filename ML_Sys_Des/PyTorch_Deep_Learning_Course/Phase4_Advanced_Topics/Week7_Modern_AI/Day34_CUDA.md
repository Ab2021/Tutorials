# Day 34: Custom Kernels (CUDA & Triton) - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: GPU Architecture, CUDA, and OpenAI Triton

## 1. Theoretical Foundation: GPU Architecture

*   **SM (Streaming Multiprocessor)**: The core unit.
*   **Threads & Blocks**: Threads are grouped into Blocks. Blocks are mapped to SMs.
*   **Memory Hierarchy**:
    *   **HBM (Global Memory)**: Huge (80GB), Slow.
    *   **SRAM (Shared Memory)**: Tiny (100KB), Fast. User-managed cache.
    *   **Registers**: Fastest.

**The Goal**: Keep data in SRAM/Registers. Minimize HBM access.

## 2. CUDA (Compute Unified Device Architecture)

C++ extension. Hard to write.
Requires manual memory management, synchronization (`__syncthreads()`), and tiling.

## 3. OpenAI Triton

Python-like language for writing kernels.
Compiles to PTX (CUDA assembly).
Automates memory coalescing and shared memory management.
Used in PyTorch 2.0 (`torch.compile`).

## 4. Implementation: Vector Addition in Triton

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first vector
    y_ptr,  # Pointer to second vector
    output_ptr, # Pointer to output
    n_elements, # Size of vector
    BLOCK_SIZE: tl.constexpr, # Meta-parameter
):
    # 1. Identify which program instance (thread block) this is
    pid = tl.program_id(axis=0)
    
    # 2. Identify offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. Mask (boundary check)
    mask = offsets < n_elements
    
    # 4. Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 5. Compute
    output = x + y
    
    # 6. Store
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch
def add(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

## 5. Fused Kernels

Why write kernels? **Fusion**.
PyTorch: `x * y + z` reads/writes HBM 3 times.
Kernel: Load x, y, z. Compute. Store result. 1 read/write cycle.
Speedup: 2x-5x.
