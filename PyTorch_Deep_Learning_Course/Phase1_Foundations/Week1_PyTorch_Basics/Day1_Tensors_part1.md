# Day 1: PyTorch Tensors - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Internals, Dispatcher, and JIT

## 1. The PyTorch Dispatcher

When you call `torch.add(x, y)`, what actually happens?
PyTorch is not just a Python library; it's a complex C++ engine (ATen - A Tensor Library) wrapped in Python.

1.  **Python API**: `torch.add` is called.
2.  **Dispatcher**: Inspects the inputs.
    *   Are they CPU or CUDA tensors?
    *   Are they Quantized?
    *   Do they require Autograd?
3.  **Kernel Selection**: Based on the dispatch key, it routes the call to the specific C++ kernel (e.g., `add_kernel_cuda` or `add_kernel_cpu`).

This mechanism allows PyTorch to be extensible. You can register custom backends (e.g., TPU, MPS) without changing the core logic.

## 2. Memory Formats: NCHW vs NHWC

In Computer Vision, we deal with 4D tensors: `(Batch, Channels, Height, Width)`.
*   **NCHW (Contiguous)**: PyTorch default. `Stride: (C*H*W, H*W, W, 1)`.
*   **NHWC (Channels Last)**: `(Batch, Height, Width, Channels)`. `Stride: (H*W*C, W*C, C, 1)`.

**Why does it matter?**
NVIDIA Tensor Cores (hardware accelerators for matrix mul) are optimized for **NHWC**.
Using "Channels Last" memory format can boost performance by 20-30% on modern GPUs (Ampere/Hopper) because it allows better vectorization of the channel dimension (usually a multiple of 8 or 32).

```python
x = torch.randn(32, 3, 224, 224).cuda()

# Convert to Channels Last
x = x.to(memory_format=torch.channels_last)

# Conv2d layers automatically detect this and use optimized kernels
```

## 3. Tensor Cores and Precision

Standard `float32` (FP32) uses 32 bits.
**Tensor Cores** are specialized hardware units on NVIDIA GPUs that perform $D = A \times B + C$ in mixed precision (FP16/BF16) in a single clock cycle.

*   **FP32**: 23 bits mantissa, 8 bits exponent. High precision.
*   **FP16**: 10 bits mantissa, 5 bits exponent. Fast, but prone to overflow/underflow.
*   **BF16 (Brain Float)**: 7 bits mantissa, 8 bits exponent. Same range as FP32, lower precision. Best of both worlds for DL.

**Automatic Mixed Precision (AMP)**:
PyTorch `torch.amp` automatically casts tensors to FP16 for expensive ops (MatMul, Conv) and keeps FP32 for sensitive ops (Softmax, Sum), maintaining stability while gaining speed.

## 4. Zero-Copy Interop (DLPack)

How do you move data between PyTorch, TensorFlow, and JAX without copying?
**DLPack** is a standard intermediate in-memory representation.

```python
import torch
import jax.dlpack
import jax.numpy as jnp

# PyTorch Tensor
x = torch.randn(5, 5).cuda()

# To JAX (Zero Copy)
dlpack = torch.utils.dlpack.to_dlpack(x)
jax_array = jax.dlpack.from_dlpack(dlpack)

# Modifying jax_array modifies x!
```

## 5. Advanced Indexing: `register_buffer`

In `nn.Module`, we distinguish between:
*   **Parameters**: Learnable weights (gradients computed).
*   **Buffers**: State that is part of the model but not learned via SGD (e.g., Running Mean/Var in BatchNorm).

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(100))
```
Buffers are saved in the `state_dict` but `requires_grad=False` by default.

## 6. Pro Tip: `torch.compile` (PyTorch 2.0)

Python is slow. The interpreter adds overhead for every small operation.
`torch.compile` captures the graph of operations and fuses them.

**Kernel Fusion**:
Instead of:
1.  Read x, y -> Add -> Write temp
2.  Read temp -> Mul z -> Write result

Fusion does:
1.  Read x, y, z -> Compute (x+y)*z -> Write result

This reduces **Memory Bandwidth** usage, which is often the bottleneck in DL (not Compute).
