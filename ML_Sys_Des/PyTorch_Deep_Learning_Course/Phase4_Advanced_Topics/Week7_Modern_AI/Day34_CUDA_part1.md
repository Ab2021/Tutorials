# Day 34: CUDA & Triton - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Memory Coalescing, Tiling, and Flash Attention

## 1. Memory Coalescing

Global memory is accessed in transactions (e.g., 32 bytes).
*   **Coalesced**: Consecutive threads access consecutive memory addresses. 1 transaction.
*   **Uncoalesced**: Random access. Multiple transactions. Slow.
*   Triton handles this automatically for block loads.

## 2. Tiling (Block Blocking)

Matrix Multiplication $C = A \times B$.
Naive: Read row A, col B for every element.
**Tiled**:
1.  Load a small block of A and B into **SRAM** (Shared Memory).
2.  Compute partial product.
3.  Iterate.
*   Reuses data in SRAM. Drastically reduces HBM bandwidth.

## 3. Flash Attention (Triton Implementation)

Flash Attention is essentially a Tiled Softmax + MatMul kernel.
*   Loads blocks of Q, K, V into SRAM.
*   Computes Attention Score.
*   Updates Output.
*   Uses **Online Softmax** trick to avoid materializing the full $N \times N$ matrix.

## 4. Autotuning

Triton has `@triton.autotune`.
It runs the kernel with different configurations (`BLOCK_SIZE`, `num_warps`) and picks the fastest one for the specific hardware.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...): ...
```

## 5. PyTorch 2.0 `torch.compile`

Uses **TorchInductor** backend.
1.  Captures the graph.
2.  Fuses pointwise operations.
3.  Generates **Triton** kernels automatically.
4.  Compiles to GPU code.
