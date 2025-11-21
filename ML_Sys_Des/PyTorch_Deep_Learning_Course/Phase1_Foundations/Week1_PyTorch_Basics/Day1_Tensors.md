# Day 1: PyTorch Tensors - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Tensors, Memory Layout, and Hardware Acceleration

## 1. Theoretical Foundation: What is a Tensor?

### Mathematical Definition
In mathematics (specifically multilinear algebra), a tensor is a geometric object that maps a set of vectors and dual vectors to a scalar. It is defined by its **rank** (or order):
*   **Rank 0**: Scalar (Magnitude only). $x \in \mathbb{R}$
*   **Rank 1**: Vector (Magnitude + Direction). $x \in \mathbb{R}^n$
*   **Rank 2**: Matrix (Linear Map). $x \in \mathbb{R}^{n \times m}$
*   **Rank $k$**: $k$-dimensional array.

### Computer Science Definition
In Deep Learning, a "Tensor" is a generalization of a matrix to $N$ dimensions. It is a container for numerical data, optimized for:
1.  **Parallel Computation**: Operations on tensors can be parallelized across thousands of GPU cores.
2.  **Automatic Differentiation**: Tensors track their history to compute gradients.

## 2. Memory Layout: The "View" vs "Storage" Paradigm

This is the most critical concept for understanding PyTorch performance.

### The Storage
The `Storage` is a **contiguous** 1D array of bytes in physical memory (RAM or VRAM). It holds the actual numbers.

### The View (Tensor)
The `Tensor` object is just a lightweight wrapper containing **metadata** describing how to interpret the storage:
*   **Size (Shape)**: The dimensions (e.g., $3 \times 4$).
*   **Stride**: The number of elements to skip in memory to move to the next index in a specific dimension.
*   **Offset**: The starting index in the storage.

$$ \text{Address}(i, j) = \text{Base} + \text{Offset} + i \times \text{Stride}[0] + j \times \text{Stride}[1] $$

### Why does this matter?
Operations like `transpose`, `view`, and `slice` are **zero-copy**. They simply manipulate the strides and shape, leaving the massive storage array untouched. This makes them $O(1)$ in time complexity.

```python
import torch

# Create a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# Storage: [1, 2, 3, 4, 5, 6]
# Shape: (2, 3)
# Stride: (3, 1) -> To go down a row, skip 3. To go right, skip 1.

# Transpose
y = x.t()
# Storage: [1, 2, 3, 4, 5, 6] (UNCHANGED!)
# Shape: (3, 2)
# Stride: (1, 3) -> To go down a row, skip 1. To go right, skip 3.
```

## 3. Contiguity and Performance

A tensor is **contiguous** if its elements are stored in memory in the same order as they are iterated (row-major).
*   **CPU/GPU Cache**: Hardware fetches memory in "cache lines" (chunks). Accessing contiguous memory is fast (Cache Hit). Jumping around (large strides) causes Cache Misses.
*   **Vectorization**: SIMD (Single Instruction, Multiple Data) instructions require contiguous data.

**Critical Rule**: Many PyTorch operations (like `view`) require contiguous tensors. If a tensor is non-contiguous (e.g., after a transpose), you must call `.contiguous()` which forces a memory copy to rearrange the data.

## 4. Implementation: Basic Operations

```python
import torch

# 1. Initialization
x = torch.randn(3, 3) # Standard Normal Distribution
y = torch.zeros(3, 3)

# 2. Math
z = x + y       # Element-wise addition
z = x @ y       # Matrix Multiplication (Dot Product)
z = x * y       # Element-wise multiplication (Hadamard Product)

# 3. Broadcasting (The Magic)
# How to add (3, 3) and (3, 1)?
# PyTorch "stretches" the (3, 1) tensor to match (3, 3) without copying data.
a = torch.randn(3, 3)
b = torch.randn(3, 1)
c = a + b 
```

## 5. Hardware Acceleration: CUDA

PyTorch allows explicit control over device placement.
*   **Host**: CPU (System RAM). Good for sequential logic, data loading.
*   **Device**: GPU (VRAM). Good for massive parallel matrix math.

**Data Transfer Cost**: Moving data between CPU and GPU (PCIe bus) is slow. Minimize transfers. Keep data on GPU as long as possible.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move to GPU (Allocates VRAM and copies data)
x_gpu = x.to(device)

# Compute on GPU
y_gpu = x_gpu @ x_gpu

# Move back to CPU (Synchronizes and copies)
y_cpu = y_gpu.cpu()
```

## 6. Advanced: Einstein Summation

`einsum` provides a concise way to express multi-dimensional tensor operations using index notation. It is often more efficient because it avoids intermediate memory allocations.

$$ C_{ik} = \sum_j A_{ij} B_{jk} $$

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# Matrix Multiplication
C = torch.einsum('ij,jk->ik', A, B)

# Outer Product
v1 = torch.randn(5)
v2 = torch.randn(4)
outer = torch.einsum('i,j->ij', v1, v2) # (5, 4)
```
