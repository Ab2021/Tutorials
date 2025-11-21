# Day 2 (Part 1): Advanced NumPy & Tensor Mechanics

> **Phase**: 6 - Deep Dive
> **Topic**: Advanced Numerical Computing
> **Focus**: Einsum, Strides, and Low-Level Optimization
> **Reading Time**: 60 mins

---

## 1. Einstein Summation (`einsum`)

The Swiss Army Knife of tensor operations. It can replace `dot`, `matmul`, `transpose`, `sum`, `trace`.

### 1.1 The Syntax
`np.einsum('ik,kj->ij', A, B)`
*   **Indices**: `i, k` are dimensions of A. `k, j` are dimensions of B.
*   **Rule**: Repeated indices (`k`) are summed over. Unrepeated indices (`i, j`) appear in output.

### 1.2 Examples
*   **Matrix Mul**: `'ik,kj->ij'`
*   **Dot Product**: `'i,i->'` (Sum over i)
*   **Outer Product**: `'i,j->ij'`
*   **Transpose**: `'ij->ji'`
*   **Batch Matrix Mul**: `'bik,bkj->bij'` (Batch dimension `b` is preserved).

### 1.3 Why use it?
*   **Readability**: Explicitly shows which dimensions are being summed.
*   **Performance**: Often faster because it optimizes the loop order and avoids intermediate memory allocations.

---

## 2. Stride Tricks: `as_strided`

**Warning**: This can crash your Python interpreter (Segfault) if used wrong.

### 2.1 The Concept
You can create a sliding window view of an array *without copying data*.
*   **Scenario**: Create a dataset of sliding windows of size 3 from `[1, 2, 3, 4, 5]`.
    *   Output: `[[1, 2, 3], [2, 3, 4], [3, 4, 5]]`.
*   **Naive**: Loop and copy.
*   **Strided**:
    ```python
    from numpy.lib.stride_tricks import as_strided
    x = np.arange(5)
    byte_step = x.strides[0] # 8 bytes
    # Shape: (3 windows, 3 elements per window)
    # Strides: (Jump 1 element to next window, Jump 1 element to next item in window)
    view = as_strided(x, shape=(3, 3), strides=(byte_step, byte_step))
    ```
*   **Result**: Zero memory overhead.

---

## 3. Advanced Indexing

### 3.1 Integer vs. Boolean Indexing
*   **Slice**: `arr[1:5]` -> **View**.
*   **Boolean Mask**: `arr[arr > 0]` -> **Copy**. (Size is unknown beforehand).
*   **Integer Array (Fancy Indexing)**: `arr[[1, 5, 2]]` -> **Copy**.

### 3.2 `ix_` (Open Mesh)
*   **Scenario**: You want to select rows `[1, 3]` and columns `[2, 4]`.
*   **Wrong**: `arr[[1, 3], [2, 4]]` gives points `(1,2)` and `(3,4)`.
*   **Right**: `np.ix_([1, 3], [2, 4])` creates a meshgrid to select the submatrix.

---

## 4. Under the Hood: BLAS & LAPACK

NumPy doesn't do the math. It calls libraries.
*   **BLAS (Basic Linear Algebra Subprograms)**: Low-level vector/matrix ops.
    *   *Level 1*: Vector-Vector.
    *   *Level 2*: Matrix-Vector.
    *   *Level 3*: Matrix-Matrix (GEMM).
*   **LAPACK**: Higher level (SVD, Eigenvalues).
*   **Implementations**:
    *   **OpenBLAS**: Open source.
    *   **MKL (Intel)**: Highly optimized for Intel CPUs.
    *   **Accelerate (Apple)**: For M1/M2 chips.
*   **Debug**: `np.show_config()` tells you what you are running.

---

## 5. Tricky Interview Questions

### Q1: Implement 2D Convolution using `as_strided`.
> **Answer**:
> 1.  Input Image: $H \times W$. Kernel: $K \times K$.
> 2.  Use `as_strided` to create a 4D view of the image: $(H', W', K, K)$.
>     *   This creates "windows" of size $K \times K$ for every pixel.
> 3.  Reshape to $(N, K^2)$.
> 4.  Flatten Kernel to $(K^2, 1)$.
> 5.  Perform Matrix Multiplication.
> *   This is how `im2col` works in Caffe/PyTorch (conceptually).

### Q2: What happens if you assign to a Boolean Masked array?
> **Answer**: `arr[arr < 0] = 0`.
> *   NumPy calls `__setitem__`.
> *   It does *not* create a copy. It modifies the original array in-place using the mask.
> *   However, `b = arr[arr < 0]` *does* create a copy.

### Q3: Why is `np.sum` faster than Python `sum`?
> **Answer**:
> 1.  **Type Checking**: Python `sum` checks type of every element. NumPy knows they are all floats.
> 2.  **C Loop**: NumPy iterates in C.
> 3.  **SIMD**: NumPy uses vector instructions to add 4/8 numbers at once.
> 4.  **Accumulator**: NumPy uses a higher precision accumulator (float64) even for float32 arrays to prevent overflow/precision loss.

