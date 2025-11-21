# Day 2: NumPy & Vectorization - Interview Questions

> **Topic**: High-Performance Computing
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Why is NumPy faster than Python lists? Explain memory layout and SIMD.
**Answer:**
*   **Contiguous Memory**: NumPy arrays are stored in a continuous block of memory. Python lists are arrays of pointers to objects scattered in memory. Contiguous memory improves CPU cache locality.
*   **Fixed Type**: NumPy elements are all the same type (e.g., `int64`), avoiding type checking overhead during iteration.
*   **SIMD**: NumPy uses Vectorized CPU instructions (Single Instruction, Multiple Data) to perform operations on multiple numbers simultaneously.

### 2. What is Broadcasting in NumPy? What are the rules for two arrays to be broadcastable?
**Answer:**
*   **Broadcasting**: Mechanism to perform arithmetic on arrays of different shapes.
*   **Rules**: Align shapes from the right (trailing dimensions). Two dimensions are compatible if:
    1.  They are equal.
    2.  One of them is 1.
*   **Example**: `(3, 1)` and `(3, 5)` -> Compatible. `(3, 2)` and `(3, 5)` -> Error.

### 3. Explain the difference between a View and a Copy in NumPy. Give an example of an operation that creates a view.
**Answer:**
*   **View**: Looks at the *same* memory buffer. Changing the view changes the original. Fast (no data copying).
    *   Example: Slicing `arr[0:5]`.
*   **Copy**: Allocates new memory. Changing copy doesn't affect original. Slower.
    *   Example: Fancy indexing `arr[[0, 1, 2]]` or `arr.copy()`.

### 4. How would you compute the pairwise Euclidean distance between two matrices A (MxD) and B (NxD) without loops?
**Answer:**
*   Use the expansion: $(a-b)^2 = a^2 + b^2 - 2ab$.
*   `A_sq = np.sum(A**2, axis=1, keepdims=True)` (Shape Mx1)
*   `B_sq = np.sum(B**2, axis=1)` (Shape N)
*   `Dist = np.sqrt(A_sq + B_sq - 2 * np.dot(A, B.T))`
*   This uses matrix multiplication which is highly optimized (BLAS).

### 5. What is the difference between `np.dot`, `np.matmul`, and `*` (element-wise multiplication)?
**Answer:**
*   `*`: Element-wise multiplication (Hadamard product).
*   `np.dot`: Dot product for 1D, Matrix mult for 2D. For N-D, it's a sum product over the last axis of a and second-to-last of b.
*   `np.matmul` (`@`): Matrix multiplication. Treats N-D arrays as a stack of matrices. It broadcasts. It is generally preferred for batch matrix multiplication.

### 6. How does `np.einsum` work? Write an einsum string for Matrix Transpose.
**Answer:**
*   **Einstein Summation**: A mini-language to specify dimension manipulation.
*   **Transpose**: `np.einsum('ij->ji', A)`.
*   **Matrix Mult**: `np.einsum('ik,kj->ij', A, B)`.
*   It's often faster because it optimizes the loop order and avoids intermediate arrays.

### 7. What are "Strides" in a NumPy array? How do they allow for O(1) Transpose?
**Answer:**
*   **Strides**: The number of bytes to step to reach the next element in each dimension.
*   **Transpose**: To transpose, NumPy simply **swaps the strides** and shape in the metadata. It does *not* move any data in memory. Hence, it is $O(1)$.

### 8. Explain the difference between Integer Indexing (Fancy Indexing) and Boolean Masking. Which one creates a copy?
**Answer:**
*   **Integer Indexing**: `arr[[0, 2]]`. Selects specific indices. Creates a **Copy**.
*   **Boolean Masking**: `arr[arr > 0]`. Selects elements where condition is True. Creates a **Copy**.
*   (Note: Basic slicing `arr[0:2]` creates a View).

### 9. How do you handle `NaN` values in NumPy? What is the difference between `np.mean` and `np.nanmean`?
**Answer:**
*   `NaN` (Not a Number) propagates. `1 + NaN = NaN`.
*   `np.mean`: If array has any NaN, result is NaN.
*   `np.nanmean`: Ignores NaNs and computes mean of valid numbers.

### 10. What is the purpose of `np.newaxis`?
**Answer:**
*   It increases the dimension of the array by 1.
*   Used for broadcasting.
*   Example: Convert 1D array `(N,)` to column vector `(N, 1)` using `arr[:, np.newaxis]`.

### 11. Explain how `np.vectorize` works. Is it actually faster than a loop?
**Answer:**
*   `np.vectorize` is a convenience function that takes a scalar function and applies it to arrays.
*   **Performance**: It is essentially a **for-loop** implemented in C (mostly). It is **not** true SIMD vectorization. It is usually slower than writing the function using native NumPy operations.

### 12. What is the difference between `C-contiguous` (Row-major) and `F-contiguous` (Column-major) arrays? Why does it matter for performance?
**Answer:**
*   **C (C-style)**: Rows are stored together. Iterating over rows is fast.
*   **F (Fortran-style)**: Columns are stored together. Iterating over columns is fast.
*   **Performance**: Accessing memory sequentially is fast (cache hits). Jumping around (strided access) is slow (cache misses).

### 13. How would you implement One-Hot Encoding using `np.eye` and fancy indexing?
**Answer:**
```python
labels = np.array([0, 2, 1])
num_classes = 3
one_hot = np.eye(num_classes)[labels]
# Result: [[1,0,0], [0,0,1], [0,1,0]]
```

### 14. What is `np.memmap`? When would you use it?
**Answer:**
*   **Memory Map**: Maps a file on disk directly into virtual memory.
*   **Usage**: Allows accessing small segments of a huge array (larger than RAM) without reading the whole file. The OS handles paging data in/out of RAM.

### 15. How do you find the index of the N-th largest value in an array efficiently?
**Answer:**
*   `np.argpartition(arr, -N)[-N]`.
*   This uses the Introselect algorithm ($O(n)$) instead of full sort ($O(n \log n)$).

### 16. Explain the concept of "Universal Functions" (ufuncs).
**Answer:**
*   Ufuncs are functions that operate element-by-element on ndarrays.
*   They support broadcasting, type casting, and output arguments.
*   Examples: `np.add`, `np.sin`, `np.exp`.

### 17. What is the difference between `np.concatenate`, `np.stack`, and `np.hstack`?
**Answer:**
*   `concatenate`: Joins arrays along an *existing* axis.
*   `stack`: Joins arrays along a *new* axis.
*   `hstack`: Horizontal stack (equivalent to concatenate on axis 1).

### 18. How does NumPy handle overflow for different data types (e.g., `int8` vs `float32`)?
**Answer:**
*   **Integers**: Fixed precision. They **wrap around** (modular arithmetic). `np.int8(127) + 1` becomes `-128`. No warning by default.
*   **Floats**: Overflow to `inf`. `np.float32(1e38) * 10` becomes `inf`.

### 19. Write a NumPy snippet to normalize a matrix (subtract mean, divide by std) column-wise.
**Answer:**
```python
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / (std + 1e-8) # Add epsilon for stability
```

### 20. What is the difference between `np.random.rand` and `np.random.randn`?
**Answer:**
*   `rand`: Uniform distribution $[0, 1)$.
*   `randn`: Standard Normal distribution (Mean 0, Std 1).
