# Lab 3: Tensor Operations & Broadcasting

## Objective
Master PyTorch tensors.
Understanding **Broadcasting** is critical for writing efficient code (e.g., Attention mechanisms).
Avoid `for` loops at all costs.

## 1. Broadcasting Rules
Two tensors are "broadcastable" if:
1.  Each tensor has at least one dimension.
2.  When iterating from the last dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

## 2. The Lab (`tensors.py`)

```python
import torch
import time

# 1. Basic Broadcasting
a = torch.tensor([[1, 2, 3], [4, 5, 6]]) # (2, 3)
b = torch.tensor([10, 20, 30])           # (3,)

print(f"a + b:\n{a + b}")
# b is broadcast to [[10, 20, 30], [10, 20, 30]]

# 2. Outer Product (The "Magic" of Broadcasting)
x = torch.arange(5)  # (5,)
y = torch.arange(4)  # (4,)
# Goal: Create a (5, 4) matrix where M[i, j] = x[i] * y[j]

# Reshape to (5, 1) and (1, 4)
x_col = x.view(-1, 1)
y_row = y.view(1, -1)

M = x_col * y_row
print(f"Outer Product:\n{M}")

# 3. Vectorization Speedup
size = 10000
A = torch.randn(size, size)
B = torch.randn(size, size)

# Naive Loop (Don't run this on large sizes, it's slow)
def naive_add(A, B):
    C = torch.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] + B[i, j]
    return C

# Vectorized
start = time.time()
C = A + B
end = time.time()
print(f"Vectorized Time: {end - start:.5f}s")
```

## 3. Challenge: Pairwise Distance
Given a batch of points `X` (N, D) and `Y` (M, D), calculate the pairwise Euclidean distance matrix `D` (N, M) **without loops**.
Hint: $(a-b)^2 = a^2 + b^2 - 2ab$

## 4. Submission
Submit the code for the Pairwise Distance challenge.
