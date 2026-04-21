# 🔢 Easy: Matrix Operations from Scratch
> **Platforms:** Deep-ML + TensorTonic | **Difficulty:** Easy | **Must Know Cold ❄️**

---

## WHY MATRIX OPS MATTER IN ML INTERVIEWS

Every ML algorithm — from linear regression to transformers — is fundamentally matrix operations. Interviewers test these to confirm you understand the mathematical substrate of ML, not just library calls.

**Key rule:** Implement WITHOUT numpy.matmul or numpy.dot for matrix multiply — show you understand the triple-loop.

---

## 🟢 PROBLEM 1: Matrix Transpose

### Theory
For matrix A of shape (m, n), transpose A^T has shape (n, m) where:
$$A^T[i][j] = A[j][i]$$

**Properties:**
- $(A^T)^T = A$
- $(AB)^T = B^T A^T$ (order reverses)
- For symmetric matrices: $A = A^T$

### Things to Focus On
- ✅ Shape changes from (m, n) → (n, m)
- ✅ In-place transpose is tricky; usually create new matrix
- ✅ Numpy: `A.T` or `np.transpose(A)` — know both

### Implementation
```python
import numpy as np
from typing import List

def transpose_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """
    Transpose a 2D matrix: A[i][j] → A[j][i]
    
    Time: O(m*n)
    Space: O(m*n) for the new matrix
    """
    if not matrix or not matrix[0]:
        return []
    
    m, n = len(matrix), len(matrix[0])
    # Transposed matrix has shape (n, m)
    result = [[0.0] * m for _ in range(n)]
    
    for i in range(m):
        for j in range(n):
            result[j][i] = matrix[i][j]
    
    return result

# NumPy version (for comparison):
def transpose_numpy(A: np.ndarray) -> np.ndarray:
    return A.T  # or np.transpose(A)

# Test
A = [[1, 2, 3],
     [4, 5, 6]]   # Shape: 2×3
print(transpose_matrix(A))
# [[1, 4],
#  [2, 5],
#  [3, 6]]  ← Shape: 3×2 ✓
```

---

## 🟢 PROBLEM 2: Matrix Multiplication

### Theory
For A of shape (m, k) and B of shape (k, n):
$$C[i][j] = \sum_{t=0}^{k-1} A[i][t] \cdot B[t][j]$$

Result C has shape (m, n). The inner dimensions must match.

**Memory for matrix sizes:**
```
A: (m×k) × B: (k×n) = C: (m×n)
    ↑         ↑
  must match (inner dims)
```

**Complexity:** O(m × k × n)

### Things to Focus On
- ✅ Inner dimensions must match: A.cols == B.rows
- ✅ Result shape: (A.rows, B.cols)
- ✅ Matrix multiply is NOT element-wise (that's Hadamard product)
- ✅ Not commutative: AB ≠ BA in general
- ✅ Associative: (AB)C = A(BC)

### Implementation
```python
def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Matrix multiplication: C[i][j] = sum(A[i][t] * B[t][j] for t)
    
    A: shape (m, k)
    B: shape (k, n)
    C: shape (m, n)
    
    Time: O(m*k*n)
    """
    m = len(A)
    k = len(A[0])  # = len(B)
    n = len(B[0])
    
    assert len(B) == k, f"Dimension mismatch: A has {k} cols, B has {len(B)} rows"
    
    # Initialize result matrix with zeros
    C = [[0.0] * n for _ in range(m)]
    
    for i in range(m):       # Iterate over rows of A
        for j in range(n):   # Iterate over cols of B
            for t in range(k):  # Inner dimension
                C[i][j] += A[i][t] * B[t][j]
    
    return C

def dot_product(a: List[float], b: List[float]) -> float:
    """1D dot product: sum(a[i] * b[i])"""
    assert len(a) == len(b), "Vectors must have same length"
    return sum(a[i] * b[i] for i in range(len(a)))

# Test
A = [[1, 2], [3, 4]]   # 2×2
B = [[5, 6], [7, 8]]   # 2×2
C = matrix_multiply(A, B)
print(C)
# [[1*5+2*7, 1*6+2*8],   = [[19, 22],
#  [3*5+4*7, 3*6+4*8]]     [43, 50]]

# Verify with numpy
print(np.array(A) @ np.array(B))  # Same result ✓
```

---

## 🟢 PROBLEM 3: Matrix Trace

### Theory
The trace of a square matrix is the sum of its diagonal elements:
$$\text{tr}(A) = \sum_{i=0}^{n-1} A[i][i]$$

**Properties:**
- $\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(AB) = \text{tr}(BA)$ (cyclic property)
- $\text{tr}(A) = \sum_i \lambda_i$ (sum of eigenvalues)

### Things to Focus On
- ✅ Only defined for square matrices
- ✅ Cyclic property: tr(ABC) = tr(CAB) = tr(BCA) — useful in matrix calculus
- ✅ Used in: regularization (Frobenius norm²), matrix derivative proofs

### Implementation
```python
def matrix_trace(A: List[List[float]]) -> float:
    """
    Trace: sum of diagonal elements. Only for square matrices.
    """
    n = len(A)
    assert all(len(row) == n for row in A), "Trace requires a square matrix"
    
    return sum(A[i][i] for i in range(n))

# Test
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
print(matrix_trace(A))  # 1 + 5 + 9 = 15
print(np.trace(A))      # 15 ✓
```

---

## 🟢 PROBLEM 4: Matrix/Vector Normalization

### Theory
**L2 Normalization (unit vector):**
$$\hat{x} = \frac{x}{\|x\|_2} = \frac{x}{\sqrt{\sum_i x_i^2}}$$

**MinMax Normalization (to [0, 1]):**
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

**Frobenius Norm (for matrices):**
$$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

**Z-score Standardization:**
$$x' = \frac{x - \mu}{\sigma}$$

### Things to Focus On
- ✅ L2 norm → unit vector (used in cosine similarity, embedding normalization)
- ✅ MinMax: sensitive to outliers (one extreme value dominates)
- ✅ Z-score: robust to outliers, standard for neural net inputs
- ✅ Robust scaling: use IQR instead of std — robust to outliers
- ✅ Never apply normalization before train/test split (use fit on train, transform on test)

### Implementation
```python
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Normalize to unit length along specified axis."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)  # eps prevents division by zero

def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. Sensitive to outliers."""
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)

def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Standardize: zero mean, unit variance."""
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sigma

def robust_scale(x: np.ndarray) -> np.ndarray:
    """Scale using median and IQR — robust to outliers."""
    median = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    return (x - median) / (iqr + 1e-8)

def frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm: sqrt(sum of squared elements)"""
    return np.sqrt(np.sum(A ** 2))

# Test
x = np.array([3.0, 4.0])
print(l2_normalize(x))  # [0.6, 0.8] — unit vector

v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(minmax_normalize(v))  # [0.  0.25 0.5  0.75 1. ]
print(zscore_normalize(v))  # [-1.414, -0.707, 0., 0.707, 1.414]
```

---

## 🟢 PROBLEM 5: Cosine Similarity

### Theory
Measures the cosine of the angle between two vectors:
$$\text{cosine}(a, b) = \frac{a \cdot b}{\|a\|_2 \|b\|_2} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \cdot \sqrt{\sum_i b_i^2}}$$

**Range:** [-1, 1]
- 1: Identical direction
- 0: Orthogonal (no similarity)
- -1: Opposite direction

**Vs Euclidean distance:** Cosine ignores magnitude, only cares about direction. Euclidean cares about both.

### Things to Focus On
- ✅ Use cosine for text/embedding similarity (magnitude shouldn't matter)
- ✅ Use Euclidean when magnitude matters (e.g., fraud amount deviation)
- ✅ Numerically: eps in denominator to prevent division by zero
- ✅ Applied in RAG retrieval, recommendation systems, NLP

### Implementation
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity: dot(a,b) / (||a|| * ||b||)
    Range: [-1, 1]. Higher = more similar direction.
    """
    eps = 1e-8
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all rows of A and all rows of B.
    Returns matrix of shape (len(A), len(B)).
    Used in RAG retrieval: query embeddings vs document embeddings.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T

# Test
a = np.array([1.0, 0.0])  # Points right
b = np.array([0.0, 1.0])  # Points up
c = np.array([1.0, 1.0])  # Points diagonal (45°)
d = np.array([-1.0, 0.0]) # Points left

print(cosine_similarity(a, a))  # 1.0   (identical)
print(cosine_similarity(a, b))  # 0.0   (orthogonal)
print(cosine_similarity(a, c))  # 0.707 (45°)
print(cosine_similarity(a, d))  # -1.0  (opposite)
```

---

## 🟢 PROBLEM 6: Euclidean Distance

### Theory
$$d(a, b) = \|a - b\|_2 = \sqrt{\sum_i (a_i - b_i)^2}$$

**Variants:**
- Manhattan (L1): $\|a - b\|_1 = \sum_i |a_i - b_i|$
- Minkowski: $\|a - b\|_p = (\sum_i |a_i - b_i|^p)^{1/p}$

Used in KNN, K-Means centroid assignment.

### Things to Focus On
- ✅ Euclidean vs Manhattan: Euclidean penalizes large deviations more (squared)
- ✅ For KNN in high dimensions: curse of dimensionality makes distance meaningless
- ✅ Broadcasting trick for computing pairwise distances efficiently

### Implementation
```python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise Euclidean distances between rows of X and Y.
    Returns matrix of shape (len(X), len(Y)).
    
    Efficient using broadcasting:
    ||a-b||² = ||a||² + ||b||² - 2a·b^T
    """
    # Broadcasting approach (efficient)
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
    cross = X @ Y.T                                 # (n, m)
    dist_sq = X_sq + Y_sq.T - 2 * cross
    return np.sqrt(np.maximum(dist_sq, 0))          # clip for numerical stability

# Test
a = np.array([0.0, 0.0])
b = np.array([3.0, 4.0])
print(euclidean_distance(a, b))  # 5.0 (3-4-5 triangle) ✓
```

---

## 🟢 PROBLEM 7: Covariance Matrix

### Theory
The covariance matrix Σ captures how variables co-vary:
$$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

For data matrix X of shape (n_samples, n_features):
$$\Sigma = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$

**Properties:**
- Symmetric: $\Sigma = \Sigma^T$
- Positive semi-definite: $v^T \Sigma v \geq 0$ for all v
- Diagonal entries: variances of each feature
- Off-diagonal: covariances (positive → same direction, negative → opposite)

### Things to Focus On
- ✅ Use (n-1) denominator, not n (Bessel's correction for unbiased estimate)
- ✅ Covariance ≠ correlation (correlation is normalized covariance)
- ✅ Used in: PCA (eigen decomposition of covariance), Gaussian distribution, Mahalanobis distance
- ✅ Correlation matrix: Σ / (σ_i × σ_j)

### Implementation
```python
def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix of data matrix X.
    
    Args:
        X: shape (n_samples, n_features)
    Returns:
        Cov: shape (n_features, n_features)
    """
    n = X.shape[0]
    # Center the data (subtract column means)
    X_centered = X - X.mean(axis=0)
    # Cov = X_centered^T @ X_centered / (n-1)
    return (X_centered.T @ X_centered) / (n - 1)

def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Normalize covariance to get correlation matrix."""
    cov = covariance_matrix(X)
    std = np.sqrt(np.diag(cov))  # Standard deviations
    return cov / np.outer(std, std)

# Test
X = np.array([[1.0, 2.0],
              [2.0, 3.0],
              [3.0, 2.0],
              [4.0, 1.0]])
print(covariance_matrix(X))
# [[ 1.667, -0.333],
#  [-0.333,  0.667]]
# Variance of X[:,0] = 1.667, X[:,1] = 0.667
# Correlation = negative (as feature 0 goes up, feature 1 tends down)

# Verify with numpy
print(np.cov(X.T))  # Same result ✓
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"When would matrix multiply fail?"** → When inner dimensions don't match. A(m×k) × B(k×n) requires k columns in A = k rows in B.
2. **"What's the difference between cosine similarity and dot product?"** → Cosine normalizes by magnitude; dot product doesn't. Two vectors in the same direction but different magnitudes have cosine=1 but different dot products.
3. **"Why use (n-1) in covariance denominator?"** → Bessel's correction: n-1 gives an unbiased estimator of population covariance. n would underestimate it.
4. **"How does PCA relate to the covariance matrix?"** → PCA eigenvectors of the covariance matrix are the principal components. Eigenvalues represent variance explained by each component.
