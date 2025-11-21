# Day 8: Linear Algebra - Interview Questions

> **Topic**: Matrix Math
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is a Scalar, Vector, Matrix, and Tensor?
**Answer:**
*   **Scalar**: A single number (0D tensor).
*   **Vector**: An array of numbers (1D tensor). Magnitude and direction.
*   **Matrix**: A 2D grid of numbers (2D tensor). Represents a linear transformation.
*   **Tensor**: N-dimensional array (Generalization).

### 2. Explain Matrix Multiplication. What are the dimension requirements?
**Answer:**
*   To multiply $A (m \times n)$ and $B (p \times q)$, we must have $n = p$.
*   The result is $(m \times q)$.
*   $C_{ij}$ is the dot product of Row $i$ of A and Column $j$ of B.

### 3. What is the Transpose of a matrix?
**Answer:**
*   Flipping a matrix over its main diagonal.
*   $(A^T)_{ij} = A_{ji}$.
*   Properties: $(AB)^T = B^T A^T$.

### 4. What is the Identity Matrix?
**Answer:**
*   Square matrix with 1s on the diagonal and 0s elsewhere.
*   $AI = IA = A$.
*   Acts like the number "1" in scalar multiplication.

### 5. What is the Inverse of a matrix? When does it exist?
**Answer:**
*   $A^{-1}$ is a matrix such that $AA^{-1} = I$.
*   Exists only if A is **Square** and **Non-Singular** (Determinant $\neq 0$).

### 6. What is the Determinant of a matrix? What does it represent geometrically?
**Answer:**
*   A scalar value associated with a square matrix.
*   **Geometric Meaning**: The scaling factor of the volume (or area) of the linear transformation.
*   If Det = 0, the transformation collapses space (e.g., 2D plane to a line), meaning no inverse exists.

### 7. What is the Trace of a matrix?
**Answer:**
*   Sum of diagonal elements.
*   Invariant under change of basis. Sum of eigenvalues = Trace.

### 8. Explain Eigenvalues and Eigenvectors.
**Answer:**
*   $Av = \lambda v$.
*   **Eigenvector ($v$)**: A vector whose direction doesn't change after transformation A.
*   **Eigenvalue ($\lambda$)**: The amount by which the vector is stretched or shrunk.

### 9. What is Matrix Decomposition? Name a few types.
**Answer:**
*   Breaking a matrix into product of simpler matrices.
*   **LU**: Lower/Upper triangular. Used for solving linear systems.
*   **QR**: Orthogonal/Upper triangular.
*   **Eigendecomposition**: $PDP^{-1}$.
*   **SVD**: Singular Value Decomposition.

### 10. Explain Singular Value Decomposition (SVD).
**Answer:**
*   $A = U \Sigma V^T$.
*   Any matrix (even non-square) can be decomposed into:
    *   $U$: Orthogonal (Rotation).
    *   $\Sigma$: Diagonal (Scaling).
    *   $V^T$: Orthogonal (Rotation).
*   Used in PCA, Compression, Denoising.

### 11. What is the Rank of a matrix?
**Answer:**
*   The number of linearly independent rows (or columns).
*   Dimension of the image space.
*   Full Rank = Invertible (if square).

### 12. What is a Symmetric Matrix?
**Answer:**
*   $A = A^T$.
*   Important property: All eigenvalues are **Real**, and eigenvectors are **Orthogonal**.
*   Covariance matrices are always symmetric.

### 13. What is an Orthogonal Matrix?
**Answer:**
*   $Q^T Q = I$ (or $Q^T = Q^{-1}$).
*   Columns are orthonormal (unit length, perpendicular).
*   Preserves lengths and angles (Rotation/Reflection).

### 14. What is the Dot Product? What does it represent geometrically?
**Answer:**
*   $a \cdot b = |a| |b| \cos(\theta)$.
*   Represents projection of one vector onto another.
*   If 0, vectors are perpendicular (Orthogonal).

### 15. What is the Cross Product?
**Answer:**
*   Vector operation in 3D. Result is a vector perpendicular to both inputs.
*   Magnitude = Area of parallelogram spanned by vectors.

### 16. Explain the concept of Linear Independence.
**Answer:**
*   A set of vectors is independent if no vector can be written as a linear combination of others.
*   $c_1 v_1 + ... + c_n v_n = 0$ implies all $c_i = 0$.

### 17. What is a Basis Vector?
**Answer:**
*   A set of linearly independent vectors that span the entire space.
*   Any vector in the space can be uniquely represented as a combination of basis vectors.
*   Standard Basis for $R^2$: $(1,0), (0,1)$.

### 18. What is the Norm of a vector (L1, L2)?
**Answer:**
*   Measure of length.
*   **L2 (Euclidean)**: $\sqrt{\sum x_i^2}$. Shortest distance.
*   **L1 (Manhattan)**: $\sum |x_i|$. Grid distance. Promotes sparsity.

### 19. What is a Positive Definite Matrix?
**Answer:**
*   Symmetric matrix where $x^T A x > 0$ for all non-zero vectors $x$.
*   All eigenvalues are positive.
*   Analogy: Like a positive number. Ensures convex optimization landscape.

### 20. How is Linear Algebra used in PCA (Principal Component Analysis)?
**Answer:**
*   PCA finds the eigenvectors of the Covariance Matrix ($X^T X$).
*   These eigenvectors (Principal Components) point in directions of maximum variance.
*   We project data onto the top K eigenvectors to reduce dimensions.
