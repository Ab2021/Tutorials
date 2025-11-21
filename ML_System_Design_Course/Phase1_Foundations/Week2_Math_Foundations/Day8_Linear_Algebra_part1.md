# Day 8 (Part 1): Advanced Linear Algebra & Tensors

> **Phase**: 6 - Deep Dive
> **Topic**: The Language of Data
> **Focus**: Tensor Decompositions, Matrix Calculus, and Spectral Theory
> **Reading Time**: 60 mins

---

## 1. Tensor Decompositions

SVD is for matrices (2D). What about Tensors (3D+)?

### 1.1 CP Decomposition (CANDECOMP/PARAFAC)
*   Approximates a tensor as a sum of rank-1 tensors.
*   $X \approx \sum a_r \circ b_r \circ c_r$.
*   **Use Case**: Separating sources, compression.

### 1.2 Tucker Decomposition
*   "Higher-order PCA".
*   Decomposes tensor into a "Core Tensor" (small) multiplied by factor matrices along each mode.
*   **Use Case**: Compressing neural network weights.

---

## 2. Matrix Calculus

Essential for deriving gradients.

### 2.1 The Jacobian
*   Vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$.
*   $J$ is an $m \times n$ matrix of partial derivatives.

### 2.2 Common Identities
*   $\nabla_x (a^T x) = a$
*   $\nabla_x (x^T A x) = (A + A^T)x$. (If symmetric, $2Ax$).
*   $\nabla_X \text{tr}(AX) = A^T$.

---

## 3. Spectral Theory

### 3.1 Positive Definite Matrices (PSD)
*   **Definition**: $x^T A x > 0$ for all non-zero $x$.
*   **Properties**: All eigenvalues $> 0$. Cholesky decomposition exists ($A = LL^T$).
*   **Importance**: Hessian matrix must be PSD for a local minimum. Covariance matrices are always PSD.

### 3.2 Condition Number
*   $\kappa(A) = |\lambda_{\max} / \lambda_{\min}|$.
*   **High $\kappa$**: Ill-conditioned. Matrix inversion is numerically unstable. Small noise in input -> Huge error in output.
*   **Regularization**: Adding $\lambda I$ (Ridge) improves condition number by boosting eigenvalues: $\frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$.

---

## 4. Tricky Interview Questions

### Q1: What is the Rank of a matrix?
> **Answer**: The number of linearly independent rows/columns.
> *   **Geometric**: The dimension of the space spanned by the columns.
> *   **Low Rank**: Implies redundancy. Recommender matrices are low rank (users behave similarly).

### Q2: Why is SVD numerical gold?
> **Answer**:
> 1.  **Existence**: Exists for *every* matrix (unlike Eigen decomposition).
> 2.  **Stability**: Singular values are stable w.r.t perturbations.
> 3.  **Approximation**: Truncated SVD gives the optimal low-rank approximation (Eckart-Young Theorem).

### Q3: How do you compute the inverse of a 1 Million x 1 Million matrix?
> **Answer**: **You don't.**
> *   Inverting is $O(N^3)$.
> *   You solve $Ax = b$ using iterative methods (Conjugate Gradient) or approximate it.
> *   Or use Woodbury Matrix Identity if $A$ is a low-rank update to a diagonal matrix.

---

## 5. Practical Edge Case: Sparse Matrices
*   **CSR (Compressed Sparse Row)**: Efficient for arithmetic.
*   **COO (Coordinate)**: Efficient for construction.
*   **Trap**: Converting sparse to dense (`.toarray()`) blows up RAM. Always keep sparse as long as possible.

