# Day 8: Linear Algebra - The Engine of ML

> **Phase**: 1 - Foundations
> **Week**: 2 - Mathematical Foundations
> **Focus**: Vectors, Matrices, and Dimensionality
> **Reading Time**: 60 mins

---

## 1. Vectors & Spaces

Linear Algebra is the study of linear transformations. In ML, data is represented as vectors, and models are often sequences of linear transformations (matrices).

### 1.1 The Dot Product
$a \cdot b = \sum a_i b_i = ||a|| ||b|| \cos \theta$
*   **Geometric Interpretation**: It measures how much one vector "goes in the direction" of another.
*   **ML Application**:
    *   **Similarity**: High dot product (normalized) means vectors are similar (Cosine Similarity).
    *   **Projection**: Projecting data onto a feature vector.
    *   **Orthogonality**: If dot product is 0, vectors are uncorrelated/independent.

### 1.2 Basis Vectors
A basis is a set of vectors that can represent any point in a space through linear combination.
*   **Standard Basis**: x-axis (1,0), y-axis (0,1).
*   **Change of Basis**: PCA (Principal Component Analysis) essentially finds a *better* basis for the data—one where the axes align with the directions of maximum variance.

---

## 2. Matrix Decompositions

Decomposing a matrix helps us understand its intrinsic properties.

### 2.1 Eigenvalues and Eigenvectors
$Av = \lambda v$
*   **Concept**: An eigenvector $v$ is a vector that, when transformed by matrix $A$, does not change direction; it only scales by factor $\lambda$ (eigenvalue).
*   **ML Application**: In PCA, the eigenvectors of the Covariance Matrix are the "Principal Components" (directions of variance), and eigenvalues tell us how much variance exists in that direction.

### 2.2 Singular Value Decomposition (SVD)
Any matrix $A$ can be decomposed into $U \Sigma V^T$.
*   **Interpretation**: Rotation $\rightarrow$ Scaling $\rightarrow$ Rotation.
*   **Application**:
    *   **Dimensionality Reduction**: Truncated SVD (keeping top $k$ singular values) is the best low-rank approximation of a matrix.
    *   **Recommender Systems**: Decomposing the User-Item interaction matrix into User Factors and Item Factors.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The Curse of Dimensionality
**Scenario**: As you add more features (dimensions), the volume of the space increases exponentially. Data becomes sparse. Distance metrics (Euclidean) lose meaning because all points become roughly equidistant.
**Solution**:
1.  **Dimensionality Reduction**: PCA, t-SNE, Autoencoders.
2.  **Regularization**: L1 (Lasso) forces coefficients to zero, effectively selecting features.

### Challenge 2: Sparse Matrices
**Scenario**: In NLP (Bag of Words) or Recommenders, you have a matrix of 1 Million Users x 100k Items. 99.99% of entries are zero. Storing this as a standard dense matrix would require Terabytes of RAM.
**Solution**:
*   **Sparse Formats**: CSR (Compressed Sparse Row) or CSC. Store only the non-zero values and their indices.
*   **Libraries**: Scipy.sparse, PyTorch sparse tensors.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the geometric interpretation of the determinant?**
> **Answer**: The determinant represents the scaling factor of the "volume" transformed by the matrix.
> *   In 2D, it's the area scaling.
> *   If $\det(A) = 0$, the transformation collapses the space into a lower dimension (e.g., a plane into a line). This means the matrix is **singular** (not invertible) and destroys information.

**Q2: Why is SVD preferred over Eigendecomposition for non-square matrices?**
> **Answer**: Eigendecomposition only exists for square matrices. SVD exists for **any** matrix (rectangular). SVD generalizes the concept of eigenvalues (singular values) to all matrices, making it a universal tool for tasks like Latent Semantic Analysis (LSA) on term-document matrices.

**Q3: Explain the difference between L1 and L2 norm geometrically.**
> **Answer**:
> *   **L2 Norm (Euclidean)**: Distance "as the crow flies." The unit ball is a circle/sphere. In regularization (Ridge), it shrinks weights towards zero but rarely *to* zero.
> *   **L1 Norm (Manhattan)**: Sum of absolute differences. The unit ball is a diamond. The corners of the diamond touch the axes, which promotes **sparsity** (weights becoming exactly zero). This makes L1 useful for feature selection.

---

## 5. Further Reading
- [Essence of Linear Algebra (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/)
