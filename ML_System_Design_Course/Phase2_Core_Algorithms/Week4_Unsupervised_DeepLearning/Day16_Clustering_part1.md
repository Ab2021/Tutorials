# Day 16 (Part 1): Advanced Clustering

> **Phase**: 6 - Deep Dive
> **Topic**: Unsupervised Structures
> **Focus**: Spectral Clustering, EM Algorithm, and Stability
> **Reading Time**: 60 mins

---

## 1. Spectral Clustering

K-Means assumes spherical blobs. Spectral Clustering finds connected shapes (spirals, moons).

### 1.1 The Laplacian
1.  Build Similarity Graph (Adjacency Matrix $W$).
2.  Compute Laplacian $L = D - W$ (Degree - Adjacency).
3.  Compute Eigenvectors of $L$.
4.  Run K-Means on the eigenvectors.
*   **Intuition**: Eigenvectors of Laplacian represent the "vibration modes" of the graph, separating disconnected components.

---

## 2. Gaussian Mixture Models (GMM) & EM

### 2.1 Expectation-Maximization (EM)
*   **E-Step**: Estimate probability (responsibility) of each point belonging to each cluster. (Soft assignment).
*   **M-Step**: Update cluster parameters ($\mu, \Sigma, \pi$) using weighted data.
*   **Convergence**: Guaranteed to converge to local optimum.

### 2.2 Covariance Types
*   **Spherical**: Like K-Means.
*   **Diagonal**: Ellipses aligned with axes.
*   **Full**: Arbitrary ellipses. (Most flexible, prone to overfitting).

---

## 3. Tricky Interview Questions

### Q1: Is K-Means guaranteed to find the global optimum?
> **Answer**: No. It converges to a local optimum. Depends on initialization.
> *   **Fix**: K-Means++ (Seeding far apart) or running multiple times.

### Q2: What is the relationship between K-Means and GMM?
> **Answer**: K-Means is a special case of GMM where:
> 1.  Covariance is spherical and fixed ($\Sigma = I$).
> 2.  Hard assignment (Probability is 0 or 1) instead of Soft.

### Q3: How to select K? (Beyond Elbow Method)
> **Answer**:
> *   **Silhouette Score**: Separation vs Cohesion.
> *   **Gap Statistic**: Compare dispersion to a null reference distribution.
> *   **Stability**: Subsample data, cluster, and check if clusters are consistent.

---

## 4. Practical Edge Case: High Dimensions
*   **Problem**: Distance becomes meaningless. K-Means fails.
*   **Fix**: Dimensionality Reduction (PCA/UMAP) *before* clustering. Or use Cosine Similarity (Spherical K-Means).

