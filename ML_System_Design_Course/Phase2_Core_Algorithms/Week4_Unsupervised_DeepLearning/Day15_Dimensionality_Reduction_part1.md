# Day 15 (Part 1): Advanced Dimensionality Reduction

> **Phase**: 6 - Deep Dive
> **Topic**: Manifold Learning
> **Focus**: Kernel PCA, t-SNE Internals, and VAEs
> **Reading Time**: 60 mins

---

## 1. Kernel PCA

PCA is linear. Kernel PCA is non-linear.
*   **Idea**: Map data to high-dim space $\phi(x)$, do PCA there.
*   **Trick**: Compute Kernel Matrix $K$. Center it. Eigendecompose.
*   **Use Case**: Separating concentric circles.

---

## 2. t-SNE Internals

### 2.1 The Crowding Problem
*   In high dimensions, there is "more volume" at a distance.
*   Mapping to 2D crushes this volume. Points get crowded.
*   **Fix**: t-Distribution (Heavy tails) in low dimension. Allows distant points to be further apart in 2D than in High-D.

### 2.2 Barnes-Hut Optimization
*   Naive t-SNE is $O(N^2)$.
*   **Barnes-Hut**: Approximates forces using a Quadtree. $O(N \log N)$.
*   **Perplexity**: The knob. Effective number of neighbors.

---

## 3. UMAP (Uniform Manifold Approximation and Projection)

*   **Topology**: Assumes data is uniformly distributed on a Riemannian manifold.
*   **Speed**: Much faster than t-SNE. Preserves Global Structure better.

---

## 4. Tricky Interview Questions

### Q1: Why is PCA variance maximization equivalent to reconstruction error minimization?
> **Answer**: Pythagorean theorem. Total Variance = Explained Variance + Unexplained Variance (Error). Maximizing one minimizes the other.

### Q2: Can you inverse t-SNE?
> **Answer**: No. It learns a non-parametric mapping. There is no function $f(x)$ to map new points. You must rerun optimization (or use parametric t-SNE).

### Q3: Variational Autoencoder (VAE) vs. Autoencoder (AE)?
> **Answer**:
> *   **AE**: Maps input to a single point in latent space. Good for compression. Bad for generation (latent space has holes).
> *   **VAE**: Maps input to a *distribution* (Mean, Var). Regularizes latent space to be Gaussian. Good for generation.

---

## 5. Practical Edge Case: PCA on Time Series
*   **Problem**: PCA ignores time order.
*   **Fix**: **SSA (Singular Spectrum Analysis)** or Dynamic Mode Decomposition (DMD). Embed time series into a Hankel matrix (lagged copies) before SVD.

