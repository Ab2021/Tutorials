# Day 15: Dimensionality Reduction

> **Phase**: 2 - Core Algorithms
> **Week**: 4 - Unsupervised & Deep Learning
> **Focus**: Compressing Reality
> **Reading Time**: 50 mins

---

## 1. Linear Reduction: PCA

Principal Component Analysis (PCA) is the workhorse of reduction.

### 1.1 The Intuition
PCA rotates the dataset to align with its "principal axes"—the directions of maximum variance. It then drops the axes with the least variance (noise).
*   **Eigenvectors**: The directions of the new axes.
*   **Eigenvalues**: The amount of variance explained by each axis.

### 1.2 When to use PCA
*   **Preprocessing**: Remove correlation between features (Whitening).
*   **Visualization**: Project 100D to 2D.
*   **Noise Reduction**: Reconstructing data from top $k$ components filters out noise.

---

## 2. Non-Linear Reduction: Manifold Learning

Real-world data (like images of faces) lies on a low-dimensional "manifold" that is curved and twisted inside the high-dimensional space. PCA (linear) cannot unfold a Swiss Roll.

### 2.1 t-SNE (t-Distributed Stochastic Neighbor Embedding)
*   **Goal**: Preserve **local structure**. Points close in high-D should be close in low-D.
*   **Mechanism**: Converts distances to probabilities. Minimizes KL Divergence between high-D and low-D distributions.
*   **Cons**: Slow ($O(N^2)$). Does not preserve global structure (cluster distances might be meaningless).

### 2.2 UMAP (Uniform Manifold Approximation and Projection)
*   **The Modern Standard**: Faster than t-SNE. Preserves more global structure. Mathematically grounded in Riemannian geometry and algebraic topology.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The Interpretation Gap
**Scenario**: You run PCA and get "Component 1". Business asks: "What does Component 1 mean?"
**Problem**: Component 1 is a linear combination of 50 features ($0.1 \times Age - 0.5 \times Income + \dots$). It has no physical meaning.
**Solution**: Look at the **Loadings** (weights). If Component 1 has high weights for "Income", "Spending", "Debt", you can label it "Financial Status".

### Challenge 2: Out of Sample Extension
**Scenario**: You run t-SNE on training data. A new user arrives. You want to plot them.
**Problem**: t-SNE is non-parametric. It doesn't learn a function $f(x)$. You have to re-run t-SNE on the whole dataset + new point.
**Solution**: Use **Parametric UMAP** or **Autoencoders** (Neural Nets), which learn a mapping function $f(x) \rightarrow z$.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why do we standardize data before PCA?**
> **Answer**: PCA maximizes variance. If one feature is "Salary" (range 0-100k) and another is "Age" (0-100), Salary has 1000x more variance simply due to units. PCA will focus entirely on Salary and ignore Age. Standardization makes variances comparable.

**Q2: Explain the difference between PCA and Autoencoders.**
> **Answer**:
> *   **PCA**: Linear transformation. Fast, deterministic, exact math. Limited to linear manifolds.
> *   **Autoencoder**: Neural Network. Non-linear (with activations). Can learn complex curved manifolds. Harder to train, non-deterministic. A linear Autoencoder converges to the PCA subspace.

**Q3: How do you choose the number of components ($k$) in PCA?**
> **Answer**: Plot the **Scree Plot** (Explained Variance Ratio vs. Component Index). Look for the "Elbow" where adding more components yields diminishing returns. Or pick $k$ to explain 95% of total variance.

---

## 5. Further Reading
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [UMAP: Uniform Manifold Approximation](https://umap-learn.readthedocs.io/)
