# Day 15: Dimensionality Reduction - Interview Questions

> **Topic**: Unsupervised Feature Extraction
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Why is Dimensionality Reduction important? (Curse of Dimensionality).
**Answer:**
*   **Curse**: As dimensions increase, data becomes sparse. "Distance" loses meaning (all points are equidistant). Models overfit.
*   **Reduction**: Compresses data, removes noise, speeds up training, allows visualization.

### 2. Explain Principal Component Analysis (PCA).
**Answer:**
*   Linear transformation that projects data to a new coordinate system.
*   New axes (Principal Components) are directions of **Maximum Variance**.
*   Axes are orthogonal (uncorrelated).

### 3. What are Principal Components? Are they orthogonal?
**Answer:**
*   Eigenvectors of the Covariance Matrix.
*   **Yes**, they are orthogonal. PC1 is perpendicular to PC2.

### 4. How do you determine the number of components to keep in PCA? (Scree Plot).
**Answer:**
*   **Scree Plot**: Plot eigenvalues (variance explained) vs component index. Look for the "Elbow".
*   **Threshold**: Keep components explaining 95% of variance.

### 5. Does PCA require Feature Scaling? Why?
**Answer:**
*   **Yes**.
*   PCA maximizes variance. If one feature is in km (0-1000) and another in m (0-1), PCA will focus only on km.
*   Standardize to Mean 0, Var 1.

### 6. What is the reconstruction error in PCA?
**Answer:**
*   Difference between original data and data projected back from reduced space.
*   Minimizing reconstruction error $\iff$ Maximizing variance.

### 7. Explain t-SNE (t-Distributed Stochastic Neighbor Embedding).
**Answer:**
*   Non-linear technique for visualization (2D/3D).
*   Preserves **Local Structure** (Neighbors stay neighbors).
*   Uses Student's t-distribution to handle "Crowding problem".

### 8. What is the difference between PCA and t-SNE?
**Answer:**
*   **PCA**: Linear. Preserves Global structure (Variance). Deterministic. Good for compression.
*   **t-SNE**: Non-linear. Preserves Local structure (Clusters). Stochastic. Good for visualization.

### 9. What is LDA (Linear Discriminant Analysis)? How does it differ from PCA?
**Answer:**
*   **LDA**: Supervised. Finds axes that maximize separation between **Classes**.
*   **PCA**: Unsupervised. Finds axes that maximize **Variance**.
*   LDA is better for classification preprocessing.

### 10. Explain UMAP. Why is it often preferred over t-SNE?
**Answer:**
*   Uniform Manifold Approximation and Projection.
*   **Pros**: Faster than t-SNE. Preserves more **Global Structure** (distance between clusters means something). Deterministic (optional).

### 11. What is an Autoencoder? How is it used for dimensionality reduction?
**Answer:**
*   Neural Network: Input -> Encoder -> Bottleneck -> Decoder -> Output.
*   Trained to output Input ($x = \hat{x}$).
*   **Bottleneck**: Compressed representation (Non-linear PCA).

### 12. What is the difference between Linear and Non-linear dimensionality reduction?
**Answer:**
*   **Linear** (PCA, LDA): Projections on planes. Fails on "Swiss Roll" dataset.
*   **Non-linear** (t-SNE, UMAP, Kernel PCA): Unfolds manifolds.

### 13. What is Factor Analysis?
**Answer:**
*   Statistical method. Assumes observed variables are linear combinations of latent "Factors" + Noise.
*   Similar to PCA but focuses on covariance/correlation structure.

### 14. Explain the concept of "Explained Variance Ratio".
**Answer:**
*   Percentage of the dataset's total information (variance) captured by each PC.
*   PC1 might explain 40%, PC2 20%, etc.

### 15. Can PCA be used for Feature Selection?
**Answer:**
*   **No**, it does Feature **Extraction**.
*   New features are mixtures of all old features. You lose interpretability.
*   Unless you look at loading vectors to see which original features contribute most.

### 16. What is Kernel PCA?
**Answer:**
*   PCA + Kernel Trick.
*   Maps data to high dimensions, does PCA there.
*   Allows separating non-linear clusters (concentric circles).

### 17. What is the Crowding Problem in t-SNE?
**Answer:**
*   In high dimensions, there is more "room" for neighbors. In 2D, points get squashed together.
*   t-SNE uses heavy-tailed distribution (t-dist) in low dimension to allow distant points to be further apart.

### 18. When would you use PCA vs t-SNE?
**Answer:**
*   **PCA**: Preprocessing for ML models (Denoising, Speed).
*   **t-SNE**: Visualization / EDA only. (Can't apply to new data easily).

### 19. How does dimensionality reduction help with Overfitting?
**Answer:**
*   Removes noise.
*   Reduces number of parameters model needs to learn.
*   Increases data density.

### 20. What is Manifold Learning?
**Answer:**
*   Assumption that high-dimensional data lies on a lower-dimensional surface (Manifold) embedded in the space.
*   Goal: Unroll the manifold.
