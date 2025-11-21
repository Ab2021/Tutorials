# Day 16: Clustering - Interview Questions

> **Topic**: Unsupervised Grouping
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is Clustering? Give real-world examples.
**Answer:**
*   Grouping data points such that points in the same group are more similar to each other than to those in other groups.
*   **Examples**: Customer segmentation, Image compression (Color quantization), Anomaly detection, Document grouping.

### 2. Explain the K-Means algorithm.
**Answer:**
1.  Initialize K centroids randomly.
2.  Assign each point to the nearest centroid.
3.  Update centroids to be the mean of assigned points.
4.  Repeat until convergence (centroids don't move).

### 3. How do you choose K in K-Means? (Elbow Method, Silhouette Score).
**Answer:**
*   **Elbow Method**: Plot Inertia (Sum of squared distances) vs K. Look for the "elbow" where improvement slows down.
*   **Silhouette Score**: Measures how close points are to their own cluster vs neighbor cluster. Range [-1, 1]. Higher is better.

### 4. What is K-Means++ initialization?
**Answer:**
*   Smart initialization to avoid bad local minima.
1.  Pick 1st centroid randomly.
2.  Pick next centroid with probability proportional to distance squared from nearest existing centroid.
*   Spreads centroids out.

### 5. What are the limitations of K-Means?
**Answer:**
*   Assumes spherical clusters. Fails on concentric circles or irregular shapes.
*   Sensitive to outliers.
*   Must specify K.
*   Can get stuck in local minima.

### 6. Explain Hierarchical Clustering (Agglomerative vs Divisive).
**Answer:**
*   **Agglomerative (Bottom-up)**: Start with N clusters. Merge closest pair. Repeat until 1 cluster.
*   **Divisive (Top-down)**: Start with 1 cluster. Split recursively.

### 7. What is a Dendrogram?
**Answer:**
*   Tree diagram showing the hierarchy of merges.
*   Y-axis represents the distance at which clusters were merged.
*   Cut the tree at a specific height to get K clusters.

### 8. Explain DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
**Answer:**
*   Groups points that are closely packed together (high density).
*   Parameters: `eps` (radius), `min_samples`.
*   Points in low-density regions are marked as **Noise** (Outliers).

### 9. What are the advantages of DBSCAN over K-Means?
**Answer:**
*   Can find arbitrarily shaped clusters.
*   Robust to outliers (classifies them as noise).
*   Does not require specifying K.

### 10. What is Gaussian Mixture Model (GMM)?
**Answer:**
*   Probabilistic model. Assumes data is generated from a mixture of K Gaussian distributions.
*   Soft clustering (gives probability of belonging to each cluster).

### 11. What is the difference between Hard Clustering and Soft Clustering?
**Answer:**
*   **Hard**: Point belongs to exactly one cluster (K-Means).
*   **Soft**: Point has a probability distribution over clusters (GMM).

### 12. Explain the Expectation-Maximization (EM) algorithm.
**Answer:**
*   Used to fit GMM.
*   **E-step**: Estimate probability of each point belonging to each cluster (Responsibilities).
*   **M-step**: Update parameters (Mean, Covariance) based on weighted points.

### 13. How do you evaluate Clustering performance? (Silhouette, Davies-Bouldin).
**Answer:**
*   **Silhouette**: Separation distance.
*   **Davies-Bouldin**: Ratio of within-cluster scatter to between-cluster separation. Lower is better.
*   **Rand Index**: If ground truth labels exist.

### 14. Does K-Means guarantee a global optimum?
**Answer:**
*   **No**. It guarantees convergence to a **Local Optimum**.
*   Result depends on initialization. Run multiple times with different seeds.

### 15. How does scaling affect Clustering algorithms?
**Answer:**
*   **Crucial**. Distance-based algorithms (K-Means, DBSCAN) are sensitive to scale.
*   Always standardize data before clustering.

### 16. What is Spectral Clustering?
**Answer:**
*   Uses eigenvalues of the Similarity Matrix (Graph Laplacian).
*   Projects data to lower dimension, then runs K-Means.
*   Good for non-convex clusters.

### 17. What is Mean Shift clustering?
**Answer:**
*   Centroid-based algorithm.
*   Shifts centroids towards the mean of points within a window (Kernel Density Estimation).
*   Finds modes of the density. No need to specify K.

### 18. How do you handle outliers in K-Means?
**Answer:**
*   Remove them beforehand.
*   Use **K-Medoids** (uses actual points as centers, less sensitive to outliers).

### 19. Can you use Clustering for Anomaly Detection?
**Answer:**
*   **Yes**. Points far from any cluster centroid are anomalies.
*   Points in small/sparse clusters are anomalies.

### 20. What is the complexity of K-Means?
**Answer:**
*   $O(N \cdot K \cdot I \cdot D)$.
*   N=points, K=clusters, I=iterations, D=dimensions.
*   Linear in N (Fast).
