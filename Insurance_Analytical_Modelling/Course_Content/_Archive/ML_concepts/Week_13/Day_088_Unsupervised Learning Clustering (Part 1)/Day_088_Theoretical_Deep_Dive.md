# Unsupervised Learning: Clustering (Part 1) - Theoretical Deep Dive

## Overview
Supervised learning requires labels (e.g., "Fraud" or "Not Fraud"). But what if we don't have labels? **Unsupervised Learning** finds hidden structures in data. This session covers **K-Means** and **Hierarchical Clustering** for customer segmentation and territory analysis.

---

## 1. Conceptual Foundation

### 1.1 The Clustering Problem

*   **Goal:** Group similar data points together.
*   **Input:** Data matrix $X$ (e.g., Age, Income, Policy Count).
*   **Output:** Cluster assignments $C_1, C_2, \dots, C_k$.
*   **Intuition:** "Birds of a feather flock together."

### 1.2 K-Means Clustering

*   **Algorithm:**
    1.  Pick $K$ random centroids.
    2.  Assign every point to the nearest centroid.
    3.  Move the centroid to the average of its assigned points.
    4.  Repeat until convergence.
*   **Pros:** Fast, simple.
*   **Cons:** Must specify $K$ in advance. Sensitive to outliers.

### 1.3 Hierarchical Clustering

*   **Agglomerative (Bottom-Up):**
    1.  Start with $N$ clusters (every point is a cluster).
    2.  Merge the two closest clusters.
    3.  Repeat until only 1 cluster remains.
*   **Dendrogram:** A tree diagram showing the merge history.
*   **Pros:** No need to pick $K$ upfront. Visualizes the hierarchy.
*   **Cons:** Slow ($O(N^3)$).

---

## 2. Mathematical Framework

### 2.1 K-Means Objective Function (Inertia)

$$ J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2 $$
*   Minimize the **Within-Cluster Sum of Squares (WCSS)**.
*   We want points to be close to their centroid $\mu_i$.

### 2.2 Distance Metrics

*   **Euclidean:** Straight line distance. $\sqrt{\sum (x_i - y_i)^2}$.
*   **Manhattan:** Taxicab distance. $\sum |x_i - y_i|$.
*   *Actuarial Note:* For territory clustering, use **Haversine Distance** (distance on a sphere) if using Lat/Lon coordinates.

---

## 3. Theoretical Properties

### 3.1 The Elbow Method

*   How to choose $K$?
*   Plot Inertia vs. $K$.
*   As $K$ increases, Inertia decreases.
*   **Elbow:** The point where the decrease slows down. That's the optimal $K$.

### 3.2 Scaling is Critical

*   If Feature A is "Income" (0-100,000) and Feature B is "Age" (0-100).
*   K-Means will only care about Income because the distance is dominated by the large numbers.
*   *Rule:* **Always Standardize** (Z-Score) your data before clustering.

---

## 4. Modeling Artifacts & Implementation

### 4.1 K-Means with Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Prepare Data
X = df[['Age', 'Premium', 'Claims']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow Method
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias, marker='o')
plt.title('Elbow Plot')
plt.show()

# 3. Final Model
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

### 4.2 Hierarchical Clustering (Dendrogram)

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage Matrix (Ward's method minimizes variance)
Z = linkage(X_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Customer Segmentation Dendrogram")
plt.show()
```

---

## 5. Evaluation & Validation

### 5.1 Silhouette Score

*   Measures how similar a point is to its own cluster compared to other clusters.
*   Range: -1 to +1.
*   **+1:** Perfect clustering.
*   **0:** Overlapping clusters.
*   **-1:** Wrong assignment.

### 5.2 Business Validation

*   **Profiling:** Calculate the mean of each feature for each cluster.
    *   *Cluster 1:* Young, High Premium, High Claims -> "High Risk Youth".
    *   *Cluster 2:* Old, Low Premium, Low Claims -> "Loyal Seniors".
*   *Test:* Do these profiles make sense to the Marketing team?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Categorical Variables**
    *   K-Means uses Euclidean distance, which doesn't work well for "Car Brand" (Ford vs. Toyota).
    *   *Fix:* Use **K-Modes** or **K-Prototypes** for mixed data.

2.  **Trap: The "Blob" Assumption**
    *   K-Means assumes clusters are spherical blobs.
    *   If your data looks like a "Crescent Moon", K-Means fails.
    *   *Fix:* Use DBSCAN (Density-Based Clustering).

### 6.2 Implementation Challenges

1.  **Random Initialization:**
    *   K-Means results can change every time you run it.
    *   *Fix:* Set `random_state=42` and use `n_init=10` (runs it 10 times and picks the best).

---

## 7. Advanced Topics & Extensions

### 7.1 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

*   Groups points that are packed closely together.
*   **Pros:** Can find arbitrary shapes. Handles outliers (noise) automatically.
*   **Cons:** Hard to tune `epsilon` and `min_samples`.

### 7.2 Gaussian Mixture Models (GMM)

*   Probabilistic Clustering.
*   Assumes data comes from a mix of Gaussian distributions.
*   *Output:* Probability of belonging to Cluster A vs. Cluster B (Soft Clustering).

---

## 8. Regulatory & Governance Considerations

### 8.1 Redlining & Fairness

*   **Risk:** Clustering Zip Codes might accidentally recreate "Redlining" maps (discriminating by neighborhood).
*   **Governance:** Check if the clusters correlate highly with Protected Classes (Race, Religion). If so, investigate.

---

## 9. Practical Example

### 9.1 Worked Example: Territory Analysis

**Scenario:**
*   Pricing Auto Insurance in a new state.
*   **Data:** 1,000 Zip Codes with Loss Ratios.
*   **Method:** Hierarchical Clustering on (Lat, Lon, Loss Ratio).
*   **Result:** Grouped Zips into 10 "Rating Territories".
*   **Outcome:** Simplified the rating plan while capturing geographic risk.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **K-Means** partitions data into K spheres.
2.  **Hierarchical** builds a tree of clusters.
3.  **Scaling** is mandatory.

### 10.2 When to Use This Knowledge
*   **Marketing:** Customer Personas.
*   **Pricing:** Territory definition.
*   **Fraud:** Outlier detection (Small clusters).

### 10.3 Critical Success Factors
1.  **Interpretability:** If you can't name the cluster, it's useless.
2.  **Stability:** Do the clusters change if you remove 10% of the data?

### 10.4 Further Reading
*   **Jain:** "Data Clustering: 50 Years Beyond K-Means".

---

## Appendix

### A. Glossary
*   **Centroid:** The center of a cluster.
*   **Inertia:** Sum of squared distances to the nearest centroid.
*   **Dendrogram:** Tree diagram for hierarchical clustering.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Euclidean Dist** | $\sqrt{\sum (x-y)^2}$ | Similarity Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
