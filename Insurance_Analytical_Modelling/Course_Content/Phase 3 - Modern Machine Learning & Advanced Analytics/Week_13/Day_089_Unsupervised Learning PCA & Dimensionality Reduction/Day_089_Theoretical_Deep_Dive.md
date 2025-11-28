# Unsupervised Learning: PCA & Dimensionality Reduction - Theoretical Deep Dive

## Overview
"Big Data" is great, but "Wide Data" (too many columns) is a nightmare. This session covers **Dimensionality Reduction**: How to compress 500 variables into 10 **Principal Components** without losing the signal.

---

## 1. Conceptual Foundation

### 1.1 The Curse of Dimensionality

*   **Sparsity:** As dimensions increase, data points get further apart.
*   **Overfitting:** A model with 1000 features can easily memorize 1000 data points.
*   **Multicollinearity:** "Car Length", "Car Width", "Car Weight" are all highly correlated. GLMs hate this.

### 1.2 Principal Component Analysis (PCA)

*   **Goal:** Find new axes (Principal Components) that capture the maximum variance.
*   **Mechanism:** Rotate the dataset so that the first axis (PC1) points in the direction of greatest spread.
*   **Result:** PC1 captures 60% of info, PC2 captures 20%, etc. You can drop PC3-PC500 and still keep 80% of the info.

### 1.3 t-SNE & UMAP (Manifold Learning)

*   **PCA** is linear. It fails if the data is shaped like a "Swiss Roll".
*   **t-SNE / UMAP:** Non-linear. They try to preserve local neighborhoods.
*   **Use:** Visualization. Compressing 100 dimensions into 2D scatter plots to see clusters.

---

## 2. Mathematical Framework

### 2.1 Eigenvalues and Eigenvectors

*   **Covariance Matrix ($\Sigma$):** Describes how variables vary together.
*   **Eigenvector ($v$):** The direction of the new axis.
*   **Eigenvalue ($\lambda$):** The amount of variance captured by that axis.
*   **Equation:** $\Sigma v = \lambda v$.

### 2.2 Variance Explained Ratio

$$ \text{Explained Variance}_i = \frac{\lambda_i}{\sum \lambda_j} $$
*   We select the top $k$ components such that the cumulative explained variance > 95%.

---

## 3. Theoretical Properties

### 3.1 Orthogonality

*   Principal Components are **Orthogonal** (Perpendicular) to each other.
*   *Actuarial Benefit:* This eliminates Multicollinearity. You can feed PC1, PC2, PC3 into a GLM and the p-values will be stable.

### 3.2 Interpretability (The PCA Trade-off)

*   **Original Variable:** "Age". (Easy to interpret).
*   **PC1:** $0.5 \times \text{Age} + 0.3 \times \text{Income} - 0.2 \times \text{Claims}$. (Hard to interpret).
*   *Solution:* Look at the **Loadings** (the weights) to name the components (e.g., "Socio-Economic Factor").

---

## 4. Modeling Artifacts & Implementation

### 4.1 PCA with Scikit-Learn

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 1. Standardize (Crucial!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 2. Fit PCA
pca = PCA(n_components=0.95) # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original Dimensions: {X_scaled.shape[1]}")
print(f"Reduced Dimensions: {X_pca.shape[1]}")

# 3. Scree Plot
plt.plot(pca.explained_variance_ratio_)
plt.title("Scree Plot")
plt.show()
```

### 4.2 UMAP for Visualization

```python
import umap

reducer = umap.UMAP()
embedding = reducer.fit_transform(X_scaled)

plt.scatter(embedding[:, 0], embedding[:, 1], c=df['ClaimCount'], cmap='Spectral')
plt.title("UMAP Projection of Policyholders")
plt.show()
```

---

## 5. Evaluation & Validation

### 5.1 Reconstruction Error

*   Can we reconstruct the original data from the compressed PCs?
*   Low error = Good compression.

### 5.2 Downstream Performance

*   Train a GLM on Raw Features vs. PCA Features.
*   If PCA-GLM has similar Gini Score but is faster and more stable, PCA wins.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Scaling**
    *   If you don't scale, the variable with the biggest numbers (e.g., "Sum Insured") will dominate PC1.
    *   *Fix:* `StandardScaler` is mandatory before PCA.

2.  **Trap: Categorical Data**
    *   PCA is for continuous data. One-hot encoded variables don't work well with standard PCA.
    *   *Fix:* Use **MCA (Multiple Correspondence Analysis)** for categorical data.

### 6.2 Implementation Challenges

1.  **Data Leakage:**
    *   Fit PCA on Training Data, then `transform` Test Data.
    *   Do *not* fit PCA on the whole dataset.

---

## 7. Advanced Topics & Extensions

### 7.1 Autoencoders (Non-linear PCA)

*   A Neural Network that compresses data (Encoder) and reconstructs it (Decoder).
*   Can capture non-linear relationships that PCA misses.

### 7.2 Factor Analysis

*   Similar to PCA but assumes underlying "Latent Factors" cause the observed variables.
*   More interpretable for social sciences (and potentially behavioral insurance).

---

## 8. Regulatory & Governance Considerations

### 8.1 "Black Box" Features

*   **Regulator:** "Why did you rate this person high?"
*   **Actuary:** "Because PC1 is high."
*   **Regulator:** "What is PC1?"
*   **Actuary:** "It's a linear combination of 50 variables..."
*   *Risk:* Lack of transparency. Use PCA for *internal* analysis or *unregulated* lines first.

---

## 9. Practical Example

### 9.1 Worked Example: Vehicle Feature Reduction

**Scenario:**
*   Pricing Auto Insurance.
*   **Data:** 50 vehicle specs (Length, Width, Height, Wheelbase, Curb Weight, HP, Torque...).
*   **Problem:** High multicollinearity.
*   **PCA:**
    *   **PC1:** Loaded heavily on Length, Width, Weight. -> Named "Size Factor".
    *   **PC2:** Loaded on HP, Torque, 0-60 time. -> Named "Performance Factor".
*   **GLM:** Used "Size" and "Performance" as predictors. Stable and intuitive.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **PCA** rotates data to maximize variance.
2.  **Eigenvalues** tell you how important a component is.
3.  **UMAP** is for pretty pictures (visualization).

### 10.2 When to Use This Knowledge
*   **GLM Modeling:** Removing multicollinearity.
*   **EDA:** Visualizing high-dimensional datasets.

### 10.3 Critical Success Factors
1.  **Scale your data.**
2.  **Check the loadings** to interpret the components.

### 10.4 Further Reading
*   **Jolliffe:** "Principal Component Analysis".

---

## Appendix

### A. Glossary
*   **Orthogonal:** At 90 degrees (uncorrelated).
*   **Scree Plot:** A line plot of eigenvalues.
*   **Manifold:** A lower-dimensional shape embedded in high-dimensional space.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PCA Transformation** | $Z = XW$ | Project Data |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
