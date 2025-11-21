# Day 14: Support Vector Machines (SVM) - Interview Questions

> **Topic**: Max Margin Classifiers
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Explain the intuition behind SVM.
**Answer:**
*   Find the hyperplane that separates classes with the **Maximum Margin**.
*   Margin = Distance to nearest data points (Support Vectors).
*   "Wide road" separation is more robust to noise.

### 2. What is a Hyperplane?
**Answer:**
*   A subspace of dimension $D-1$.
*   In 2D: Line. In 3D: Plane.
*   Equation: $w^T x + b = 0$.

### 3. What are Support Vectors?
**Answer:**
*   The data points closest to the hyperplane.
*   They "support" (define) the margin.
*   If you remove other points, the boundary doesn't change. Only SVs matter.

### 4. What is the Margin? Why do we want to maximize it?
**Answer:**
*   Distance between the decision boundary and the closest points.
*   **Maximize**: Large margin implies lower generalization error (Structural Risk Minimization).

### 5. Explain Hard Margin vs Soft Margin SVM.
**Answer:**
*   **Hard**: Assumes data is perfectly separable. No errors allowed. Sensitive to outliers.
*   **Soft**: Allows some misclassification (Slack variables $\xi$). Robust to noise. Controlled by C.

### 6. What is the role of the C parameter in SVM?
**Answer:**
*   Regularization parameter. Tradeoff between Margin Width and Classification Error.
*   **High C**: Strict. Small margin. Penalizes errors heavily. Risk of Overfitting.
*   **Low C**: Loose. Wide margin. Allows errors. Smoother boundary.

### 7. What is the Kernel Trick?
**Answer:**
*   Mapping data to a higher-dimensional space where it becomes linearly separable.
*   **Trick**: We don't compute the mapping $\phi(x)$ explicitly. We only need the dot product $K(x, y) = \phi(x) \cdot \phi(y)$.
*   Allows infinite-dimensional spaces (RBF) cheaply.

### 8. Name some common Kernels (Linear, Polynomial, RBF).
**Answer:**
*   **Linear**: $x \cdot y$.
*   **Polynomial**: $(x \cdot y + c)^d$.
*   **RBF (Gaussian)**: $e^{-\gamma ||x-y||^2}$.

### 9. Explain the RBF (Radial Basis Function) Kernel.
**Answer:**
*   Measures similarity based on distance.
*   Projects data into infinite dimensions.
*   Creates circular/blobby decision boundaries around data points.

### 10. What is the Gamma parameter in RBF Kernel?
**Answer:**
*   Defines "reach" of a single training example.
*   **High Gamma**: Narrow peak. Points must be very close to affect each other. Overfitting (Islands).
*   **Low Gamma**: Broad bump. Smooth boundary. Underfitting.

### 11. What is the Hinge Loss?
**Answer:**
*   Loss function for SVM.
*   $L = \max(0, 1 - y(w^T x + b))$.
*   If point is correctly classified and outside margin ($y \cdot pred > 1$), Loss = 0.
*   If inside margin or wrong, Loss increases linearly.

### 12. Can SVM be used for Regression? (SVR).
**Answer:**
*   Yes. Support Vector Regression.
*   Goal: Fit a tube of width $\epsilon$ around the data.
*   Points inside tube have 0 loss. Points outside are penalized.

### 13. How does SVM handle non-linear data?
**Answer:**
*   Using Kernels (RBF, Poly).

### 14. Does SVM require Feature Scaling? Why?
**Answer:**
*   **Yes, absolutely**.
*   SVM maximizes margin (distance). Distance calculations are dominated by large-scale features.
*   Must normalize inputs.

### 15. How does SVM handle Multiclass Classification?
**Answer:**
*   Not natively supported.
*   Uses **One-vs-One** (Train $K(K-1)/2$ classifiers) or **One-vs-Rest**.
*   Sklearn handles this automatically.

### 16. What are the pros and cons of SVM?
**Answer:**
*   **Pros**: Effective in high dimensions ($D > N$). Memory efficient (only stores SVs). Global optimum (Convex).
*   **Cons**: Slow for large datasets ($O(N^2)$). No probability output (requires Platt scaling). Sensitive to noise/C.

### 17. When would you use a Linear Kernel over an RBF Kernel?
**Answer:**
*   When number of features is very large (Text classification). Data is likely linearly separable in high dimensions.
*   When speed is critical.

### 18. What is the time complexity of training an SVM?
**Answer:**
*   Between $O(N^2)$ and $O(N^3)$.
*   Depends on solver (SMO).
*   Bad for $N > 100,000$.

### 19. How does SVM compare to Logistic Regression?
**Answer:**
*   **Linear SVM** $\approx$ **Logistic Regression**.
*   SVM focuses on points near boundary (SVs). LR considers all points (via Log Loss).
*   SVM is better for "hard" margins. LR gives probabilities.

### 20. What is the "Dual Problem" in SVM optimization?
**Answer:**
*   Primal: Optimize weights $w$.
*   Dual: Optimize Lagrange multipliers $\alpha$.
*   Dual formulation allows using the Kernel Trick (depends only on dot products).
