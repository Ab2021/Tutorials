# Day 14: Support Vector Machines (SVM)

> **Phase**: 2 - Core Algorithms
> **Week**: 3 - Supervised Learning
> **Focus**: Margins & Kernels
> **Reading Time**: 45 mins

---

## 1. The Maximum Margin Classifier

SVMs are mathematically elegant. Their goal is not just to separate classes, but to separate them with the **widest possible street** (margin).

### 1.1 The Hyperplane
In $N$ dimensions, a hyperplane is an $N-1$ dimensional flat subspace.
*   **Support Vectors**: The data points closest to the hyperplane. These are the *only* points that matter. If you move other points, the boundary doesn't change. This makes SVM memory efficient (in theory).

### 1.2 Hard vs. Soft Margin
*   **Hard Margin**: Strictly no errors allowed. Only works if data is perfectly separable. Sensitive to outliers.
*   **Soft Margin (C parameter)**: Allows some misclassification to achieve a wider margin.
    *   **High C**: Strict. Narrow margin. Low bias, High variance.
    *   **Low C**: Loose. Wide margin. High bias, Low variance.

---

## 2. The Kernel Trick: Bending Space

What if data is not linearly separable (e.g., concentric circles)?
We project data into a higher dimension where it *is* separable.

### 2.1 The Trick
Computing coordinates in high dimensions is expensive. The Kernel Trick allows us to compute the dot product in the high-dimensional space **without ever actually visiting it**.
$$K(x, y) = \phi(x) \cdot \phi(y)$$
*   **RBF Kernel (Radial Basis Function)**: Infinite dimensional projection. Measures similarity based on distance. Most popular.
*   **Polynomial Kernel**: Projects into polynomial feature space.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Scaling
**Scenario**: You have 1 Million rows. You try `SVC()`. It hangs forever.
**Theory**: Standard SVM solvers are $O(N^2)$ or $O(N^3)$. They compute the kernel matrix ($N \times N$). 1M x 1M is too big.
**Solution**:
*   **LinearSVC**: Uses a different solver (LibLinear) which is $O(N)$.
*   **SGDClassifier**: Use Gradient Descent with Hinge Loss. Approximates SVM.
*   **Approximation**: Nystroem method to approximate the kernel map.

### Challenge 2: Feature Scaling
**Scenario**: Feature A is "Age" (0-100). Feature B is "Salary" (0-100,000).
**Result**: SVM is distance-based. Salary will dominate the distance calculation. The decision boundary will ignore Age.
**Solution**: **Always** normalize/standardize data (Mean=0, Var=1) before using SVMs.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the Hinge Loss?**
> **Answer**: $L = \max(0, 1 - y \cdot f(x))$.
> *   If the point is correctly classified and outside the margin ($y \cdot f(x) > 1$), loss is 0.
> *   If it's inside the margin or misclassified, loss increases linearly.
> *   This forces the model to focus on the "hard" examples (support vectors).

**Q2: Why is SVM less popular in the Deep Learning era?**
> **Answer**:
> 1.  **Scalability**: Kernel SVMs don't scale to massive datasets ($O(N^2)$).
> 2.  **Feature Engineering**: SVMs require manual kernels/features. Neural Nets learn features automatically.
> 3.  **Data Types**: SVMs struggle with raw perceptual data (images, audio) compared to CNNs/RNNs.

**Q3: What is the relationship between Logistic Regression and Linear SVM?**
> **Answer**: Both are linear classifiers.
> *   **LogReg**: Minimizes Log Loss. Outputs probabilities. Focuses on all points (though confident ones have low weight).
> *   **SVM**: Minimizes Hinge Loss. Outputs class labels. Focuses only on Support Vectors.

---

## 5. Further Reading
- [Support Vector Machines (Scikit-Learn)](https://scikit-learn.org/stable/modules/svm.html)
- [The Kernel Trick Explained](https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f)
