# Day 43: ML Coding Round - Algorithms

> **Phase**: 5 - Interview Mastery
> **Week**: 9 - Behavioral & Coding
> **Focus**: "Implement X from Scratch"
> **Reading Time**: 60 mins

---

## 1. The "From Scratch" Round

Interviewers want to see if you understand the math, not just `import sklearn`.

### 1.1 K-Means Clustering
*   **Steps**:
    1.  Initialize K centroids randomly.
    2.  **Assignment**: Assign each point to nearest centroid (Euclidean distance).
    3.  **Update**: Move centroid to the mean of its assigned points.
    4.  Repeat until convergence.
*   **Edge Case**: Empty clusters. Initialization sensitivity (K-Means++).

### 1.2 Softmax & Cross-Entropy
*   **Naive Softmax**: $e^{x_i} / \sum e^{x_j}$.
*   **Issue**: If $x_i = 1000$, $e^{1000}$ overflows float.
*   **Stable Softmax**: Subtract max value. $e^{x_i - m} / \sum e^{x_j - m}$.
*   **LogSumExp**: For loss calculation, work in log space to avoid underflow.

### 1.3 Self-Attention
*   **Task**: Implement `attention(Q, K, V)`.
*   **Key**: Handle shapes `(Batch, Seq, Dim)`. Use `torch.matmul` or `np.dot`. Don't forget scaling factor.

---

## 2. Practice Problems

### Problem 1: IoU (Intersection over Union)
**Task**: Given two bounding boxes `[x1, y1, x2, y2]`, compute IoU.
**Logic**:
*   `inter_x1 = max(box1[0], box2[0])`
*   `inter_x2 = min(box1[2], box2[2])`
*   `inter_area = max(0, inter_x2 - inter_x1) * ...`

### Problem 2: Stratified Split
**Task**: Split a dataset retaining class ratios.
**Logic**: Group indices by class. Shuffle. Split each group. Combine.

---

## 3. Interview Preparation

### Conceptual Questions

**Q1: Why do we divide by N-1 in sample variance?**
> **Answer**: Bessel's Correction. It provides an unbiased estimator of the population variance. Dividing by N underestimates variance because the sample mean is closer to the sample data than the true population mean.

**Q2: How do you implement Gradient Descent?**
> **Answer**:
> `w = w - learning_rate * gradient(loss, w)`
> You need to manually compute the derivative of the loss function (e.g., MSE derivative is $2(y - \hat{y}) \cdot x$).

---

## 5. Further Reading
- [ML Algorithms from Scratch (Repo)](https://github.com/eriklindernoren/ML-From-Scratch)
- [The Softmax Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
