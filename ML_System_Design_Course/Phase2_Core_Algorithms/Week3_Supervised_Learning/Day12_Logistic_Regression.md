# Day 12: Logistic Regression & Classification Metrics

> **Phase**: 2 - Core Algorithms
> **Week**: 3 - Supervised Learning
> **Focus**: Binary Classification & Evaluation
> **Reading Time**: 50 mins

---

## 1. Logistic Regression: It's a Classifier

Despite the name, Logistic Regression is used for classification, not regression. It predicts the **probability** that an instance belongs to the positive class.

### 1.1 The Sigmoid Function
$$P(y=1|x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$
*   **S-Curve**: Maps any real number $(-\infty, \infty)$ to $(0, 1)$.
*   **Decision Boundary**: The line where probability = 0.5 (i.e., $w^T x = 0$). It is a **linear classifier**. It cannot solve XOR problems without feature engineering.

### 1.2 Log Loss (Binary Cross-Entropy)
We cannot use MSE (Mean Squared Error) because the sigmoid makes the loss function non-convex (many local minima).
$$J(w) = - \frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$
This loss is convex, guaranteeing a global minimum.

---

## 2. Evaluation Metrics: Beyond Accuracy

Accuracy is dangerous. In a fraud dataset (99.9% legit, 0.1% fraud), a model that predicts "Legit" for everyone has 99.9% accuracy but 0 value.

### 2.1 Precision & Recall
*   **Precision**: Of all predicted positives, how many were real? (Quality). "Don't spam users."
*   **Recall (Sensitivity)**: Of all real positives, how many did we catch? (Quantity). "Don't miss fraud."
*   **F1 Score**: Harmonic mean of Precision and Recall.

### 2.2 ROC and AUC
*   **ROC Curve**: Plot of TPR (Recall) vs. FPR (False Alarm Rate) at all possible thresholds (0.0 to 1.0).
*   **AUC (Area Under Curve)**: Probability that the model ranks a random positive example higher than a random negative example.
    *   AUC = 0.5: Random guessing.
    *   AUC = 1.0: Perfect classifier.
    *   **Benefit**: AUC is threshold-invariant. It measures the *ranking quality* of the model.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Imbalanced Data
**Scenario**: Detecting rare diseases (1 in 10,000).
**Problem**: The model learns to output 0.0001 for everyone to minimize loss.
**Solution**:
1.  **Resampling**: Oversample the minority class (SMOTE) or undersample the majority.
2.  **Class Weights**: Modify the loss function to penalize errors on the minority class 10,000x more.
3.  **Threshold Moving**: Don't use 0.5. Use 0.01 as the cutoff.

### Challenge 2: Linearity
**Scenario**: Data is concentric circles (Class 0 inside, Class 1 outside).
**Problem**: Logistic Regression draws a straight line. It will fail (Accuracy ~50%).
**Solution**: Feature Engineering. Add $x_1^2$ and $x_2^2$ as features. In the squared space, the circles become linearly separable.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why is the decision boundary of Logistic Regression linear?**
> **Answer**: The decision boundary is where $P(y=1) = 0.5$.
> $\frac{1}{1 + e^{-z}} = 0.5 \implies e^{-z} = 1 \implies z = 0$.
> Since $z = w^T x + b$, the boundary is the hyperplane $w^T x + b = 0$, which is linear.

**Q2: Can Logistic Regression be used for multi-class classification?**
> **Answer**: Yes.
> 1.  **One-vs-Rest (OvR)**: Train $K$ binary classifiers.
> 2.  **Softmax Regression (Multinomial)**: Generalizes the sigmoid to $K$ classes. $\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$.

**Q3: What happens if your data is perfectly linearly separable?**
> **Answer**: The weights will explode to infinity. To drive the predicted probability to exactly 1.0 or 0.0, the dot product $w^T x$ must go to $\pm \infty$. Regularization (L2) is required to keep weights finite.

---

## 5. Further Reading
- [Google ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Understanding AUC - ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
