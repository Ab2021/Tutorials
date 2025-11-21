# Day 12: Logistic Regression - Interview Questions

> **Topic**: Classification Basics
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the Sigmoid function? Write its formula.
**Answer:**
*   S-shaped curve mapping real numbers to $(0, 1)$.
*   $\sigma(z) = \frac{1}{1 + e^{-z}}$.

### 2. Why is it called "Regression" if it's used for Classification?
**Answer:**
*   It predicts a **probability** (continuous value) between 0 and 1.
*   We apply a threshold (e.g., 0.5) to classify, but the underlying model is a regression on the log-odds.

### 3. What is the Cost Function for Logistic Regression? (Log Loss).
**Answer:**
*   Binary Cross-Entropy.
*   $J = - \frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$.
*   Convex function (unlike MSE for logistic).

### 4. Why can't we use Mean Squared Error (MSE) for Logistic Regression?
**Answer:**
*   If we put Sigmoid into MSE, the cost function becomes **Non-Convex** (wavy).
*   Gradient Descent would get stuck in local minima.
*   Log Loss guarantees convexity.

### 5. How do you interpret the coefficients in Logistic Regression? (Log-odds).
**Answer:**
*   $\beta_j$ is the change in the **Log-Odds** of the positive class for a 1-unit increase in $X_j$.
*   $Odds = e^{\beta_0 + \beta_1 x}$.
*   Increase $x$ by 1 -> Odds multiply by $e^{\beta_1}$.

### 6. What is the Decision Boundary?
**Answer:**
*   The line (or hyperplane) where probability = 0.5.
*   Occurs when $z = \beta^T x = 0$.
*   It is a linear boundary.

### 7. How does Logistic Regression handle Multiclass Classification? (OvR vs Softmax).
**Answer:**
*   **One-vs-Rest (OvR)**: Train K binary classifiers (Class 1 vs All, Class 2 vs All). Pick max confidence.
*   **Softmax (Multinomial)**: Generalization of Sigmoid. Predicts distribution over K classes directly.

### 8. What is the Softmax function?
**Answer:**
*   $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$.
*   Ensures probabilities sum to 1.

### 9. Explain the concept of "Linear Separability".
**Answer:**
*   Data can be separated into classes by a straight line (or hyperplane).
*   Logistic Regression works best here. XOR problem is not linearly separable.

### 10. How do outliers affect Logistic Regression?
**Answer:**
*   Less sensitive than Linear Regression (Sigmoid saturates), but extreme outliers can still shift the decision boundary.

### 11. What is the Confusion Matrix?
**Answer:**
*   Table showing TP, TN, FP, FN.
*   Compares Predicted vs Actual.

### 12. Define Precision, Recall, and F1-Score.
**Answer:**
*   **Precision**: TP / (TP + FP). "Of all predicted positives, how many are real?" (Quality).
*   **Recall**: TP / (TP + FN). "Of all real positives, how many did we find?" (Quantity).
*   **F1**: Harmonic mean. $2 \cdot \frac{P \cdot R}{P + R}$.

### 13. What is the ROC Curve? What does AUC represent?
**Answer:**
*   **ROC**: Plot of TPR (Recall) vs FPR (1-Specificity) at various thresholds.
*   **AUC**: Area Under Curve. Probability that a random positive example is ranked higher than a random negative example. 0.5 = Random. 1.0 = Perfect.

### 14. When would you choose Precision over Recall?
**Answer:**
*   When **False Positives** are costly.
*   Example: Spam filter (Don't want to delete important email).

### 15. When would you choose Recall over Precision?
**Answer:**
*   When **False Negatives** are dangerous.
*   Example: Cancer detection (Better to flag healthy person than miss a sick one).

### 16. How do you handle imbalanced datasets in Logistic Regression?
**Answer:**
*   **Class Weights**: Penalize errors on minority class more ($Loss \times Weight$).
*   **Resampling**: Oversample minority (SMOTE) or Undersample majority.
*   **Change Metric**: Use F1 or AUC, not Accuracy.

### 17. Does Logistic Regression require feature scaling? Why?
**Answer:**
*   **Yes**, if using Regularization (L1/L2).
*   If features have different scales, regularization penalizes weights of small features unfairly.
*   Also helps Gradient Descent converge faster.

### 18. Can Logistic Regression solve non-linear problems? How?
**Answer:**
*   **Feature Engineering**.
*   Add polynomial features ($x^2, x_1 x_2$) manually. The model is linear in the *new* feature space, but non-linear in original space.

### 19. What is the impact of correlation among independent variables in Logistic Regression?
**Answer:**
*   Multicollinearity.
*   Coefficients become unstable and uninterpretable.
*   Doesn't affect predictive power much, but ruins inference.

### 20. Compare Logistic Regression with Linear Regression.
**Answer:**
*   **Target**: Categorical vs Continuous.
*   **Function**: Sigmoid vs Identity.
*   **Loss**: Log Loss vs MSE.
*   **Distribution**: Bernoulli vs Gaussian.
