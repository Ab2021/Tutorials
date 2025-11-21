# Day 11: Linear Regression - The Hello World of ML

> **Phase**: 2 - Core Algorithms
> **Week**: 3 - Supervised Learning
> **Focus**: Regression & Regularization
> **Reading Time**: 50 mins

---

## 1. The Theory of Linearity

Linear Regression is the simplest model, yet it powers massive systems (e.g., ad bidding, stock forecasting).

### 1.1 The Model
$$y = w_0 + w_1 x_1 + \dots + w_n x_n + \epsilon$$
*   **Assumption**: The relationship between inputs $X$ and target $y$ is linear.
*   **Assumption**: The error term $\epsilon$ is normally distributed with constant variance (Homoscedasticity).

### 1.2 Solving the Model
1.  **Closed Form (OLS)**: $w = (X^T X)^{-1} X^T y$.
    *   *Pros*: Exact solution.
    *   *Cons*: Inverting a matrix is $O(N^3)$. Fails if $N$ is huge or if columns are collinear (singular matrix).
2.  **Gradient Descent**: Iteratively update weights.
    *   *Pros*: Scales to infinite data (SGD). Works when $X^T X$ is non-invertible.

---

## 2. Regularization: Controlling Complexity

When a model fits noise instead of signal, it is **overfitting**. Regularization adds a penalty to the Loss function to discourage complex models (large weights).

### 2.1 Ridge Regression (L2)
$$Loss = MSE + \lambda \sum w_i^2$$
*   **Effect**: Shrinks weights towards zero but never *exactly* to zero.
*   **Use Case**: When you have many correlated features (Multicollinearity). L2 spreads the weight among them.

### 2.2 Lasso Regression (L1)
$$Loss = MSE + \lambda \sum |w_i|$$
*   **Effect**: Forces some weights to become **exactly zero**.
*   **Use Case**: Feature Selection. If you have 1000 features but suspect only 10 matter, Lasso will find them.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Outliers
**Scenario**: You are predicting house prices. One mansion costs \$100M.
**Result**: Squared Error ($(y - \hat{y})^2$) explodes. The line is pulled drastically towards the outlier to minimize this huge error.
**Solution**:
*   **Huber Loss**: Quadratic for small errors, Linear for large errors. (Robust Regression).
*   **Log-Transform**: Compress the target variable range.

### Challenge 2: Collinearity
**Scenario**: You have two features: "Size in sq ft" and "Size in sq meters". They are perfectly correlated.
**Result**: The OLS solution becomes unstable (determinant is 0). The weights might be huge and opposite (e.g., $+1000$ for sq ft, $-10000$ for sq meters) to cancel each other out.
**Solution**: Drop one feature, use PCA, or use Ridge (L2) Regularization.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What are the assumptions of Linear Regression?**
> **Answer**:
> 1.  **Linearity**: The relationship is linear.
> 2.  **Independence**: Errors are independent (no autocorrelation).
> 3.  **Homoscedasticity**: The variance of errors is constant across all levels of X.
> 4.  **Normality**: Errors are normally distributed (important for hypothesis testing, not for prediction).
> 5.  **No Multicollinearity**: Features are not perfectly correlated.

**Q2: Explain the Bias-Variance Tradeoff.**
> **Answer**:
> *   **Bias**: Error due to overly simplistic assumptions (Underfitting). Linear Regression has high bias if the data is curved.
> *   **Variance**: Error due to sensitivity to fluctuations in the training set (Overfitting). A high-degree polynomial has high variance.
> *   **Goal**: Find the sweet spot (Total Error = Bias$^2$ + Variance + Irreducible Error).

**Q3: Why is R-squared not a perfect metric?**
> **Answer**: $R^2$ never decreases when you add more features, even if they are junk. A model with 1000 random features will have high $R^2$. **Adjusted $R^2$** penalizes the addition of useless features.

---

## 5. Further Reading
- [The Elements of Statistical Learning (Hastie et al.)](https://hastie.su.domains/ElemStatLearn/)
- [Regularization Visualized](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
