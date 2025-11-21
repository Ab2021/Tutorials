# Day 11 (Part 1): Advanced Linear Models

> **Phase**: 6 - Deep Dive
> **Topic**: Beyond OLS
> **Focus**: GLMs, Solvers, and Diagnostics
> **Reading Time**: 60 mins

---

## 1. Generalized Linear Models (GLM)

Linear Regression assumes Gaussian noise. What if target is Count (Poisson) or Binary (Bernoulli)?

### 1.1 The Components
1.  **Random Component**: Distribution of $Y$ (Exponential Family).
2.  **Systematic Component**: Linear predictor $\eta = X\beta$.
3.  **Link Function**: $g(\mu) = \eta$. Connects mean of $Y$ to linear predictor.
    *   *Linear*: Identity link.
    *   *Logistic*: Logit link.
    *   *Poisson*: Log link.

### 1.2 Solving GLMs
*   No closed form (like OLS Normal Equation).
*   **IRLS (Iteratively Reweighted Least Squares)**: Solves using weighted OLS in a loop.

---

## 2. Regularization Internals

### 2.1 Coordinate Descent (Lasso)
*   Lasso (L1) is non-differentiable at 0. Gradient Descent struggles.
*   **Coordinate Descent**: Optimize one weight $\beta_j$ at a time while fixing others.
*   **Soft Thresholding**: The closed-form update for $\beta_j$ sets it exactly to 0 if the correlation is weak. This creates sparsity.

### 2.2 Elastic Net
*   Combines L1 (Sparsity) and L2 (Grouping).
*   **Scenario**: Correlated features A and B.
    *   Lasso picks A, drops B (randomly).
    *   Ridge shrinks both.
    *   Elastic Net keeps both (grouped).

---

## 3. Tricky Interview Questions

### Q1: What happens to OLS if $X^T X$ is not invertible?
> **Answer**:
> *   **Cause**: Perfect Multicollinearity (Feature A = 2 * Feature B) or $N < P$.
> *   **Result**: Infinite solutions.
> *   **Fix**: Regularization (Ridge adds $\lambda I$ to diagonal, making it invertible). PCA.

### Q2: Explain Heteroscedasticity.
> **Answer**: The variance of errors is not constant (e.g., variance increases as prediction increases).
> *   **Impact**: OLS coefficients are unbiased but inefficient. Standard errors are wrong (Hypothesis tests fail).
> *   **Fix**: Log transform target, or use Weighted Least Squares (WLS).

### Q3: Why does Ridge Regression shrink coefficients but not set them to zero?
> **Answer**:
> *   **Geometry**: L2 penalty is a circle. Loss contours touch the circle usually at a non-axis point.
> *   **Calculus**: Derivative $2\beta$ goes to 0 linearly.
> *   **Lasso**: L1 penalty is a diamond. Corners are on axes. Derivative is constant ($\pm \lambda$), pushing weights all the way to 0.

---

## 4. Practical Edge Case: Outliers
*   **Problem**: MSE is sensitive to outliers (Quadratic penalty).
*   **Fix**: **Huber Loss**. Quadratic near 0, Linear far from 0. Combines best of MSE and MAE.

