# Day 11: Linear Regression - Interview Questions

> **Topic**: Supervised Learning Basics
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What are the assumptions of Linear Regression?
**Answer:**
1.  **Linearity**: Relationship between X and Y is linear.
2.  **Independence**: Observations are independent.
3.  **Homoscedasticity**: Constant variance of errors.
4.  **Normality**: Errors are normally distributed.
5.  **No Multicollinearity**: Features are not perfectly correlated.

### 2. Explain the Ordinary Least Squares (OLS) method.
**Answer:**
*   Method to estimate coefficients $\beta$.
*   Minimizes the Sum of Squared Residuals (SSR): $\sum (y_i - \hat{y}_i)^2$.
*   Has a closed-form solution: $\beta = (X^T X)^{-1} X^T y$.

### 3. What is the Cost Function for Linear Regression?
**Answer:**
*   Mean Squared Error (MSE).
*   $J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$.

### 4. What is R-squared ($R^2$)? How do you interpret it?
**Answer:**
*   Coefficient of Determination.
*   Proportion of variance in Y explained by X.
*   $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$.
*   1 = Perfect fit. 0 = Model is as good as predicting the mean. < 0 = Worse than mean.

### 5. What is Adjusted R-squared? Why is it better than $R^2$?
**Answer:**
*   $R^2$ always increases (or stays same) when you add features, even junk ones.
*   **Adjusted $R^2$** penalizes complexity (number of predictors $p$).
*   Decreases if the new feature doesn't improve the model enough to justify the cost.

### 6. Explain Multicollinearity. Why is it a problem?
**Answer:**
*   When features are highly correlated.
*   **Problem**: Coefficients become unstable (high variance). Small change in data -> huge change in $\beta$.
*   Interpretation becomes impossible ("Holding other variables constant" makes no sense if they move together).

### 7. How do you detect Multicollinearity? (Hint: VIF).
**Answer:**
*   **Correlation Matrix**: High values (> 0.8).
*   **VIF (Variance Inflation Factor)**: $VIF_i = 1 / (1 - R_i^2)$.
*   VIF > 5 or 10 indicates problem.

### 8. What is Heteroscedasticity?
**Answer:**
*   Variance of errors is **not constant** (e.g., errors get larger as Y gets larger).
*   Violates OLS assumption. Standard errors become wrong -> Hypothesis tests fail.
*   **Fix**: Log transform Y.

### 9. What is the difference between Simple and Multiple Linear Regression?
**Answer:**
*   **Simple**: One feature ($y = mx + c$).
*   **Multiple**: Many features ($y = \beta_0 + \beta_1 x_1 + ...$).

### 10. Explain Ridge Regression (L2 Regularization).
**Answer:**
*   Adds penalty $\lambda \sum \beta_j^2$ to cost.
*   Shrinks coefficients towards zero but not exactly zero.
*   Handles multicollinearity well.

### 11. Explain Lasso Regression (L1 Regularization).
**Answer:**
*   Adds penalty $\lambda \sum |\beta_j|$ to cost.
*   Shrinks coefficients to **exactly zero**.
*   Performs **Feature Selection**.

### 12. What is Elastic Net?
**Answer:**
*   Combination of Ridge and Lasso.
*   $\lambda_1 L1 + \lambda_2 L2$.
*   Good when there are correlated features (Lasso picks one at random, Elastic Net picks both).

### 13. Why does Lasso result in sparse models (feature selection)?
**Answer:**
*   Geometric intuition: The L1 constraint region is a diamond (square rotated).
*   The loss contours usually hit the "corners" of the diamond first. Corners lie on axes where some weights are zero.
*   L2 region is a circle; contours hit anywhere (non-zero).

### 14. What is the Normal Equation? When should you use it vs Gradient Descent?
**Answer:**
*   $\beta = (X^T X)^{-1} X^T y$.
*   **Use**: Small datasets ($N < 10,000$). Exact solution. No learning rate.
*   **Avoid**: Large datasets. Inverting matrix is $O(N^3)$.

### 15. How do outliers affect Linear Regression?
**Answer:**
*   Squared error term $(y - \hat{y})^2$ penalizes large errors heavily.
*   A single outlier can pull the regression line significantly (High leverage).
*   **Fix**: Remove outliers or use Robust Regression (Huber Loss).

### 16. What is Polynomial Regression? Is it still a "linear" model?
**Answer:**
*   Modeling non-linear relationships by adding powers: $y = \beta_0 + \beta_1 x + \beta_2 x^2$.
*   **Yes**, it is linear in the **parameters** ($\beta$). We can solve it using OLS.

### 17. How do you interpret the coefficients of a Linear Regression model?
**Answer:**
*   $\beta_j$: The change in Y for a 1-unit increase in $X_j$, **holding all other variables constant**.

### 18. What is the Bias-Variance Tradeoff in the context of Linear Regression?
**Answer:**
*   **Simple Linear**: High Bias (Underfitting), Low Variance.
*   **High-degree Polynomial**: Low Bias, High Variance (Overfitting).
*   Regularization optimizes this tradeoff.

### 19. How do you handle categorical variables in Linear Regression?
**Answer:**
*   **One-Hot Encoding** (Dummy variables).
*   Must drop one column (Dummy Variable Trap) to avoid perfect multicollinearity if using OLS.

### 20. What metrics would you use to evaluate a regression model?
**Answer:**
*   **RMSE**: Root Mean Squared Error. Interpretable (same units as Y). Penalizes large errors.
*   **MAE**: Mean Absolute Error. Robust to outliers.
*   **R-squared**: Explained variance.
