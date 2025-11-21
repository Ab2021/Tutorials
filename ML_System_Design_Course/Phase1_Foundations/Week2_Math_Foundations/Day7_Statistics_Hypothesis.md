# Day 7: Statistics & Hypothesis Testing

> **Phase**: 1 - Foundations
> **Week**: 2 - Mathematical Foundations
> **Focus**: Statistical Inference & Experimentation
> **Reading Time**: 60 mins

---

## 1. Distributions in the Wild

Understanding the shape of data is the first step in modeling it.

### 1.1 The Normal (Gaussian) Distribution
The "Bell Curve."
*   **Importance**: Many ML algorithms (Linear Regression, Gaussian Naive Bayes) assume data or errors are normally distributed.
*   **Central Limit Theorem (CLT)**: States that the sum (or mean) of many independent random variables tends toward a normal distribution, regardless of the original distribution. This justifies using parametric tests (like t-tests) even on non-normal data, provided the sample size is large enough.

### 1.2 The Power Law (Long Tail)
Real-world data often defies the Bell Curve.
*   **Examples**: Wealth distribution, word frequency (Zipf's Law), user clicks, website traffic.
*   **Implication**: In a normal distribution, outliers are rare. In a power law, outliers are common and massive. Using "Mean" and "Variance" on power-law data is misleading. A single "whale" user can skew the average revenue.
*   **ML Fix**: Log-transform the data to make it look more normal before feeding it to a model.

---

## 2. A/B Testing: The Gold Standard

In ML Systems, offline metrics (Accuracy, AUC) are proxies. The truth is online metrics (Click-Through Rate, Revenue). A/B testing bridges this gap.

### 2.1 The Setup
*   **Null Hypothesis ($H_0$)**: The new model B is no better than the old model A.
*   **Alternative Hypothesis ($H_1$)**: The new model B is different/better.
*   **p-value**: The probability of observing the result (or more extreme) assuming $H_0$ is true. If $p < 0.05$, we reject $H_0$.

### 2.2 Statistical Power
The probability of correctly detecting an effect when one exists.
*   **Underpowered Tests**: If your sample size is too small, you might fail to detect a real improvement (Type II Error). You need to calculate the required sample size *before* starting the test.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The Peeking Problem
**Scenario**: You run an A/B test. You check the dashboard every hour. On Hour 4, p-value drops to 0.04. You stop the test and declare victory.
**Theory**: By checking repeatedly, you are performing multiple comparisons. The probability of finding a "significant" result by random chance increases dramatically with every peek.
**Solution**:
1.  **Fixed Horizon**: Decide sample size in advance (e.g., 10,000 users) and DO NOT look until the end.
2.  **Sequential Testing**: Use specialized statistical methods (SPRT) that allow for early stopping with valid p-values.

### Challenge 2: Simpson's Paradox
**Scenario**: Model B looks better than Model A overall. But when you split by country, Model A is better in *every single country*.
**Theory**: A confounding variable (e.g., Country) influences both the treatment assignment and the outcome. Maybe Model B was shown mostly to US users who click more in general.
**Solution**: Randomized Controlled Trials (RCT) ensure balanced assignment. Stratified analysis helps detect these paradoxes.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the difference between Type I and Type II errors?**
> **Answer**:
> *   **Type I Error (False Positive)**: Rejecting the Null Hypothesis when it is actually true. (Convicting an innocent person). Controlled by significance level $\alpha$ (usually 0.05).
> *   **Type II Error (False Negative)**: Failing to reject the Null Hypothesis when it is actually false. (Letting a guilty person go free). Related to Statistical Power ($1 - \beta$).

**Q2: When should you use the Median instead of the Mean?**
> **Answer**: Use the Median when the distribution is skewed or contains significant outliers (e.g., Income, House Prices). The Mean is sensitive to outliers (non-robust), while the Median is robust. In ML preprocessing, we often use Median Imputation for missing values in skewed features.

**Q3: Explain the concept of "Statistical Significance" vs "Practical Significance".**
> **Answer**:
> *   **Statistical Significance**: We are confident the effect is not due to noise (p < 0.05). With huge datasets (common in tech), even a tiny difference (0.001% lift) can be statistically significant.
> *   **Practical Significance**: Is the effect large enough to matter? A 0.001% revenue increase might not justify the cost of deploying a complex new model. Business decisions depend on practical significance (ROI).

---

## 5. Further Reading
- [A/B Testing at Scale (Netflix Tech Blog)](https://netflixtechblog.com/)
- [Simpson's Paradox Interactive](https://seeing-theory.brown.edu/)
