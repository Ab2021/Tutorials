# Random Forests & Bagging (Part 1) - Theoretical Deep Dive

## Overview
A single decision tree is unstable. **Random Forests** solve this by training hundreds of trees and averaging their predictions. This session covers **Bagging (Bootstrap Aggregating)**, the **Random Forest Algorithm**, and the concept of **Out-of-Bag (OOB) Error**.

---

## 1. Conceptual Foundation

### 1.1 The Wisdom of Crowds

*   **Condorcet's Jury Theorem:** If each juror has a >50% chance of being right, the probability that the *majority* is right approaches 100% as the number of jurors increases.
*   **Ensemble Learning:** Combining multiple "Weak Learners" (Trees) to create a "Strong Learner".
*   **Condition:** The learners must be **diverse** (uncorrelated). If all trees make the same mistake, averaging them doesn't help.

### 1.2 Bagging (Bootstrap Aggregating)

*   **Bootstrap:** Sampling *with replacement* from the training data.
    *   If we have $N$ claims, we create a new dataset of size $N$ by picking claims at random. Some claims appear twice, some zero times.
*   **Aggregating:** Train a full, unpruned tree on each bootstrap sample.
*   **Prediction:** Average the predictions (Regression) or take the majority vote (Classification).
*   *Result:* Reduces Variance without increasing Bias.

### 1.3 Random Forest: Bagging + Feature Randomness

*   Bagging alone isn't enough. If there is one very strong predictor (e.g., "Drunk Driving"), *every* tree will split on it first. The trees will be highly correlated.
*   **The Random Forest Twist:** At each split, the tree can only choose from a **random subset of features** (e.g., $\sqrt{p}$ features).
*   *Effect:* Forces trees to use other variables. Decorrelates the trees. Increases diversity.

---

## 2. Mathematical Framework

### 2.1 The Variance Reduction

*   Variance of an average of $B$ i.i.d. variables with variance $\sigma^2$:
    $$ Var(\bar{X}) = \frac{\sigma^2}{B} $$
*   Variance of an average of $B$ *correlated* variables (correlation $\rho$):
    $$ Var(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2 $$
*   **Insight:** As $B \to \infty$, the second term vanishes, but the first term ($\rho \sigma^2$) remains.
*   **Random Forest Goal:** Reduce $\rho$ (correlation between trees) without increasing $\sigma^2$ (variance of individual trees) too much.

### 2.2 Out-of-Bag (OOB) Error

*   In Bootstrapping, about 36.8% of data points are left out of each sample ($1 - (1 - 1/N)^N \approx 1 - 1/e \approx 0.632$).
*   **OOB Method:**
    1.  For each claim $i$, find the trees that *did not* see this claim during training.
    2.  Average the predictions of those trees.
    3.  Calculate the error (OOB Error).
*   *Benefit:* It's a free validation set! No need to split data into Train/Test.

---

## 3. Theoretical Properties

### 3.1 Feature Importance (MDI)

*   **Mean Decrease Impurity (MDI):**
    *   Every time a tree splits on "Age", the impurity decreases.
    *   Sum these decreases across all trees.
    *   Normalize so they sum to 1.
*   *Result:* A ranking of which variables are most useful for prediction.

### 3.2 Robustness to Noise

*   Random Forests are very hard to overfit.
*   Adding more trees does *not* cause overfitting (it just stabilizes the average).
*   They handle outliers well (because outliers are likely to be left out of many bootstrap samples).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Random Forest in Python

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Simulate Data
X = pd.DataFrame(np.random.rand(1000, 10), columns=[f'Feat_{i}' for i in range(10)])
y = 2 * X['Feat_0'] + 0.5 * X['Feat_1']**2 + np.random.normal(0, 0.1, 1000)

# Fit Random Forest
# n_estimators: Number of trees
# max_features: 'sqrt' is standard for classification, '1.0' (all) or 'log2' for regression
rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', oob_score=True, random_state=42)
rf.fit(X, y)

# OOB Score (R^2)
print(f"OOB Score: {rf.oob_score_:.4f}")

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
```

### 4.2 The "Number of Trees" Convergence

```python
# Check how OOB error stabilizes as we add trees
n_trees = [10, 50, 100, 200, 500, 1000]
oob_scores = []

for n in n_trees:
    rf_temp = RandomForestRegressor(n_estimators=n, max_features='sqrt', oob_score=True, n_jobs=-1, random_state=42)
    rf_temp.fit(X, y)
    oob_scores.append(rf_temp.oob_score_)

plt.plot(n_trees, oob_scores, marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("OOB Score")
plt.title("Convergence of Random Forest")
plt.show()
```

---

## 5. Evaluation & Validation

### 5.1 OOB vs. Cross-Validation

*   **OOB:** Fast. Unbiased for large $N$.
*   **CV:** Better if you have time.
*   *Rule of Thumb:* For massive datasets, use OOB. For small datasets, use CV.

### 5.2 Bias in Feature Importance

*   **Warning:** MDI Feature Importance is biased towards high-cardinality features (e.g., "Policy ID" or "Zip Code").
*   *Why?* Because they offer many split points, so they get picked often by chance.
*   *Fix:* Use **Permutation Importance** (Day 92).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Extrapolation**
    *   Random Forests are just averages of Trees. Trees cannot extrapolate.
    *   Therefore, Random Forests cannot extrapolate.
    *   *Impact:* If inflation pushes claims higher than ever seen before, RF will predict the historical max, not the new trend.

2.  **Trap: Correlated Features**
    *   If "Vehicle Value" and "Sum Insured" are 99% correlated, RF will split the importance between them.
    *   Neither will look "Super Important", but together they are.

### 6.2 Implementation Challenges

1.  **Model Size:**
    *   A forest with 1000 deep trees can be 500MB+.
    *   *Deployment:* Slow to predict in real-time.
    *   *Fix:* Limit `max_depth` or use `min_samples_leaf` to prune the trees inside the forest.

---

## 7. Advanced Topics & Extensions

### 7.1 Extremely Randomized Trees (ExtraTrees)

*   **Random Forest:** Finds the *best* split among the random subset of features.
*   **ExtraTrees:** Picks a *random* split point for each feature, then chooses the best among those.
*   *Result:* More randomness, lower variance, faster training.

### 7.2 Quantile Regression Forests

*   Standard RF predicts the Mean.
*   **Quantile RF:** Stores all the values in the leaf nodes. Can predict the Median, 90th Percentile, etc.
*   *Use:* Predicting "Worst Case" claims (VaR).

---

## 8. Regulatory & Governance Considerations

### 8.1 Reproducibility

*   RF is stochastic.
*   You **must** set the `random_state` (seed).
*   Otherwise, you get a different price every time you run the script.

---

## 9. Practical Example

### 9.1 Worked Example: The "Non-Linear" Age Curve

**Scenario:**
*   Age vs. Risk is U-shaped (High for young, low for middle, high for old).
*   **GLM:** Needs a polynomial term ($Age^2$) or categorical binning.
*   **Random Forest:** Fits it perfectly without any feature engineering.
*   **Result:** RF outperforms GLM on raw data.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Bagging** reduces variance.
2.  **Feature Randomness** decorrelates trees.
3.  **OOB Error** is a built-in validation metric.

### 10.2 When to Use This Knowledge
*   **Benchmarking:** RF is the best "out of the box" model. Use it to set a baseline.
*   **Feature Selection:** Use Feature Importance to find variables for your GLM.

### 10.3 Critical Success Factors
1.  **Number of Trees:** More is better (until diminishing returns).
2.  **Max Features:** The most important hyperparameter to tune.

### 10.4 Further Reading
*   **Breiman (2001):** "Random Forests".
*   **Strobl et al.:** "Bias in random forest variable importance measures".

---

## Appendix

### A. Glossary
*   **Ensemble:** A group of models.
*   **Bootstrap:** Sampling with replacement.
*   **OOB:** Out-of-Bag.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Bagging Variance** | $\rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$ | Theory |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
