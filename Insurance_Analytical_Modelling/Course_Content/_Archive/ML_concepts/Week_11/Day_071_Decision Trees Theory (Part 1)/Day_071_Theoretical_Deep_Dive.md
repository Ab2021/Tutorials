# Decision Trees Theory (Part 1) - Theoretical Deep Dive

## Overview
Phase 3 marks the transition from "Classical Statistics" (GLMs) to "Machine Learning". We start with **Decision Trees**, the building blocks of modern actuarial data science. This session covers the **CART Algorithm**, **Splitting Criteria**, and why trees are excellent at capturing **Interactions**.

---

## 1. Conceptual Foundation

### 1.1 From Linear to Non-Linear

*   **GLM:** Assumes a linear relationship (or log-linear) between predictors and the response. $Y = \exp(\beta_0 + \beta_1 X_1 + ...)$.
*   **Decision Tree:** Makes no assumption about linearity. It segments the data into "Rectangles" (Nodes) and fits a simple constant in each node.
*   **Analogy:**
    *   GLM is like drawing a smooth curve through the data.
    *   Decision Tree is like playing "20 Questions" to narrow down the answer.

### 1.2 The CART Algorithm

*   **CART:** Classification and Regression Trees.
*   **Mechanism:** Recursive Partitioning.
    1.  Start with the whole dataset (Root Node).
    2.  Find the *single best split* (e.g., Age < 25) that separates the data into two purer groups.
    3.  Repeat the process for each child node.
    4.  Stop when a criterion is met (e.g., Node size < 50).

### 1.3 Why Actuaries Like Trees

1.  **Interactions:** Trees automatically find interactions. If the first split is "Age < 25" and the next split *under* that node is "Car = Sports", the tree has found the "Young Driver + Sports Car" interaction without you specifying it.
2.  **Missing Values:** Trees can handle missing data (by treating "Missing" as a category or using surrogate splits).
3.  **Interpretability:** You can draw the tree and show it to a regulator.

---

## 2. Mathematical Framework

### 2.1 Splitting Criteria (Regression)

For pricing (Frequency/Severity), we use Regression Trees.
*   **Goal:** Minimize the variance within each node.
*   **Metric:** Sum of Squared Errors (SSE) or Poisson Deviance.
    $$ SSE = \sum_{i \in Node} (y_i - \bar{y})^2 $$
*   **Split Rule:** Choose split $s$ to maximize the reduction in SSE:
    $$ \Delta = SSE_{parent} - (SSE_{left} + SSE_{right}) $$

### 2.2 Splitting Criteria (Classification)

For fraud detection or lapse prediction (0/1), we use Classification Trees.
*   **Gini Impurity:** Measures the probability of misclassification.
    $$ Gini = 1 - \sum_{k} p_k^2 $$
    *   $p_k$: Proportion of class $k$ in the node.
    *   *Pure Node:* Gini = 0.
    *   *Mixed Node (50/50):* Gini = 0.5.
*   **Entropy (Information Gain):**
    $$ Entropy = - \sum_{k} p_k \log_2(p_k) $$
    *   Similar to Gini but computationally more expensive (logs).

### 2.3 Stopping Rules

To prevent the tree from memorizing the data (Overfitting), we need constraints:
1.  **Max Depth:** Limit the number of levels (e.g., 5).
2.  **Min Samples Split:** Don't split a node if it has fewer than 100 claims.
3.  **Min Impurity Decrease:** Don't split if the gain is negligible.

---

## 3. Theoretical Properties

### 3.1 The Interaction Effect

*   In a GLM, to model "Young Driver on a Sports Car", you must explicitly add an interaction term: $\beta_{Age} + \beta_{Car} + \beta_{Age \times Car}$.
*   In a Tree, this happens naturally.
    *   Split 1: Age < 25? (Yes).
    *   Split 2 (Given Yes): Car = Sports? (Yes).
    *   Leaf Node: High Risk.
*   *Benefit:* You don't need to know *which* interactions exist beforehand. The tree finds them.

### 3.2 Instability

*   **The Flaw:** Trees are high variance.
*   A small change in the training data (e.g., removing one claim) can result in a completely different tree structure.
*   *Why?* Because the splits are greedy. If the first split changes, the entire structure below it changes.
*   *Solution:* Ensembling (Random Forests, GBMs) - covered in later sessions.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Regression Tree in Python (sklearn)

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Simulate Insurance Data
np.random.seed(42)
n = 1000
age = np.random.randint(18, 70, n)
car_power = np.random.randint(50, 300, n)
# Interaction: Young + High Power = High Frequency
freq = 0.1 + 0.2 * (age < 25) + 0.001 * car_power + 0.5 * ((age < 25) & (car_power > 200))
y = np.random.poisson(freq)

X = pd.DataFrame({'Age': age, 'Power': car_power})

# Fit Tree
# max_depth=3 for interpretability
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50)
tree.fit(X, y)

# Visualization
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, filled=True, precision=3)
plt.title("Claim Frequency Decision Tree")
plt.show()

# Interpretation
# Look at the first split. Is it Age or Power?
# Look at the leaf nodes. Which one has the highest 'value' (predicted frequency)?
```

### 4.2 Gini Calculation Script

```python
def calculate_gini(labels):
    total = len(labels)
    if total == 0: return 0
    counts = np.unique(labels, return_counts=True)[1]
    proportions = counts / total
    gini = 1 - np.sum(proportions**2)
    return gini

# Example
group_a = [0, 0, 0, 0, 0] # Pure
group_b = [0, 0, 1, 1, 0] # Mixed
group_c = [0, 1, 0, 1, 0, 1] # Very Mixed

print(f"Gini A: {calculate_gini(group_a):.2f}")
print(f"Gini B: {calculate_gini(group_b):.2f}")
print(f"Gini C: {calculate_gini(group_c):.2f}")
```

---

## 5. Evaluation & Validation

### 5.1 Visual Inspection

*   **Sanity Check:** Does the tree make sense?
*   If the first split is "PolicyID < 5000", the model is overfitting to the ID.
*   If the split "Age < 25" leads to *lower* risk, check the data (maybe young drivers have parents on the policy).

### 5.2 Variable Importance

*   Trees provide a "Feature Importance" score.
*   It measures how much each feature contributed to decreasing the impurity (weighted by node size).
*   *Use:* Identify the key drivers of risk.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Continuous Variables**
    *   Trees handle continuous variables by testing *every possible split point*.
    *   *Issue:* If a variable has many unique values (e.g., Exact Credit Score), the tree might bias towards splitting it just because it has more options.

2.  **Trap: Extrapolation**
    *   Trees cannot extrapolate.
    *   If the max Age in training is 70, and you feed it Age 80, it will predict the value of the "Age > 70" leaf. It won't project a trend.

### 6.2 Implementation Challenges

1.  **Categorical Variables with Many Levels:**
    *   "Zip Code" has 30,000 levels.
    *   Standard CART tries $2^{30000}$ splits. (Impossible).
    *   *Solution:* Target Encoding (replace Zip with Average Claim Cost of that Zip) before feeding to the tree.

---

## 7. Advanced Topics & Extensions

### 7.1 Poisson Trees

*   Standard `DecisionTreeRegressor` minimizes MSE (Gaussian assumption).
*   For Claim Counts, we want to minimize **Poisson Deviance**.
*   *Library:* `sklearn` doesn't support Poisson split criteria natively (historically). `H2O` or `XGBoost` do.
*   *Workaround:* Transform target $\log(y)$ or use specialized libraries.

### 7.2 Surrogate Splits

*   Used for missing data.
*   If "Credit Score" is missing, the tree looks for a "Surrogate" (e.g., Income) that mimics the split of Credit Score.

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Black Box" Debate

*   Single Trees are "White Box" (Transparent).
*   Ensembles (Forests) are "Black Box".
*   **Regulators:** Often accept Single Trees for segmentation but prefer GLMs for final rating factors.
*   **Hybrid Approach:** Use a Tree to find interactions, then hard-code those interactions into a GLM.

---

## 9. Practical Example

### 9.1 Worked Example: The "Segmentation" Project

**Scenario:**
*   You have a GLM for Home Insurance.
*   The Loss Ratio in "Coastal Regions" is terrible.
*   **Action:** Fit a Decision Tree on the residuals of the GLM ($Actual - Predicted$).
*   **Result:** The Tree splits on "Distance to Coast < 1 mile" AND "Roof Type = Shingle".
*   **Insight:** The GLM was missing the interaction between Coast and Roof. You add this interaction to the GLM.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Recursive Partitioning** splits data into pure nodes.
2.  **Gini/Entropy** are the metrics for purity.
3.  **Interactions** are the superpower of trees.

### 10.2 When to Use This Knowledge
*   **Exploratory Data Analysis (EDA):** Quickly understanding key drivers.
*   **Segmentation:** Creating tariff cells.

### 10.3 Critical Success Factors
1.  **Pruning:** Don't let the tree grow too deep.
2.  **Data Prep:** Handle high-cardinality categoricals (Zip Codes).
3.  **Validation:** Check stability on a test set.

### 10.4 Further Reading
*   **Breiman et al.:** "Classification and Regression Trees" (The Bible of CART).
*   **James et al.:** "An Introduction to Statistical Learning" (Chapter 8).

---

## Appendix

### A. Glossary
*   **Root Node:** The top of the tree.
*   **Leaf Node:** The bottom nodes (final prediction).
*   **Impurity:** How mixed the data is.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Gini** | $1 - \sum p_k^2$ | Classification Split |
| **SSE** | $\sum (y - \bar{y})^2$ | Regression Split |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
