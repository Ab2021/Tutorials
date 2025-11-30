# Decision Trees Theory (Part 2) - Theoretical Deep Dive

## Overview
A fully grown decision tree is a "High Variance" model. It memorizes the noise in the training data (Overfitting). To make it useful for insurance pricing, we must **Prune** it. This session covers **Pre-Pruning**, **Post-Pruning (CCP)**, and **Hyperparameter Tuning**.

---

## 1. Conceptual Foundation

### 1.1 The Overfitting Problem

*   **Scenario:** You fit a tree to 10,000 claims.
*   **Result:** The tree grows until every single claim is in its own leaf node.
*   **Training Error:** 0.
*   **Test Error:** Huge.
*   **Why?** The tree learned that "Policy 123 had a claim". This is not a pattern; it's an anecdote.

### 1.2 Pre-Pruning (Early Stopping)

*   Stop the tree *while* it is growing.
*   **Constraints:**
    1.  **Max Depth:** Stop at depth 5.
    2.  **Min Samples Leaf:** A leaf must have at least 100 claims.
    3.  **Min Impurity Decrease:** Only split if Gini improves by 0.01.
*   *Pros:* Fast.
*   *Cons:* "Horizon Effect". A split might look useless now (Gain = 0.001) but lead to a massive gain in the next level. Pre-pruning misses this.

### 1.3 Post-Pruning (Cost Complexity Pruning)

*   Grow the tree to the max (Overfit).
*   Then, cut back the branches that don't "pay their rent".
*   **Metric:** Cost-Complexity Score.
    $$ R_\alpha(T) = R(T) + \alpha |T| $$
    *   $R(T)$: Total Error (Misclassification or SSE).
    *   $|T|$: Number of terminal nodes (Leaves).
    *   $\alpha$: The penalty for complexity.

---

## 2. Mathematical Framework

### 2.1 The Alpha Parameter ($\alpha$)

*   **$\alpha = 0$:** No penalty. The tree remains fully grown (Overfit).
*   **$\alpha = \infty$:** Infinite penalty. The tree is pruned to the Root Node (Underfit).
*   **Goal:** Find the $\alpha$ that minimizes Cross-Validation Error.

### 2.2 Weakest Link Pruning

1.  Start with the full tree $T_0$.
2.  Find the subtree $t$ that, if removed, causes the *smallest* increase in error per leaf removed.
3.  Prune it to get $T_1$.
4.  Repeat until you reach the root.
5.  This gives a sequence of trees $T_0, T_1, ..., T_{root}$ and a sequence of alphas.

---

## 3. Theoretical Properties

### 3.1 Bias-Variance Tradeoff

*   **Full Tree:** Low Bias, High Variance. (Captures every nuance, but unstable).
*   **Pruned Tree:** Higher Bias, Lower Variance. (Smoother, more stable).
*   **Sweet Spot:** The $\alpha$ that balances these two.

### 3.2 Stability

*   Even with pruning, single trees are unstable.
*   If you change the training data slightly, the "Best Split" at the root might change from "Age" to "Car Type".
*   This changes the *entire* structure.
*   *Actuarial Impact:* Hard to explain to a regulator why the model changed completely next year.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Cost Complexity Pruning in Python

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. Fit the Full Tree
clf = DecisionTreeRegressor(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 2. Train a tree for every alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 3. Plot Training vs Test Score
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
plt.xlabel("alpha")
plt.ylabel("R^2 Score")
plt.title("Accuracy vs Alpha for training and testing sets")
plt.legend()
plt.show()

# 4. Pick the Best Alpha
# Look for the peak in the 'test' curve.
best_alpha = ccp_alphas[np.argmax(test_scores)]
print(f"Optimal Alpha: {best_alpha}")
```

### 4.2 Grid Search for Pre-Pruning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [50, 100, 200, 500]
}

grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best Params: {grid.best_params_}")
```

---

## 5. Evaluation & Validation

### 5.1 The "Eye Test"

*   Plot the pruned tree.
*   Does it make sense?
*   Example:
    *   Node 1: Age < 25.
    *   Node 2: Car Power > 200.
    *   Leaf: Frequency = 0.25 (vs Base 0.10).
    *   *Verdict:* Sensible. High risk for young drivers in fast cars.

### 5.2 Stability Test

*   Run the model on 5 different folds of data.
*   Does the "Root Split" remain the same?
*   If not, the model is not robust enough for rate filing.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Using Default Parameters**
    *   `sklearn` defaults to `max_depth=None`. This guarantees overfitting.
    *   *Always* set constraints or use CCP.

2.  **Trap: One-Hot Encoding**
    *   If you One-Hot Encode "State" (50 columns), the tree has to split 50 times to capture the full effect.
    *   *Better:* Use Target Encoding or software that handles categoricals natively (H2O).

### 6.2 Implementation Challenges

1.  **Data Leakage:**
    *   If you Target Encode using the *whole* dataset, then split into Train/Test, you have leaked the Test target into the Train features.
    *   *Fix:* Fit the Target Encoder *only* on the Training set.

---

## 7. Advanced Topics & Extensions

### 7.1 Monotonic Constraints

*   **Problem:** The tree might find that Age 40 is riskier than Age 30 (noise), but Age 50 is safer.
*   **Constraint:** Force the prediction to be monotonic with respect to Age.
*   *Implementation:* `XGBoost` supports this. `sklearn` trees do not (easily).

### 7.2 Interaction Detection

*   Use the tree to *find* interactions, then feed them into a GLM.
*   **Hybrid Model:** $GLM = \beta_0 + \beta_1 Age + \beta_2 (Age < 25 \times Power > 200)$.
*   *Benefit:* You get the stability of a GLM with the non-linearity of a Tree.

---

## 8. Regulatory & Governance Considerations

### 8.1 Rate Filing

*   **Regulator Question:** "Why did the rate for 35-year-olds increase?"
*   **Tree Answer:** "Because they fell into Leaf Node #42."
*   **Better Answer:** "Because the data shows a 15% higher frequency for this segment, driven by an interaction with Vehicle Type."
*   *Tip:* Pruned trees are easier to defend than deep trees.

---

## 9. Practical Example

### 9.1 Worked Example: The "Lapse" Model

**Scenario:**
*   Predicting which Life Insurance customers will lapse (cancel).
*   **Full Tree:** 95% Training Accuracy, 60% Test Accuracy. (Overfit).
*   **Pruned Tree ($\alpha=0.01$):** 80% Training, 78% Test.
*   **Insight:** The pruned tree identified 3 key drivers:
    1.  Policy Duration (Spike at Year 10).
    2.  Premium Jump.
    3.  Age > 60.
*   **Action:** Marketing campaign targeting these specific segments.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Overfitting** is the enemy.
2.  **Pruning** (CCP) is the solution.
3.  **Alpha** controls the trade-off.

### 10.2 When to Use This Knowledge
*   **Model Building:** Every time you fit a tree.
*   **Interviews:** "How do you prevent a decision tree from overfitting?"

### 10.3 Critical Success Factors
1.  **Cross-Validation:** Never pick $\alpha$ based on Training error.
2.  **Business Logic:** If a split doesn't make sense, prune it manually.

### 10.4 Further Reading
*   **Hastie et al.:** "The Elements of Statistical Learning".
*   **scikit-learn documentation:** "Cost Complexity Pruning".

---

## Appendix

### A. Glossary
*   **CCP:** Cost Complexity Pruning.
*   **Alpha:** Complexity parameter.
*   **Subtree:** A branch of the main tree.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Cost Complexity** | $R(T) + \alpha |T|$ | Pruning Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
