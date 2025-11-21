# Day 13: Decision Trees & Ensembles - Interview Questions

> **Topic**: Tree-Based Models
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. How does a Decision Tree determine where to split?
**Answer:**
*   Greedy approach.
*   Iterates through every feature and every possible threshold.
*   Calculates a metric (Gini/Entropy) for the split.
*   Chooses the split that maximizes **Information Gain** (reduction in impurity).

### 2. Explain Entropy and Information Gain.
**Answer:**
*   **Entropy**: Randomness. $- \sum p \log p$.
*   **Info Gain**: $Entropy(Parent) - WeightedAvg(Entropy(Children))$.
*   We want children to be pure (Low Entropy).

### 3. Explain Gini Impurity.
**Answer:**
*   Probability of misclassifying a randomly chosen element if labeled randomly according to distribution.
*   $1 - \sum p_i^2$.
*   Faster to compute than Entropy (no log). Used by CART (sklearn).

### 4. What is Pruning? Why is it necessary?
**Answer:**
*   Removing branches to reduce complexity.
*   **Necessary**: Trees are prone to **Overfitting**. They can memorize noise (grow until 1 sample per leaf). Pruning improves generalization.

### 5. What is the difference between Pre-pruning and Post-pruning?
**Answer:**
*   **Pre-pruning**: Stop growing early. (Max Depth, Min Samples Leaf). Fast but might miss good splits later.
*   **Post-pruning**: Grow full tree, then cut branches that don't help validation error (Cost Complexity Pruning). Better results.

### 6. How do Decision Trees handle continuous variables?
**Answer:**
*   Sort values.
*   Consider midpoints between adjacent values as candidate thresholds.
*   Pick best threshold.

### 7. How do Decision Trees handle categorical variables?
**Answer:**
*   **One-Hot**: Treat as binary splits.
*   **Set-based**: (Not in sklearn) Split into subsets $\{A, B\}$ vs $\{C\}$.

### 8. What is Bagging (Bootstrap Aggregating)?
**Answer:**
*   Train N models independently on N **Bootstrap samples** (random sampling with replacement).
*   Average predictions (Regression) or Vote (Classification).
*   Reduces **Variance**.

### 9. Explain Random Forest. How does it differ from a single Decision Tree?
**Answer:**
*   Ensemble of Bagged Trees.
*   **Key Difference**: At each split, it considers only a **random subset of features** ($\sqrt{p}$).
*   Decorrelates the trees, making the ensemble stronger.

### 10. What is "Out-of-Bag" (OOB) Error?
**Answer:**
*   In Bagging, ~33% of data is left out of each bootstrap sample.
*   We can test the model on this unseen data.
*   Acts like a free Validation Set.

### 11. Explain Boosting.
**Answer:**
*   Sequential ensemble.
*   Train model 1. Calculate errors.
*   Train model 2 to fix errors of model 1.
*   Reduces **Bias**.

### 12. What is the difference between Bagging and Boosting?
**Answer:**
*   **Bagging**: Parallel. Reduces Variance. (Random Forest).
*   **Boosting**: Sequential. Reduces Bias. (XGBoost).

### 13. Explain AdaBoost.
**Answer:**
*   Adaptive Boosting.
*   Assigns **weights** to data points. Misclassified points get higher weight.
*   Next stump focuses on hard points.
*   Final prediction is weighted vote based on stump accuracy.

### 14. Explain Gradient Boosting (GBM).
**Answer:**
*   Generalization of boosting to arbitrary loss functions.
*   Subsequent models predict the **Negative Gradient** (Pseudo-residuals) of the loss function.
*   $y_{pred} = y_{pred} + \eta \cdot Model_2(Residuals)$.

### 15. What is XGBoost? Why is it so popular?
**Answer:**
*   Optimized GBM.
*   **Pros**: Regularization (L1/L2), Parallel processing (Block structure), Tree pruning, Handling missing values.
*   Dominates tabular competitions.

### 16. What is the difference between Random Forest and Gradient Boosting?
**Answer:**
*   **RF**: Independent trees. Deep trees. Hard to overfit. Parallel.
*   **GBM**: Dependent trees. Shallow trees (stumps). Easy to overfit. Sequential.

### 17. How does Random Forest handle missing values?
**Answer:**
*   Sklearn: Drops them (requires imputation).
*   Original RF: Uses surrogate splits (correlated features) or fills with median.

### 18. What is Feature Importance in Tree-based models?
**Answer:**
*   Calculated by summing the **Information Gain** contributed by each feature across all splits in all trees.
*   Biased towards high-cardinality features.

### 19. Do Decision Trees require Feature Scaling?
**Answer:**
*   **No**.
*   Splits are based on ordering ($x > 5$). Scaling ($x/10 > 0.5$) preserves order.
*   Monotonic transformations don't affect trees.

### 20. What are the advantages and disadvantages of Decision Trees?
**Answer:**
*   **Pros**: Interpretable, Handles non-linear, No scaling needed.
*   **Cons**: Overfitting (High variance), Unstable (small data change -> different tree), Orthogonal splits only.
