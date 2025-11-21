# Day 13: Decision Trees & Ensembles

> **Phase**: 2 - Core Algorithms
> **Week**: 3 - Supervised Learning
> **Focus**: Non-Linear Models & Boosting
> **Reading Time**: 60 mins

---

## 1. Decision Trees: The Logic of ML

Trees learn a hierarchy of if/else questions. They are non-parametric and can model complex non-linear boundaries.

### 1.1 Splitting Criteria
How does the tree decide "Is Age > 25?" is the best question?
*   **Entropy (Information Gain)**: Measures reduction in surprise.
*   **Gini Impurity**: Measures probability of misclassifying a randomly chosen element. (Faster to compute than Entropy).
*   **MSE (Regression)**: Measures variance reduction.

### 1.2 The Overfitting Problem
A tree can grow until every single leaf has 1 sample. This is 100% training accuracy but terrible generalization.
*   **Pruning**: Cutting back branches that don't add significant power.
*   **Max Depth**: Limiting the tree height.

---

## 2. Ensemble Methods: Strength in Numbers

"The wisdom of crowds." Combining many weak learners to create a strong learner.

### 2.1 Bagging (Bootstrap Aggregating) -> Random Forest
*   **Idea**: Train $N$ trees independently on random subsets of the data (Bootstrap). Average their predictions.
*   **Why it works**: Averaging reduces **Variance**. It smooths out the overfitting of individual trees.
*   **Feature Randomness**: Random Forest also selects a random subset of *features* at each split, decorrelating the trees further.

### 2.2 Boosting -> XGBoost / LightGBM / CatBoost
*   **Idea**: Train trees **sequentially**. Each tree tries to correct the errors of the previous tree.
*   **Gradient Boosting**: Each tree predicts the *negative gradient* (residual errors) of the loss function.
*   **Why it works**: Reduces **Bias**. It turns weak learners (high bias) into a complex model.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Categorical Features
**Scenario**: Feature "City" has 10,000 unique values. One-Hot Encoding creates 10,000 sparse columns. Trees struggle with sparse data (splits become weak).
**Solution**:
*   **Target Encoding**: Replace "Paris" with the average target value for Paris (e.g., 0.85).
*   **Native Handling**: LightGBM/CatBoost handle categories natively without One-Hot Encoding (using histograms or ordered target statistics).

### Challenge 2: Inference Latency
**Scenario**: A Random Forest with 1000 deep trees is slow to predict (must traverse 1000 paths).
**Solution**:
*   **Distillation**: Train a simpler model (e.g., a single shallow tree or neural net) to mimic the predictions of the ensemble.
*   **Treelite**: Compile trees into optimized C code.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Explain the difference between Bagging and Boosting.**
> **Answer**:
> *   **Bagging (Random Forest)**: Parallel training. Independent trees. Reduces Variance (Overfitting). Good for complex base learners.
> *   **Boosting (XGBoost)**: Sequential training. Dependent trees. Reduces Bias (Underfitting). Good for simple base learners (stumps).

**Q2: Why does Random Forest not overfit as you add more trees?**
> **Answer**: Adding more trees averages out the noise. The generalization error converges to a limit. It doesn't increase. However, the model becomes slower and heavier. (Note: Boosting *can* overfit if you add too many trees).

**Q3: How does XGBoost handle missing values?**
> **Answer**: XGBoost learns a "default direction" for missing values at each split. During training, it tries sending missing values left and right, calculates the gain, and chooses the path that minimizes loss.

---

## 5. Further Reading
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [A Gentle Introduction to Gradient Boosting](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
