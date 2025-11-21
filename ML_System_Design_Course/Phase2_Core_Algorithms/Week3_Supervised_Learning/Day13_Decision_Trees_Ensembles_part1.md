# Day 13 (Part 1): Advanced Trees & Ensembles

> **Phase**: 6 - Deep Dive
> **Topic**: Speed & Scale
> **Focus**: Histogram Splitting, Missing Values, and Isolation
> **Reading Time**: 60 mins

---

## 1. Histogram-Based Splitting (LightGBM/XGBoost)

Standard decision trees sort data ($O(N \log N)$) at every node. Too slow for Big Data.

### 1.1 The Algorithm
1.  **Binning**: Discretize continuous features into 255 bins (int8).
2.  **Histogram**: Build histograms for each feature ($O(N)$).
3.  **Split**: Find best split by scanning histogram (255 steps). Constant time w.r.t N!
4.  **Subtraction Trick**: Parent Hist - Left Child Hist = Right Child Hist. Speedup 2x.

---

## 2. Handling Missing Values

### 2.1 XGBoost / LightGBM
*   **Learned Direction**: The model learns a "Default Direction" (Left or Right) for NaNs at each node.
*   **Benefit**: No need for imputation. The missingness itself becomes information.

### 2.2 Random Forest (Sklearn)
*   Does not handle NaNs natively (historically). Requires imputation.

---

## 3. Isolation Forests (Anomaly Detection)

*   **Idea**: Anomalies are "few and different". They are easier to isolate.
*   **Algo**: Randomly split features.
*   **Metric**: Path Length. Anomalies are isolated near the root (short path). Normal points are deep (long path).
*   **No Labels Needed**.

---

## 4. Tricky Interview Questions

### Q1: Why does Random Forest not overfit as you add more trees?
> **Answer**:
> *   It averages the predictions. $\text{Var}(\text{Mean}) = \text{Var}(X) / N$.
> *   Adding trees reduces Variance. It does not affect Bias.
> *   The error converges to a limit. It doesn't go back up (unlike Boosting).

### Q2: Explain "Shrinkage" (Learning Rate) in Boosting.
> **Answer**:
> *   $F_{new}(x) = F_{old}(x) + \eta \cdot \text{Tree}(x)$.
> *   $\eta < 1$ (e.g., 0.1).
> *   Prevents a single tree from correcting the error too aggressively (which causes overfitting). Requires more trees, but generalizes better.

### Q3: How does Feature Importance differ between Gini and Permutation?
> **Answer**:
> *   **Gini (MDI)**: Biased towards high-cardinality features (e.g., User ID). Calculated on Train data.
> *   **Permutation**: Model-agnostic. Shuffle feature, measure drop in OOB/Validation accuracy. Unbiased and robust.

---

## 5. Practical Edge Case: Categorical Features
*   **One-Hot**: Bad for trees (sparse, splits power diluted).
*   **Target Encoding**: Replace category with mean target. (Risk: Leakage).
*   **CatBoost**: Uses "Ordered Target Encoding" to prevent leakage. Best for categorical data.

