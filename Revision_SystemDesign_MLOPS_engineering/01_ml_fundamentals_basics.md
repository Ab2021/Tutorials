# ML FUNDAMENTALS DEEP DIVE — Interview Mastery
> Root cause: You fail when interviewers drill down on basics. This file fixes that.

---

## SECTION 1: DECISION TREES — THE ROOT OF ALL TREE MODELS

### How a Decision Tree Makes Splits

A decision tree finds the **best split** by testing every feature at every possible threshold and choosing the one that gives the greatest **impurity reduction** in the children.

**Two impurity measures:**

### Gini Impurity (default in sklearn CART)
```
Gini(S) = 1 - sum(p_c^2)  for all classes c
```
- Pure node (all same class): Gini = 1 - 1^2 = **0** (perfect)
- Perfectly mixed node (50-50): Gini = 1 - (0.5^2 + 0.5^2) = **0.5** (worst)

**Worked Example — Fraud Detection:**
- Parent node: 100 claims, 10 fraud (10%), 90 not-fraud (90%)
- Parent Gini = 1 - (0.1^2 + 0.9^2) = 1 - (0.01 + 0.81) = 0.18
- Split on "claim_amount > $50K": Left=60 claims (8 fraud), Right=40 claims (2 fraud)
- Left Gini = 1 - ((8/60)^2 + (52/60)^2) = 1 - (0.018 + 0.751) = 0.231
- Right Gini = 1 - ((2/40)^2 + (38/40)^2) = 1 - (0.0025 + 0.9025) = 0.095
- Weighted Gini = (60/100)*0.231 + (40/100)*0.095 = 0.1386 + 0.038 = **0.177**
- Gini Reduction = 0.18 - 0.177 = 0.003 (this split helps a little)

### Entropy / Information Gain (used in ID3, C4.5)
```
H(S) = -sum(p_c * log2(p_c))
Information Gain(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
```
- Pure node: H = 0 (no uncertainty)
- 50-50 split: H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0 bit (maximum uncertainty)

### When to use Gini vs Entropy
- **Gini:** Faster (no log computation), default choice, slightly favors larger partitions
- **Entropy:** Information-theoretic interpretation, better for multi-class, computationally more expensive
- **Practical rule:** Both give nearly identical trees. Use Gini (default) unless you have a specific reason.

### Decision Tree Hyperparameters for Overfitting Control
| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| max_depth | Max levels in tree | 3-10 |
| min_samples_leaf | Min samples per leaf | 5-50 |
| min_samples_split | Min samples to split node | 10-100 |
| max_features | Features to consider per split | sqrt(p) or 0.5-0.8 |
| ccp_alpha | Pruning complexity parameter | 0 (no pruning) to 0.05 |

**Interview answer for decision tree split:**
> "At each node, we evaluate every feature and every possible threshold. We choose the feature-threshold pair that maximizes information gain (or minimizes weighted Gini in children). We stop splitting when max_depth is reached or min_samples_leaf is hit."

---

## SECTION 2: THE ENSEMBLE HIERARCHY

| Property | Single DT | Random Forest | GBM | XGBoost | LightGBM |
|----------|-----------|---------------|-----|---------|----------|
| Method | None | Bagging | Boosting | Boosting | Boosting |
| Tree growth | Level-wise | Level-wise | Level-wise | Level-wise | Leaf-wise |
| Reduces | - | Variance | Bias | Bias+Variance | Bias+Variance |
| Interpretability | High | Medium | Low | Low | Low |
| Training speed | Fast | Medium | Slow | Medium | Fast |
| Inference speed | Fast | Medium | Fast | Fast | Very fast |
| When to use | Baseline, rules | Many features, parallel | Best accuracy | Best accuracy + regularization | Large data, fast |

---

## SECTION 3: BIAS-VARIANCE TRADEOFF

### Intuitive Definitions
- **Bias:** Error from WRONG ASSUMPTIONS. Model is too simple. "I always predict the mean." High bias = underfitting.
- **Variance:** Error from SENSITIVITY TO TRAINING DATA. Model memorizes noise. High variance = overfitting.
- **Noise:** Irreducible error. Cannot be fixed by better models.

```
Total Error = Bias^2 + Variance + Noise
```

### Diagnosis Using Training/Validation Curves
| Observation | Diagnosis | Fix |
|-------------|-----------|-----|
| Train error HIGH, Val error HIGH | High bias (underfitting) | More features, deeper model, less regularization |
| Train error LOW, Val error HIGH | High variance (overfitting) | More training data, regularization, simpler model |
| Both errors LOW | Good fit | Deploy! |
| Train error ≈ Val error ≈ medium | Both bias and variance | Better features, try different model family |

### Practical Fixes for Each Problem

**High Bias:**
- More complex model (single DT → Random Forest → XGBoost)
- More/better features (feature engineering)
- Reduce regularization (lower alpha/C)
- Longer training (more epochs/estimators)

**High Variance:**
- More training data
- Regularization (L1/L2, dropout for NNs)
- Reduce model complexity (max_depth, min_samples_leaf)
- Cross-validation instead of single train-test split
- Ensemble methods (bagging reduces variance)

**Interview answer:**
> "I diagnose bias-variance by plotting training vs. validation metrics across model complexity or training set size. If training error is high, I need a more complex model or better features. If validation error diverges from training error, I need regularization or more data."

---

## SECTION 4: EVALUATION METRICS — CRISP DEFINITIONS

### The Confusion Matrix
```
                  Predicted Positive    Predicted Negative
Actual Positive      TP (True Pos)        FN (False Neg)
Actual Negative      FP (False Pos)        TN (True Neg)
```

**In fraud context:**
- TP = We said fraud, it IS fraud (correct catch)
- FP = We said fraud, it's NOT fraud (false alarm — costs investigator time)
- TN = We said not fraud, it's NOT fraud (correct)
- FN = We said not fraud, it IS fraud (missed — costs money)

---

### PRECISION
```
Precision = TP / (TP + FP) = "Of all claims we flagged, what fraction was actually fraud?"
```
**When to optimize PRECISION:** Real-time fraud flagging. Every flag → manual review (analyst time). Low precision = genuine customers get harassed = churn + analyst overload.

**Interview answer:**
> "I optimize for precision in real-time scoring because every false positive triggers a costly manual investigation. If precision is low, legitimate claims get held up, customers get frustrated, and underwriter bandwidth is wasted."

---

### RECALL (Sensitivity)
```
Recall = TP / (TP + FN) = "Of all actual fraud, what fraction did we catch?"
```
**When to optimize RECALL:** Batch overnight scoring. Missing fraud is costlier than extra false positives. Cast a wide net; analysts triage.

**Interview answer:**
> "For batch overnight models, I optimize recall because the cost of a missed fraud (undetected loss) far exceeds the cost of investigating a false positive. With overnight batch, we have time for analysts to triage a larger set."

---

### F1 SCORE — THE HARMONIC MEAN
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**WHY harmonic mean, not arithmetic mean?**

Harmonic mean penalizes extreme imbalance between P and R.

**Worked example:**
- Case A: Precision=25%, Recall=75%
  - Arithmetic mean = (25+75)/2 = **50%**
  - F1 (harmonic) = 2*(25*75)/(25+75) = 2*1875/100 = **37.5%**
- Case B: Precision=50%, Recall=50%
  - Arithmetic mean = (50+50)/2 = **50%** (same!)
  - F1 (harmonic) = 2*(50*50)/(50+50) = 2*2500/100 = **50%**

**Case B has a higher F1 (50% vs 37.5%) with the same arithmetic mean.**

The harmonic mean ensures a model can only score well if BOTH metrics are good — it can't cheat by sacrificing one for the other.

---

### ROC-AUC vs PR-AUC

**ROC-AUC:**
- Plots True Positive Rate (Recall) vs False Positive Rate at every threshold
- AUC = area under curve = probability model ranks a random positive higher than a random negative
- **Problem:** Misleading for imbalanced datasets (1% fraud rate)
- A model that says "not fraud" for everything: FPR=0, TPR=0 → ROC-AUC still looks okay

**PR-AUC:**
- Plots Precision vs Recall at every threshold
- More informative for imbalanced data — baseline is the fraud rate, not 0.5
- A bad model with 1% fraud: PR-AUC ≈ 0.01 (baseline). Immediately shows model is useless.

**Rule:** Always use PR-AUC for fraud detection (1-5% positive rate). ROC-AUC for balanced datasets.

**Interview answer:**
> "For fraud with 1% positive rate, ROC-AUC is overly optimistic. A model predicting 'not fraud' always achieves ROC-AUC=0.5, but that misses the point. PR-AUC correctly reflects that a random baseline would have PR-AUC ≈ 0.01, so any meaningful model must significantly exceed that."

---

### LIFT CHART / DECILE CHART

**How to read it:**
1. Sort all predictions by fraud probability, highest first
2. Divide into 10 equal buckets (deciles)
3. Calculate fraud rate in each decile
4. Compare to overall fraud rate → this is the LIFT

**Example:**
- Overall fraud rate: 5%
- Top decile fraud rate: 40%
- Lift at decile 1: 40%/5% = **8x**
- Capture@20% (top 2 deciles): (number of fraud in top 2 deciles) / (total fraud)

**Business translation:**
> "Our model in the top 25% of risk scores captures 70% of all fraud. If investigators can only review 25% of claims, they'll catch 70% of fraud using our model vs. 25% randomly."

---

### BRIER SCORE / CALIBRATION

```
Brier Score = (1/N) * sum((predicted_probability - actual_outcome)^2)
```
- Lower is better. 0 = perfect. 0.25 = no-skill baseline.
- A model that always predicts 0.5 probability has Brier Score = 0.25.

**Calibration = are predicted probabilities trustworthy?**
- Calibrated model: when it says 30% probability → 30% of those ARE fraud
- Uncalibrated model: says 90% but only 40% are fraud → thresholds don't work
- Fix: Platt scaling (sigmoid on top of model outputs) or Isotonic regression

**Interview answer:**
> "Calibration matters for fraud because we use probability thresholds for risk tiering. If I say '70% probability of fraud' but only 30% of those are actually fraud, our escalation thresholds are wrong and we're over-flagging."

---

## SECTION 5: IMBALANCED DATA — FRAUD SPECIFIC

**Problem:** 1% fraud rate → model achieves 99% accuracy by predicting 'not fraud' for everything.

### Methods and When to Use

**1. scale_pos_weight (XGBoost) / class_weight (sklearn)**
```python
scale_pos_weight = n_negatives / n_positives  # e.g., 990/10 = 99
```
- Tells model to treat each fraud case as 99 not-fraud cases
- Fast, no data modification
- **Best for:** Large datasets, when you just need to correct imbalance

**2. SMOTE (Synthetic Minority Oversampling Technique)**
- Creates synthetic fraud cases by interpolating between existing fraud examples
- Helps model learn better decision boundaries
- **When to use:** Moderate imbalance (5:1 to 50:1), smaller datasets
- **When NOT to use:** Extreme imbalance (>100:1), large datasets (too slow)

**3. Undersampling**
- Remove majority class samples to balance
- Loses information
- Only use when computation cost of full dataset is prohibitive

**4. Threshold Tuning**
- Default threshold = 0.5, but for 1% fraud rate, optimal threshold may be 0.1
- Tune on validation set by sweeping threshold and optimizing F1 or business metric
- **Always do this** regardless of other methods

**Right evaluation metrics for imbalanced data:**
- Primary: PR-AUC (captures precision-recall tradeoff)
- Secondary: Recall@K (what % fraud caught in top K% of scored claims)
- Business: Capture@25 (% fraud caught in top 25% risk tier)

---

## SECTION 6: FEATURE ENGINEERING — THE MOST IMPORTANT SKILL

### Categorical Feature Encoding for Fraud Models

**Weight of Evidence (WoE):**
```
WoE = ln(Distribution_Events / Distribution_Non_Events)
     = ln((n_fraud_in_bin / total_fraud) / (n_non_fraud_in_bin / total_non_fraud))
```
- Positive WoE → bin has higher proportion of fraud than average
- Negative WoE → bin has lower proportion of fraud than average
- Used with logistic regression — transforms categorical to continuous log-odds

**Information Value (IV):**
```
IV = sum((Dist_Events - Dist_Non_Events) * WoE)
```
| IV Value | Predictive Power |
|----------|-----------------|
| < 0.02 | Useless |
| 0.02 - 0.1 | Weak predictor |
| 0.1 - 0.3 | Medium predictor |
| > 0.3 | Strong predictor (possible leakage, check!) |

**Cramer's V:**
```
Cramer's V = sqrt(Chi-squared / (N * min(rows-1, cols-1)))
```
- Measures association between two categorical variables
- 0 = no association, 1 = perfect association
- Use to detect redundant categorical features

**PSI (Population Stability Index):**
```
PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
```
| PSI | Interpretation |
|-----|----------------|
| < 0.1 | Stable — no action needed |
| 0.1 - 0.2 | Warning — investigate |
| > 0.2 | Significant shift — retrain or check data pipeline |

**VIF (Variance Inflation Factor):**
- VIF > 10 = severe multicollinearity, remove one of the correlated features
- VIF > 5 = moderate, investigate
- VIF = 1 = no multicollinearity

---

## SECTION 7: CROSS-VALIDATION

### Types and When to Use

**K-Fold Cross-Validation:**
- Split data into K folds, train on K-1, validate on 1, rotate
- Good for: general ML, balanced datasets
- Problem: doesn't preserve class distribution → use StratifiedKFold

**Stratified K-Fold:**
- Maintains class ratio in each fold
- Always use for fraud detection (1% fraud rate must be preserved)
- Default for classification in sklearn: `StratifiedKFold(n_splits=5)`

**Time-Series Cross-Validation (Walk-Forward):**
- Train on months 1-6, validate on month 7
- Train on months 1-7, validate on month 8
- Never use future data to predict the past
- Use for: fraud models with time-based data, MMM models, any forecasting

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # train on past, validate on future
```

---

## SECTION 8: FEATURE SELECTION METHODS

**Lasso (L1 Regularization):**
- Drives some coefficients to exactly zero → automatic feature selection
- Hyperparameter alpha controls sparsity: higher alpha → more zeros
- Use with logistic regression or linear regression as a baseline

**Random Forest Feature Importance:**
- Mean Decrease in Impurity (MDI): how much each feature reduces Gini across all splits
- Permutation Importance: shuffle feature, measure performance drop
- SHAP values: game-theory-based, most accurate attribution

**Mutual Information:**
- Measures statistical dependence between feature and target
- 0 = independent, higher = more predictive
- Works for non-linear relationships (unlike correlation)

**Variance Inflation Factor (VIF):**
- For numerical features: detect and remove multicollinear features before logistic regression

---

## SECTION 9: QUICK REFERENCE CARD

| Concept | One-Line Answer |
|---------|-----------------|
| Decision tree split | Choose feature-threshold maximizing information gain / minimizing weighted Gini |
| Gini (pure node) | 0. Maximum = 0.5 (equal classes) |
| Entropy (pure node) | 0. Maximum = 1 bit (50-50 binary) |
| Random Forest reduces | Variance (via bagging/averaging) |
| XGBoost reduces | Bias (via sequential boosting), also adds regularization |
| Use Precision when | False positives are costly (real-time fraud, medical decisions) |
| Use Recall when | False negatives are costly (batch fraud catch, cancer screening) |
| Use PR-AUC when | Dataset is highly imbalanced (fraud, rare events) |
| PSI > 0.2 | Significant distribution shift — retrain or investigate |
| IV > 0.3 | Strong predictor (verify no data leakage) |
| WoE | Log-ratio of event vs non-event distribution in each bin |
| Calibration | Predicted probabilities match observed rates |
| SMOTE | Synthetic minority oversampling — use for moderate imbalance |
| scale_pos_weight | XGBoost imbalance fix = n_negatives/n_positives |

---

## SECTION 10: INTERVIEW SCENARIOS — ONE-MINUTE ANSWERS

### "Why did you use XGBoost instead of logistic regression?"

> "I always start with logistic regression as a baseline. If the baseline achieves target precision/recall, I stop there — simpler is better. In this case, LR achieved 0.72 AUC PR. When I tested XGBoost, I got 0.81 AUC PR because of non-linear interactions between claim amount, policy tenure, and claim frequency that LR couldn't capture. The 9-point improvement in PR-AUC justified the added complexity."

### "How did you handle the imbalanced fraud dataset?"

> "Fraud was about 2% of claims. I used three approaches: set scale_pos_weight=49 in XGBoost (ratio of negatives to positives), tuned the decision threshold on PR curve rather than using 0.5 default, and used PR-AUC as the primary metric rather than ROC-AUC. This gave me a model that achieved 65% precision at 80% recall — much better than the default approach."

### "What features went into your fraud model?"

> "Three categories: behavioral (claim submission timing, frequency, time-since-policy-start), structured metadata (claim amount, type, policy details, demographic risk indicators), and NLP-derived indicators from claim notes (extracted binary flags for suspicious patterns using BERT + RAG pipeline). We validated each feature's predictive power using IV > 0.1 threshold and checked for PSI stability."

### "What evaluation metrics did you use?"

> "Primary metric was PR-AUC (most relevant for 2% fraud rate). Secondary: Precision@70% recall threshold (operational SLA), Capture@25 (what % of fraud is in top 25% risk tier), and calibration (Brier score). For production monitoring, PSI on input features and drift in prediction distribution checked weekly."

---

## SECTION 11: HYPOTHESIS TESTING AND STATISTICAL FOUNDATIONS FOR INTERVIEWS

### Parametric vs Non-Parametric Tests

**Parametric tests** assume the data follows a known distribution, usually the normal distribution, and estimate parameters like mean and variance.

**Non-parametric tests** make no strong distributional assumptions. They work with ranks, medians, or counts and are safer when data is skewed, ordinal, has outliers, or sample size is small.

**When to prefer each:**
- Use parametric tests when data is approximately normal, variances are similar across groups, and sample size is adequate. They have more statistical power when assumptions hold.
- Use non-parametric tests when normality is violated, data is ordinal, sample size is small, or outliers dominate the mean.

**Interview one-liner:**
> "I choose parametric tests when the data meets normality and equal-variance assumptions because they have higher power. When those assumptions fail — common in fraud claim amounts — I switch to non-parametric rank-based tests that are robust to skewness and outliers."

### The t-test Family

**One-sample t-test:** Compares the mean of one group to a known or hypothesized value.

**Independent two-sample t-test:** Compares means of two independent groups on a continuous outcome.

**Paired t-test:** Compares means from the same group at two different times or under two conditions.

**Assumptions:**
- Dependent variable is continuous
- Observations are independent
- Data is approximately normally distributed within each group
- Groups have roughly equal variances (use Welch's t-test if variances differ)

**When an interviewer asks about small samples:**
> "With only 20 observations, a t-test is risky unless the data is close to normal. I would check normality with Shapiro-Wilk first. If normality fails, I use the Mann-Whitney U test for two independent groups or Wilcoxon signed-rank for paired data."

### ANOVA and ANCOVA

**ANOVA (Analysis of Variance):** Compares means across three or more groups. It tests whether at least one group mean differs from the others.

**ANCOVA (Analysis of Covariance):** Extends ANOVA by including one or more continuous covariates. It compares group means while controlling for the effect of the covariates.

**Key assumptions:**
- Dependent variable is continuous
- Groups are independent
- Residuals are normally distributed
- Homogeneity of variances
- For ANCOVA: relationship between covariate and dependent variable is similar across groups

**Post-hoc tests:** When ANOVA is significant, run Tukey HSD or Bonferroni-corrected pairwise comparisons to identify which specific groups differ.

**Interview framing:**
> "I use t-test for 2 groups, ANOVA for 3 or more groups, and ANCOVA when I need to control for confounding continuous variables. For example, comparing fraud rates across regions while controlling for average transaction size."

### Non-Parametric Alternatives

**Mann-Whitney U test (Wilcoxon rank-sum test):** Compares two independent groups on a continuous or ordinal variable. Tests whether distributions differ in central tendency.

**Wilcoxon signed-rank test:** For paired or matched samples. Tests whether the median difference between paired observations is zero.

**Kruskal-Wallis test:** Non-parametric alternative to one-way ANOVA. Compares three or more independent groups.

**Spearman's rank correlation:** Measures monotonic relationship between two variables. Use when data is not linear or not normally distributed.

**Kendall's Tau:** Another rank correlation. More robust than Spearman when there are many tied ranks or the sample is small.

**Interview one-liner:**
> "Mann-Whitney U replaces the independent t-test, Wilcoxon signed-rank replaces the paired t-test, and Kruskal-Wallis replaces one-way ANOVA when normality assumptions fail."

### Checking Distributions and Normality

**Visual checks:**
- Histogram: shows shape, skewness, modality
- Boxplot: shows spread, central tendency, outliers
- Q-Q plot: compares sample quantiles to theoretical normal quantiles

**Statistical tests:**
- **Shapiro-Wilk test:** Powerful for small to moderate samples. Null hypothesis = data is normally distributed.
- **Kolmogorov-Smirnov test:** Compares sample distribution to a reference distribution. Less powerful for tails.
- **Anderson-Darling test:** More sensitive to deviations in the tails than KS.

**Skewness and kurtosis:**
- Skewness > 0: right-skewed tail
- Skewness < 0: left-skewed tail
- High kurtosis: heavy tails, more outliers

**Interview one-liner:**
> "I never trust a single test for normality. I use histograms, Q-Q plots, and Shapiro-Wilk together. If the data is clearly skewed or the test rejects normality, I transform the data or use non-parametric methods."

### Data Transformations for Normality

When data violates normality but a parametric approach is still desired, transformations can help.

**Log transformation:** Compresses large values and expands small values. Best for right-skewed data like income, claim amounts, or transaction values.

**Square-root transformation:** Used for count data or moderate right skewness.

**Box-Cox transformation:** A family of power transformations that finds the optimal lambda to make data more normal. Requires positive values only.

**Yeo-Johnson transformation:** Similar to Box-Cox but handles zero and negative values.

**Important caveat:**
> "Transformations change the interpretation of coefficients. I only use them when the downstream model benefits — for tree-based models like XGBoost, transformations are usually unnecessary because trees are invariant to monotonic transformations."

### Correlation vs Causation

**Pearson correlation:** Measures linear relationship. Sensitive to outliers. Assumes normal distribution.

**Spearman correlation:** Measures monotonic relationship based on ranks. More robust.

**Mutual information:** Measures any statistical dependence, including non-linear relationships.

**Distance correlation:** Detects both linear and non-linear associations.

**Partial correlation:** Measures relationship between two variables while controlling for others.

**The causation problem:**
> "Correlation does not imply causation. Two variables can correlate because they share a common cause, because of coincidence, or because one causes the other. In fraud analytics, I validate suspected relationships with domain knowledge, controlled experiments when possible, and by checking that the relationship holds across time periods and segments."

**Interview one-liner:**
> "I start with Pearson for linear relationships, switch to Spearman or mutual information for non-linear patterns, and always sanity-check with domain knowledge before treating any association as causal."

---

## SECTION 12: MULTI-LABEL vs MULTI-CLASS FRAMING

### The Distinction

**Multi-class classification:** Each sample belongs to exactly one class out of K possible classes. The classes are mutually exclusive.

**Multi-label classification:** Each sample can belong to zero, one, or multiple labels independently. Labels are not mutually exclusive.

**Example in fraud/NLP:**
- Multi-class: Classify a claim into one category — fraud type A, fraud type B, or not fraud.
- Multi-label: Extract multiple fraud indicators from claim notes — inconsistent timeline may be true, exaggerated damages may be true, staged accident may be false.

### Why the Framing Matters

The choice of framing changes:
- Output layer: softmax for multi-class, sigmoid per label for multi-label
- Loss function: categorical cross-entropy vs binary cross-entropy summed over labels
- Evaluation metrics: per-label precision/recall vs macro/micro-averaged metrics

### Evaluation for Multi-Label Problems

**Per-label metrics:** Compute precision, recall, and F1 independently for each label. This identifies which specific labels the model struggles with.

**Micro-averaging:** Aggregate all true positives, false positives, and false negatives across labels, then compute metrics. Favors frequent labels.

**Macro-averaging:** Compute metrics per label and average them equally. Gives rare labels the same weight as common ones.

**Weighted averaging:** Average per-label metrics weighted by label frequency or business importance.

**When one label is more important:**
> "If 'person' detection is more business-critical than 'animal' detection, I report per-label metrics separately and may use a weighted composite score. I also tune the decision threshold per label rather than using a single global threshold."

### Per-Label Binary View

At the individual label level, multi-label becomes a binary classification problem for each label. This is useful for:
- Threshold tuning per label
- Cost-sensitive learning per label
- Error analysis by label

**Interview one-liner:**
> "For entity extraction, the overall task is multi-label because multiple entities can appear in the same text. But for evaluation and improvement, I analyze each entity as a binary classification problem so I can tune thresholds and fix errors per entity type."

---

## SECTION 13: F-BETA SCORE — TUNING PRECISION-RECALL BALANCE

### What F-Beta Does

The F-beta score generalizes F1 by allowing different weighting of precision and recall.

**Beta < 1:** Weights precision higher than recall. Useful when false positives are more costly.

**Beta = 1:** Gives F1, equal weight to precision and recall.

**Beta > 1:** Weights recall higher than precision. Useful when false negatives are more costly.

**Common choices:**
- F0.5: Precision is twice as important as recall
- F1: Balanced
- F2: Recall is twice as important as precision

### Business Translation

**F0.5 in real-time fraud:**
> "For real-time fraud flagging where every false positive triggers a manual review, I might use F0.5 to emphasize precision. The business can tolerate missing some fraud more than overwhelming investigators with false alarms."

**F2 in batch fraud or critical entity extraction:**
> "For batch fraud detection where missing fraud is costlier than extra reviews, or for extracting a critical entity like 'person' where missing it has compliance risk, I use F2 to emphasize recall."

### Choosing Beta

The beta value should come from the business cost of false positives vs false negatives:
- Estimate cost of FP and FN in dollars or operational impact
- Set beta so the metric reflects the true tradeoff
- Validate by checking that optimizing F-beta also optimizes business outcome

**Interview one-liner:**
> "I choose beta based on the relative cost of false positives versus false negatives. If a false negative costs twice as much as a false positive, F2 is appropriate. If a false positive costs twice as much, F0.5 is appropriate."

---

## SECTION 14: DEEPER CALIBRATION AND PROBABILITY QUALITY

### Why Calibration Matters Beyond Brier Score

A well-calibrated model means predicted probabilities match observed frequencies. This is critical when probabilities drive decisions:
- Risk tiering: claims with 70% probability should actually be fraudulent 70% of the time
- Threshold setting: wrong calibration leads to wrong operational cutoffs
- Cost-sensitive decisions: expected value calculations depend on accurate probabilities

### Reliability Diagrams

A reliability diagram bins predictions by probability and plots predicted probability versus observed frequency.
- Perfect calibration: points fall on the diagonal
- Over-confident model: predicted probabilities are higher than observed rates
- Under-confident model: predicted probabilities are lower than observed rates

### Calibration Methods

**Platt scaling:** Fits a logistic regression on top of model outputs. Simple and effective when the model's score distribution is roughly sigmoid-shaped.

**Isotonic regression:** A non-parametric calibration method that learns a monotonic mapping from scores to probabilities. More flexible than Platt scaling but requires more data.

**When to calibrate:**
> "I calibrate when probabilities are used for threshold-based decisions or risk tiering. Tree-based models like XGBoost and LightGBM often produce reasonably calibrated probabilities, but I verify with reliability diagrams and Brier score. If calibration is poor, I apply Platt scaling or isotonic regression on the validation set."

### Expected Calibration Error

Expected Calibration Error (ECE) summarizes calibration quality by averaging the absolute difference between predicted probability and observed frequency across bins.

**Interview one-liner:**
> "Calibration is not just a nice-to-have — it determines whether our 70% risk threshold actually flags 70% fraudulent cases. I check reliability diagrams and Expected Calibration Error, and apply Platt scaling or isotonic regression if needed."

---

## SECTION 15: BALANCED ACCURACY AND OTHER IMBALANCED METRICS

### Balanced Accuracy

Balanced accuracy is the average of recall for each class.

**Why it matters:**
> "In a 98% not-fraud dataset, a model that always predicts 'not fraud' gets 98% accuracy but 0% recall on fraud. Balanced accuracy exposes this: it would be 50%."

**When to use:**
- Multi-class problems with class imbalance
- When you want a single metric that treats all classes equally

### Cohen's Kappa

Cohen's Kappa measures agreement between predicted and actual labels, correcting for chance agreement.
- Kappa = 1: perfect agreement
- Kappa = 0: agreement equal to chance
- Useful when classes are imbalanced and you want to measure agreement beyond accuracy

### Matthews Correlation Coefficient

MCC is a balanced measure for binary classification that uses all four cells of the confusion matrix.
- Range: -1 to +1
- +1: perfect prediction
- 0: random prediction
- Useful for imbalanced datasets because it is not inflated by class size

**Interview one-liner:**
> "For imbalanced problems, I avoid accuracy as the primary metric. I use PR-AUC, balanced accuracy, or Matthews Correlation Coefficient depending on whether the problem is binary or multi-class and whether I need a single summary metric."

---

## SECTION 16: INTERVIEW SCENARIOS — STATISTICAL AND EVALUATION DEPTH

### "How would you check if two numerical variables are related?"

> "I would start with a scatter plot to visualize the relationship. Then I would compute Pearson correlation if the relationship looks linear and data is roughly normal. If the relationship is monotonic but non-linear, I use Spearman. If I suspect any non-linear dependency, I use mutual information. Finally, I validate with domain knowledge and consider whether correlation could be due to a confounding variable."

### "Explain the difference between t-test, ANOVA, and ANCOVA."

> "Use a t-test to compare the mean of a continuous variable between two groups. Use ANOVA when you have three or more groups and want to test whether at least one group mean differs. Use ANCOVA when you want to compare group means while controlling for one or more continuous covariates that might confound the comparison."

### "When would you use a non-parametric test?"

> "I use non-parametric tests when the data violates the assumptions of parametric tests — especially when the distribution is heavily skewed, there are significant outliers, the sample size is small, or the data is ordinal. For example, fraud claim amounts are often right-skewed, so comparing claim amounts across groups with Mann-Whitney U or Kruskal-Wallis is safer than t-test or ANOVA."

### "How do you handle a situation where you need parametric tests but the data is not normal?"

> "First, I try a transformation — log, square-root, Box-Cox, or Yeo-Johnson — to make the distribution closer to normal. I re-check normality after transformation. If transformation does not work or is not interpretable, I switch to non-parametric alternatives or use robust parametric methods like Welch's t-test for unequal variances."

### "How do you evaluate a multi-label classification model?"

> "I compute precision, recall, and F1 for each label independently to diagnose per-label performance. Then I report macro-averaged F1 if all labels are equally important, or weighted/micro-averaged F1 if frequency matters. I also use a confusion matrix per label and analyze co-occurrence errors. If one label is business-critical, I may use F-beta with beta tuned to the cost of false negatives versus false positives for that label."

### "How would you explain the difference between precision and recall to a non-technical stakeholder?"

> "Think of a fraud detection system as a net. Precision asks: of everything the net caught, how much was actually fraud? High precision means few false alarms. Recall asks: of all the fraud that existed, how much did the net catch? High recall means few missed fraud cases. The right balance depends on whether false alarms or missed fraud cost the business more."

### "How would you explain F1 score to a non-technical stakeholder?"

> "F1 score is a single number that combines precision and recall using a special average called the harmonic mean. It only gets high when both precision and recall are high. A model that is excellent at one but terrible at the other will have a low F1, which helps us avoid being misled by a model that looks good on just one metric."

### "Why is the harmonic mean used in F1 instead of a regular average?"

> "A regular average can hide weakness. For example, 25% precision and 75% recall gives an arithmetic average of 50%, same as 50% precision and 50% recall. But the harmonic mean gives 37.5% for the first case and 50% for the second. The harmonic mean forces the model to be good at both precision and recall to score well."

---

## SECTION 17: DATA VERSIONING AND DVC

### Why Version Data

A model is a function of code + data + hyperparameters. Without versioning data, you cannot reproduce a model or debug why performance changed.

### DVC (Data Version Control)

DVC versions large files and directories by storing metadata in Git and actual data in remote storage such as S3, GCS, or Azure Blob.

**What to version with DVC:**
- Raw datasets
- Processed feature files
- Trained model artifacts
- Large configuration files

**DVC vs Delta Lake:**
- DVC: version files and artifacts
- Delta Lake: version structured tables with time travel
- MLflow: link code, data version, and model together

### Best Practice

Use Git for code, DVC for large artifacts, Delta Lake for structured feature tables, and MLflow for experiment and model registry metadata.

### Interview One-Liner

> "I version code with Git, structured data with Delta Lake, and large artifacts with DVC. MLflow ties them together by recording the data version and code commit for every training run."

---

## SECTION 18: ADVANCED DISTRIBUTION DISTANCE METRICS

### KL Divergence

Kullback-Leibler divergence measures how much information is lost when using distribution Q to approximate distribution P. It is not symmetric.

**Use case:** comparing model score distributions between training and production.

### Jensen-Shannon Divergence

Jensen-Shannon divergence is a symmetric, smoothed version of KL divergence bounded between 0 and 1.

**Advantages over KL:**
- Symmetric
- Bounded
- Handles zero bins better

**Use case:** drift detection when you want a stable, bounded metric.

### Wasserstein Distance

Wasserstein distance measures how much probability mass must move to align two distributions. It is useful when the location of the distribution shift matters, not just the shape.

### Interview One-Liner

> "For operational drift monitoring I use PSI and KS. When I need a bounded symmetric metric for analysis, I use Jensen-Shannon divergence. Wasserstein distance is useful when I care about where the distribution shifted, not just that it shifted."

---

## SECTION 19: LABEL DRIFT AND FEEDBACK LOOPS

### Label Drift

Label drift means the distribution of the target variable has changed over time. In fraud, this can happen when the business approves more policies in a region with higher or lower fraud rates.

### Detection

- Track the rate of positive labels over time
- Compare label distribution by segment
- Check whether label maturation windows are consistent

### Feedback Loops

A model can influence the labels it is trained on:
- Fraud model flags claims → SIU investigates flagged claims → labels are mostly from flagged claims → model learns to replicate its own bias
- Recommendation model recommends popular items → users click them → clicks confirm popularity → model becomes more popular-biased

### Mitigation

- Randomly sample some unselected cases for labeling
- Use propensity weighting
- Hold out a population that does not see the model's decisions
- Monitor model performance on a neutral holdout set

### Interview One-Liner

> "Label drift and feedback loops are subtle. I track label rates over time, randomly sample cases that bypass the model for labeling, and maintain a holdout set where the model does not influence the outcome. This keeps training data representative."

---

## SECTION 20: MODEL EVALUATION GATES

### What Is a Model Evaluation Gate

An evaluation gate is a checkpoint that prevents a model from moving forward unless it meets defined criteria.

### Common Gates

| Gate | Purpose |
|---|---|
| Performance gate | Model beats champion or baseline by required margin |
| Guardrail gate | No degradation on fairness, latency, calibration |
| Reproducibility gate | Same result can be reproduced from recorded code and data |
| Business gate | Meets operational constraints like false positive budget |
| Security gate | No PII leakage, no adversarial vulnerabilities |

### Why Gates Matter

Gates prevent bad models from reaching production and create accountability. They also make promotion decisions objective rather than based on hope.

### Interview One-Liner

> "Before promoting any model, I run it through performance, guardrail, reproducibility, business, and security gates. This ensures the model is better than what we have, does not break constraints, and can be reproduced."

---

## SECTION 21: SLOs AND SLIs FOR ML SERVICES

### Definitions

- **SLI:** Service Level Indicator — a measurable metric
- **SLO:** Service Level Objective — target value for the SLI
- **SLA:** Service Level Agreement — contract with consequences
- **Error budget:** allowed unreliability before pausing changes

### Example ML SLOs

| SLI | Example SLO |
|---|---|
| p99 latency | < 200ms |
| Error rate | < 0.1% |
| Availability | 99.9% |
| Feature freshness | < 5 minutes |
| Prediction drift PSI | < 0.1 |

### Error Budget Policy

When the error budget is consumed, pause feature work and focus on reliability.

### Interview One-Liner

> "I define SLOs for latency, error rate, availability, feature freshness, and prediction drift. An error budget tells the team when to stop shipping features and fix reliability."

---

## SECTION 22: ADDITIONAL FUNDAMENTAL INTERVIEW SCENARIOS

### "What is the difference between a feature store and a data warehouse?"

> "A feature store provides point-in-time correct features and training-serving consistency. A data warehouse stores data for analytics but does not guarantee that the features used at training time match those used at inference time. A feature store is built on top of the warehouse and adds ML-specific abstractions."

### "How do you know if a model is overfitting?"

> "I compare training and validation metrics. If training AUC is much higher than validation AUC, the model is overfitting. I fix it with regularization, more data, simpler model, or early stopping."

### "What is the bias-variance tradeoff?"

> "High bias means the model is too simple and underfits. High variance means the model is too complex and overfits. The goal is to find the complexity that minimizes total error on unseen data."

### "When would you use a parametric vs non-parametric model?"

> "Parametric models like logistic regression make strong assumptions and need less data but may underfit complex patterns. Non-parametric models like gradient boosting or k-NN are more flexible but need more data and can overfit. I choose based on data size, feature complexity, and interpretability needs."
