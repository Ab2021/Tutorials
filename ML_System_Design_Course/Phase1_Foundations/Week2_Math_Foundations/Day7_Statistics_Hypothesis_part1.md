# Day 7 (Part 1): Advanced Statistics & Causal Inference

> **Phase**: 6 - Deep Dive
> **Topic**: Rigorous Experimentation
> **Focus**: Bootstrap, Non-Parametrics, and Causality
> **Reading Time**: 60 mins

---

## 1. Resampling Methods

When you don't know the theoretical distribution.

### 1.1 The Bootstrap
*   **Goal**: Estimate the Standard Error / Confidence Interval of *any* statistic (Mean, Median, 99th Percentile).
*   **Algorithm**:
    1.  Resample $N$ points from data $D$ *with replacement*.
    2.  Compute statistic.
    3.  Repeat 10,000 times.
    4.  Compute std dev of the results.
*   **Assumption**: The sample approximates the population.

### 1.2 Permutation Tests
*   **Goal**: Test if two groups differ (A/B test).
*   **Algorithm**:
    1.  Shuffle labels (A/B) randomly.
    2.  Compute difference in means.
    3.  Repeat.
    4.  See how often random difference > observed difference.

---

## 2. Non-Parametric Tests

When data is not Normal (e.g., Power Law, skewed).

*   **Mann-Whitney U Test**: Compare medians of two independent groups.
*   **Wilcoxon Signed-Rank Test**: Paired samples.
*   **Kruskal-Wallis**: ANOVA for non-normal data.

---

## 3. Causal Inference (The New Frontier)

Correlation $\neq$ Causation.

### 3.1 Simpson's Paradox
*   **Scenario**: Drug A looks better than Drug B overall. But Drug B is better for men AND better for women.
*   **Cause**: Confounder (e.g., Severity of illness). Drug A was given to sicker patients.
*   **Fix**: Stratification.

### 3.2 The "Do" Operator (Pearl)
*   $P(Y|X)$: Observational. "Given I see X, what is Y?"
*   $P(Y|do(X))$: Interventional. "If I force X to happen, what is Y?"
*   **Instrumental Variables**: Used when you can't randomize (e.g., effect of smoking).

---

## 4. Tricky Interview Questions

### Q1: What is "P-Hacking"?
> **Answer**: Trying multiple hypotheses/metrics until one yields $p < 0.05$.
> *   **Example**: Testing 20 different colors for a button. Even if no effect, 1 will likely be significant by chance ($1/20 = 0.05$).
> *   **Fix**: Bonferroni Correction ($\alpha / N$).

### Q2: Explain "Power" in A/B testing.
> **Answer**: Probability of correctly rejecting the Null Hypothesis when it is false ($1 - \beta$).
> *   **Factors**: Sample Size, Effect Size, Significance Level ($\alpha$).
> *   **Usage**: Calculate sample size *before* the experiment. "To detect a 1% lift with 80% power, we need 10k users."

### Q3: Can you use a T-test on non-normal data?
> **Answer**: Yes, if sample size is large (Central Limit Theorem). The distribution of the *sample mean* becomes normal even if the data is not. However, for small samples or extreme skew (outliers), T-test loses power.

---

## 5. Practical Edge Case: SRM (Sample Ratio Mismatch)
*   **Scenario**: You target 50/50 split for A/B test. You get 50.5% / 49.5%.
*   **Check**: Run a Chi-Square test on the *counts*. If significant, your assignment mechanism is broken (e.g., buggy hash function, latency timeouts affecting one group). **Invalidate the experiment.**

