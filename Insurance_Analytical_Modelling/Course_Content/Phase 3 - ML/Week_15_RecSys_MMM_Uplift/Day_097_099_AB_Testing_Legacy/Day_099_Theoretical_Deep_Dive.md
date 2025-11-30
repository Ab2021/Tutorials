# AB Testing & Experimental Design (Part 3) - Causal Inference & Observational Studies - Theoretical Deep Dive

## Overview
"What if we can't run an experiment?"
Sometimes A/B testing is impossible (unethical, too expensive, or historical).
In these cases, we must find Causality in **Observational Data**.
This day focuses on **Causal Inference**, **Propensity Score Matching (PSM)**, and **Quasi-Experiments** (Diff-in-Diff).

---

## 1. Conceptual Foundation

### 1.1 The Fundamental Problem of Causal Inference

*   **Problem:** We never see the *Counterfactual*.
    *   We see Customer A *with* the discount.
    *   We never see Customer A *without* the discount (at the same time).
*   **Goal:** Estimate the Average Treatment Effect (ATE).
    $$ ATE = E[Y_1 - Y_0] $$

### 1.2 Confounding

*   **Scenario:** Customers who bought "Premium Package" have lower loss ratios.
*   **Causality?** Did the package *cause* safety?
*   **Confounder:** Wealth. Wealthy people buy the package AND drive safer cars.
*   **Result:** Correlation $\neq$ Causation.

---

## 2. Mathematical Framework

### 2.1 Propensity Score Matching (PSM)

*   **Idea:** Create a "Synthetic Control Group".
*   **Propensity Score:** $e(x) = P(Treatment=1 | X)$. (Probability of being treated).
*   **Matching:** Find untreated customers who had the *same probability* of buying the package as the treated ones.
*   **Comparison:** Compare outcomes between Matched Treated and Matched Control.

### 2.2 Difference in Differences (DiD)

*   **Scenario:** Policy change in State A (Treatment) but not State B (Control).
*   **Assumption:** Parallel Trends. (Without treatment, A and B would have moved together).
*   **Formula:**
    $$ \tau = (Y_{A, post} - Y_{A, pre}) - (Y_{B, post} - Y_{B, pre}) $$

---

## 3. Theoretical Properties

### 3.1 Instrumental Variables (IV)

*   **Use:** When there is unobserved confounding.
*   **Instrument (Z):** A variable that affects Treatment (T) but has no direct effect on Outcome (Y).
    *   Example: Random Assignment of Agents.
    *   Z (Agent) $\to$ T (Discount) $\to$ Y (Retention).
*   **Estimator:** 2SLS (Two-Stage Least Squares).

### 3.2 Regression Discontinuity Design (RDD)

*   **Scenario:** Treatment is assigned based on a cutoff.
    *   "Credit Score > 700 gets discount."
*   **Logic:** People with 699 and 701 are basically identical.
*   **Method:** Compare outcomes just above and just below the threshold.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Propensity Score Matching (Python)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# 1. Estimate Propensity Score
model = LogisticRegression()
model.fit(X, treatment)
df['pscore'] = model.predict_proba(X)[:, 1]

# 2. Match
treated = df[df['treatment'] == 1]
control = df[df['treatment'] == 0]

nbrs = NearestNeighbors(n_neighbors=1).fit(control[['pscore']])
distances, indices = nbrs.kneighbors(treated[['pscore']])

# 3. Compare Outcomes
matched_control = control.iloc[indices.flatten()]
ate = treated['outcome'].mean() - matched_control['outcome'].mean()
print(f"Estimated ATE: {ate:.2f}")
```

### 4.2 CausalImpact (Google)

*   **Library:** `causalimpact` (Python port).
*   **Method:** Bayesian Structural Time Series.
*   **Use:** "What was the impact of the TV ad campaign?" (No control group, just time series).
*   **Counterfactual:** Predicts what *would have happened* without the ad based on correlated covariates (e.g., Weather, Competitor Stock).

---

## 5. Evaluation & Validation

### 5.1 Placebo Tests

*   **Method:** Run the DiD analysis on a period *before* the treatment.
*   **Goal:** The effect should be 0.
*   **Failure:** If you find a "Significant Effect" in the past, your Parallel Trends assumption is violated.

### 5.2 Covariate Balance

*   **Check:** After PSM, are the Treated and Control groups similar?
*   **Metric:** Standardized Mean Difference (SMD). Should be < 0.1 for all features.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Selection Bias

*   **Issue:** "Opt-in" programs.
*   **Bias:** People who opt-in to "Safe Driving App" are *already* safe drivers.
*   **Fix:** You cannot compare Opt-in vs. Non-Opt-in directly. You need an Instrument or rigorous matching.

### 6.2 Overlap (Common Support)

*   **Issue:** If all wealthy people are treated, and all poor people are control, you cannot match.
*   **Requirement:** There must be overlap in the Propensity Score distributions.

---

## 7. Advanced Topics & Extensions

### 7.1 Double Machine Learning (DML)

*   **Method:** Use ML to predict Y from X, and T from X.
*   **Residuals:** Regress $(Y - \hat{Y})$ on $(T - \hat{T})$.
*   **Benefit:** Allows using Random Forests/Neural Nets for confounding control while getting a valid statistical inference for the Treatment Effect.

### 7.2 Uplift Modeling vs. Causal Inference

*   **Uplift:** Predicts *individual* heterogeneity ($\tau_i$). (Who to target).
*   **Causal Inference:** Focuses on *average* effects and validity ($\tau$). (Did it work?).

---

## 8. Regulatory & Governance Considerations

### 8.1 Evidence Standards

*   **Context:** Proving to a regulator that a rating factor is valid.
*   **Requirement:** Correlation is often enough for *Risk*, but Causality is required for *Behavioral* discounts (e.g., UBI).

---

## 9. Practical Example

### 9.1 The "Telematics" Effect

**Question:** Does installing a Telematics device *cause* people to drive safer? Or do safe drivers just install it?
**Method:** Difference in Differences.
*   **Group A:** Installed Device.
*   **Group B:** Did not install.
*   **Metric:** Accidents per mile.
*   **Analysis:** Compare the *change* in accidents for A vs. B before and after installation.
**Result:** 5% reduction in accidents is causal (Feedback effect). The rest is selection bias.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Confounding** is the enemy.
2.  **PSM** creates a control group.
3.  **DiD** uses time and groups.

### 10.2 When to Use This Knowledge
*   **Policy Analysis:** Evaluating law changes.
*   **Program Evaluation:** Did the "Loyalty Program" work?

### 10.3 Critical Success Factors
1.  **Domain Knowledge:** You need to know *what* the confounders are to measure them.
2.  **Skepticism:** Observational studies are guilty until proven innocent.

### 10.4 Further Reading
*   **Pearl:** "The Book of Why".
*   **Imbens & Rubin:** "Causal Inference".

---

## Appendix

### A. Glossary
*   **Counterfactual:** What would have happened.
*   **Endogeneity:** When X is correlated with the error term.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **DiD** | $(\bar{Y}_{T,2}-\bar{Y}_{T,1}) - (\bar{Y}_{C,2}-\bar{Y}_{C,1})$ | Policy Effect |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
