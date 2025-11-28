# Model Interpretability (Part 2) - Fairness & Bias - Theoretical Deep Dive

## Overview
It is illegal to use Race as a rating factor. But what if your AI uses "Zip Code" as a proxy for Race? This session covers **Algorithmic Bias**, **Fairness Metrics**, and how to "De-bias" a model using **Fairlearn** and **AIF360**.

---

## 1. Conceptual Foundation

### 1.1 Types of Bias

1.  **Disparate Treatment (Intentional):** Explicitly using "Gender" or "Race" in the model (Illegal in most lines).
2.  **Disparate Impact (Unintentional):** Using a neutral variable (e.g., Credit Score) that disproportionately hurts a protected group.
3.  **Proxy Bias:** The model learns to predict Race from other variables (Zip Code, Magazine Subscriptions) and uses it to price.

### 1.2 Fairness Definitions

*   **Demographic Parity:** The acceptance rate should be equal for all groups.
    *   $P(\hat{Y}=1 | G=A) = P(\hat{Y}=1 | G=B)$.
*   **Equalized Odds:** The True Positive Rate (TPR) and False Positive Rate (FPR) should be equal.
    *   *Translation:* If we make a mistake, we should make mistakes equally for everyone.

### 1.3 The Impossibility Theorem

*   You cannot satisfy all fairness metrics simultaneously (unless the base rates are identical).
*   *Actuarial Choice:* We usually prioritize **Calibration** (Risk Score reflects true risk) over Demographic Parity.

---

## 2. Mathematical Framework

### 2.1 Disparate Impact Ratio (DIR)

$$ \text{DIR} = \frac{P(\hat{Y}=1 | \text{Unprivileged})}{P(\hat{Y}=1 | \text{Privileged})} $$
*   **Rule of Thumb:** If DIR < 0.80 (The "Four-Fifths Rule"), the model is flagged for bias.

### 2.2 Reweighing (Pre-processing)

*   **Goal:** Make the training data "fair" before the model sees it.
*   **Method:**
    *   If Group A has fewer positive labels than expected, increase the weight of those rows.
    *   If Group B has too many, decrease their weight.

---

## 3. Theoretical Properties

### 3.1 Adversarial Debiasing (In-processing)

*   **Architecture:** Two Neural Networks.
    1.  **Predictor:** Tries to predict Risk.
    2.  **Adversary:** Tries to predict Race from the Predictor's output.
*   **Game:** The Predictor is penalized if the Adversary guesses correctly.
*   **Result:** The Predictor learns to predict Risk *without* encoding Race info.

### 3.2 Post-processing (Threshold Adjustment)

*   Train the model normally.
*   Set different thresholds for different groups to achieve Equalized Odds.
*   *Legal Risk:* This is often considered "Reverse Discrimination" and is illegal in insurance (e.g., California Prop 103).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Measuring Bias with Fairlearn

```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Load Data
# y_true: Actual Default
# y_pred: Predicted Default
# sensitive_features: Race/Gender column
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}

# 2. Calculate Metrics by Group
mf = MetricFrame(
    metrics=metrics,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=df['Race']
)

print(mf.by_group)
# Output:
# Race      accuracy  selection_rate
# Group A   0.85      0.40
# Group B   0.82      0.20  <-- Disparate Impact?
```

### 4.2 Mitigating Bias (Exponentiated Gradient)

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# 1. Define Constraint
constraint = DemographicParity()

# 2. Train Mitigated Model
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=constraint
)
mitigator.fit(X, y, sensitive_features=df['Race'])

# 3. Predict
y_pred_fair = mitigator.predict(X)
```

---

## 5. Evaluation & Validation

### 5.1 The Fairness-Accuracy Trade-off

*   Removing bias usually hurts accuracy (because you are ignoring real correlations).
*   **Pareto Frontier:** Plot Accuracy vs. Fairness. Choose the model that maximizes Fairness with minimal Accuracy loss.

### 5.2 Sensitivity Analysis

*   Does the model rely on "High Risk" Zip Codes?
*   *Test:* Swap the Zip Code of a profile. Does the premium change drastically?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: "Fairness through Unawareness"**
    *   "We just deleted the Race column, so we are fair."
    *   *Wrong:* Redlining works because Zip Code proxies for Race. You need the Race column to *check* for bias.

2.  **Trap: Simpson's Paradox**
    *   The model looks biased globally, but fair within each subgroup (or vice versa).
    *   *Fix:* Analyze bias at multiple levels of granularity.

### 6.2 Implementation Challenges

1.  **Data Collection:**
    *   Insurers often *don't have* Race data (it's illegal to ask).
    *   *Fix:* Use BISG (Bayesian Improved Surname Geocoding) to infer Race for testing purposes only.

---

## 7. Advanced Topics & Extensions

### 7.1 Individual Fairness

*   "Similar individuals should be treated similarly."
*   Requires a distance metric $d(x, y)$ to define "similar". Hard to define in practice.

### 7.2 Counterfactual Fairness

*   "If Bob were Black instead of White, would the prediction be the same?"
*   Requires Causal Inference models.

---

## 8. Regulatory & Governance Considerations

### 8.1 Colorado SB 21-169

*   New law prohibiting "Unfair Discrimination" in Insurance AI.
*   Requires insurers to test for disproportionate impact on Race/Color/Religion/Sex/Sexual Orientation.

### 8.2 The "Actuarial Justification" Defense

*   If a variable (e.g., Credit Score) causes Disparate Impact, it is allowed *if and only if*:
    1.  It is actuarially justified (predicts risk).
    2.  There is no "less discriminatory alternative" (LDA) that predicts risk equally well.

---

## 9. Practical Example

### 9.1 Worked Example: The Credit Score Debate

**Scenario:**
*   Auto Insurer uses Credit Score.
*   **Analysis:**
    *   High Credit = Low Loss Ratio. (Actuarially Justified).
    *   Minorities have lower Credit Scores on average. (Disparate Impact).
*   **Search for LDA:**
    *   Tried replacing Credit with "Payment History".
    *   Result: Payment History predicted risk poorly.
*   **Conclusion:** Credit Score kept, but weight reduced slightly to balance fairness.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Disparate Impact** is the main legal risk.
2.  **Fairness through Unawareness** fails due to proxies.
3.  **Trade-off:** You often pay for Fairness with Accuracy.

### 10.2 When to Use This Knowledge
*   **Model Validation:** Every new model must pass a Bias Audit.
*   **Regulatory Compliance:** Answering "Why is this variable allowed?".

### 10.3 Critical Success Factors
1.  **Collect Protected Class Data:** You can't fix what you can't measure.
2.  **Document Everything:** Why did you choose *this* fairness metric?

### 10.4 Further Reading
*   **Barocas et al.:** "Fairness and Machine Learning".
*   **O'Neil:** "Weapons of Math Destruction".

---

## Appendix

### A. Glossary
*   **Protected Class:** A group protected by law (Race, Gender, Age, etc.).
*   **Redlining:** Denying services to residents of certain areas based on racial composition.
*   **BISG:** Method to guess race from Name + Zip.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Demographic Parity** | $P(\hat{Y}=1|A) = P(\hat{Y}=1|B)$ | Fairness Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
