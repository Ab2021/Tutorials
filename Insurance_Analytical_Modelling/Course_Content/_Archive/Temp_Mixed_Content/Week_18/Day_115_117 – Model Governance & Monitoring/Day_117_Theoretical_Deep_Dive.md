# Model Governance & Monitoring (Part 3) - Model Fairness & Ethics in AI - Theoretical Deep Dive

## Overview
"Is your model racist?"
This is no longer a philosophical question. It is a legal one.
**Colorado SB 21-169** and the **EU AI Act** mandate that insurers prove their models do not discriminate on the basis of Race, Gender, or Religion.
Today, we learn how to measure and mitigate **Algorithmic Bias**.

---

## 1. Conceptual Foundation

### 1.1 Fairness through Unawareness (The Trap)

*   **Idea:** "I removed the 'Race' column, so my model is fair."
*   **Reality:** **Proxy Discrimination**.
    *   *Example:* Zip Code is highly correlated with Race. Credit Score is correlated with Race.
    *   If you remove Race but keep Zip Code, the model "reconstructs" Race and discriminates anyway.

### 1.2 Types of Fairness

1.  **Group Fairness:** Do Men and Women get the same acceptance rate?
2.  **Individual Fairness:** Do similar individuals get similar predictions?

---

## 2. Mathematical Framework

### 2.1 Disparate Impact Ratio (DIR)

$$ \text{DIR} = \frac{P(\hat{Y}=1 | G=\text{Underprivileged})}{P(\hat{Y}=1 | G=\text{Privileged})} $$

*   *Example:* Acceptance Rate for Women / Acceptance Rate for Men.
*   **Rule of Thumb:** DIR < 0.8 (The "Four-Fifths Rule") indicates discrimination.

### 2.2 Equal Opportunity Difference

$$ \text{EOD} = TPR_{underprivileged} - TPR_{privileged} $$

*   **TPR (True Positive Rate):** Of the people who *should* be accepted, how many *were* accepted?
*   *Goal:* EOD should be close to 0.

---

## 3. Theoretical Properties

### 3.1 The Impossibility Theorem (Kleinberg et al.)

*   You cannot satisfy **Calibration**, **Equalized Odds**, and **Predictive Parity** simultaneously (unless the base rates are equal).
*   *Implication:* You must choose *which* definition of fairness matters for your business problem.

### 3.2 Bias-Variance-Fairness Tradeoff

*   Improving Fairness often degrades Accuracy.
*   *Why?* You are constraining the model to ignore certain "predictive" (but biased) signals.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Python Libraries

*   **Fairlearn (Microsoft):** Excellent for visualization and mitigation.
*   **AIF360 (IBM):** Comprehensive suite of metrics and algorithms.

### 4.2 Measuring Bias (Fairlearn)

```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score

# 1. Define Sensitive Feature
sensitive_feature = df['gender']

# 2. Calculate Metrics
mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print(mf.by_group)
# Output:
#         accuracy  selection_rate
# gender
# Female      0.85            0.20
# Male        0.86            0.40  <-- Disparate Impact!
```

---

## 5. Evaluation & Validation

### 5.1 The "Bifocal" View

*   **Performance View:** Gini, AUC, RMSE.
*   **Fairness View:** DIR, EOD, Demographic Parity.
*   *Dashboard:* You need both side-by-side.

### 5.2 Root Cause Analysis

*   Why is the model biased?
    *   **Sampling Bias:** Did we only train on data from wealthy neighborhoods?
    *   **Label Bias:** Is the historical "Fraud" label biased because investigators targeted certain groups?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: "We don't collect Race data"**
    *   *Problem:* If you don't have the label, you can't measure the bias.
    *   *Fix:* **BISG (Bayesian Improved Surname Geocoding)**. Infer race from Name + Zip Code to test for bias (standard regulatory practice).

2.  **Trap: Ignoring Intersectionality**
    *   The model might be fair for "Women" and fair for "Black People", but discriminatory against "Black Women".

---

## 7. Advanced Topics & Extensions

### 7.1 Adversarial Debiasing

*   **Architecture:**
    *   **Predictor:** Tries to predict Claims.
    *   **Adversary:** Tries to predict Race from the Predictor's output.
*   **Game:** The Predictor tries to minimize Claim Error *while maximizing* Adversary Error.
*   *Result:* A model that predicts claims but contains no information about Race.

### 7.2 Counterfactual Fairness

*   "If I changed this applicant's race from White to Black, would the prediction change?"
*   Requires Causal Graphs.

---

## 8. Regulatory & Governance Considerations

### 8.1 Colorado SB 21-169

*   Prohibits use of external data sources (ECDIS) that result in unfair discrimination.
*   Requires insurers to test their models and report results.

### 8.2 The "Business Necessity" Defense

*   If a variable causes Disparate Impact (e.g., Credit Score), you can still use it IF:
    1.  It is predictive of risk (Actuarially Sound).
    2.  There is no less discriminatory alternative.

---

## 9. Practical Example

### 9.1 Worked Example: Pricing Bias

**Scenario:**
*   **Model:** Auto Pricing.
*   **Variable:** "Credit Score".
*   **Finding:**
    *   Credit Score is highly predictive of claims (Gini +10%).
    *   Credit Score is lower for Minority groups (DIR = 0.7).
*   **Mitigation:**
    *   **Option A:** Remove Credit Score. (Gini drops to +2%). Business rejects.
    *   **Option B (Reweighting):** Weight the training data to balance the groups. (DIR improves to 0.85, Gini drops to +9%).
    *   **Decision:** Option B is the optimal trade-off.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Fairness through Unawareness** fails due to Proxies.
2.  **Disparate Impact** is the primary regulatory metric.
3.  **Trade-offs** are inevitable.

### 10.2 When to Use This Knowledge
*   **Before Deployment:** Every Tier 1 model must pass a Fairness Test.
*   **Regulatory Exams:** Be ready to show your work.

### 10.3 Critical Success Factors
1.  **Data:** You need sensitive attributes (or proxies) to test.
2.  **Ethics:** Just because it's legal doesn't mean it's right.

### 10.4 Further Reading
*   **Barocas, Hardt, Narayanan:** "Fairness and Machine Learning".

---

## Appendix

### A. Glossary
*   **Protected Class:** Race, Color, Religion, National Origin, Sex, Marital Status, Age.
*   **BISG:** Bayesian Improved Surname Geocoding.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Disparate Impact** | $P(Y=1|G=0) / P(Y=1|G=1)$ | Fairness Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
