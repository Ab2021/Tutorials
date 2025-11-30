# Litigation & Fraud Deep Dive (Part 3) - Ethics, Bias & Regulation - Theoretical Deep Dive

## Overview
"The model is mathematically optimal. But is it legal?"
Fraud models are powerful weapons. If pointed in the wrong direction, they destroy lives.
A "False Positive" isn't just a statistic; it's an innocent grandmother being interrogated by an investigator.
This day focuses on **Algorithmic Fairness**, **Proxy Discrimination**, and the emerging **Regulatory Landscape** (Colorado, EU AI Act).

---

## 1. Conceptual Foundation

### 1.1 The "Black Box" Danger

*   **Scenario:** An XGBoost model flags a claim as "95% Fraud Probability".
*   **Investigator:** "Why?"
*   **Model:** "Because Feature 42 > 0.5."
*   **Reality:** Feature 42 is highly correlated with the claimant's zip code, which is highly correlated with race.
*   **Result:** You have automated redlining.

### 1.2 Disparate Impact vs. Disparate Treatment

*   **Disparate Treatment:** Explicitly using protected class (Race, Religion) as a variable. (Illegal everywhere).
*   **Disparate Impact:** Using a neutral variable (Credit Score, Zip Code) that disproportionately hurts a protected class. (Legally gray, but increasingly regulated).

---

## 2. Mathematical Framework

### 2.1 Fairness Metrics

How do we measure if a model is racist?
1.  **Demographic Parity:** $P(\text{Flagged} | \text{Group A}) = P(\text{Flagged} | \text{Group B})$.
    *   *Critique:* If Group A actually commits more fraud, this forces the model to be inaccurate.
2.  **Equalized Odds:** $P(\text{Flagged} | \text{Fraud}, \text{Group A}) = P(\text{Flagged} | \text{Fraud}, \text{Group B})$.
    *   *Meaning:* The "True Positive Rate" (Recall) should be the same for all groups.
3.  **Predictive Parity:** $P(\text{Fraud} | \text{Flagged}, \text{Group A}) = P(\text{Fraud} | \text{Flagged}, \text{Group B})$.
    *   *Meaning:* If the model flags you, the probability you are actually a fraudster should be the same, regardless of race.

### 2.2 The Impossibility Theorem

*   **Result:** You cannot satisfy all fairness metrics simultaneously (unless the base rates are identical).
*   **Choice:** You must choose *which* definition of fairness matters most for your business.

---

## 3. Theoretical Properties

### 3.1 Proxy Discrimination

*   **Definition:** Using a variable that acts as a stand-in for a protected class.
*   **Examples:**
    *   *Zip Code:* Race.
    *   *Education Level:* Socioeconomic Status.
    *   *Shopping History:* Gender.
*   **Regulation:** Colorado Regulation 10-1-1 explicitly bans the use of external consumer data that results in unfair discrimination.

### 3.2 Human-in-the-Loop (HITL)

*   **Requirement:** High-risk AI decisions (like denying a claim) should not be fully automated.
*   **Role:** The AI provides a *recommendation* ("Refer to SIU"), but a human must make the final decision.
*   **Risk:** "Automation Bias" (The human just rubber-stamps the AI).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Measuring Bias in Python (AIF360 / Fairlearn)

```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Simulated Data
# Group 0: Majority, Group 1: Minority
group = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
# True Fraud Status (Base rates differ slightly)
y_true = np.random.binomial(1, p=np.where(group==0, 0.05, 0.06))
# Model Predictions (Biased against Group 1)
# If Group 1, we are more likely to flag them falsely
y_pred_prob = np.random.uniform(0, 1, 1000)
y_pred_prob += np.where((group==1) & (y_true==0), 0.2, 0) # Bias injection
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate False Positive Rate (FPR) by Group
def get_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)

fpr_0 = get_fpr(y_true[group==0], y_pred[group==0])
fpr_1 = get_fpr(y_true[group==1], y_pred[group==1])

print(f"FPR (Majority): {fpr_0:.1%}")
print(f"FPR (Minority): {fpr_1:.1%}")
print(f"Disparity Ratio: {fpr_1 / fpr_0:.2f}") 
# If Ratio > 1.2 or < 0.8, regulators get angry.
```

### 4.2 Adversarial Debiasing

*   **Technique:** Train two models simultaneously.
    1.  **Predictor:** Tries to predict Fraud.
    2.  **Adversary:** Tries to predict Race based on the Predictor's output.
*   **Goal:** The Predictor must minimize fraud error *while maximizing* the Adversary's error.
*   **Result:** A model that predicts fraud but contains no information about race.

---

## 5. Evaluation & Validation

### 5.1 The "Bifurcated" Test

*   **Method:** Evaluate model performance (AUC, LogLoss) separately for each protected class.
*   **Acceptance Criteria:** The AUC for the minority group must be within 5% of the majority group.

### 5.2 Counterfactual Testing

*   **Method:** Take a specific claim. Flip the "Gender" variable (or its proxies). Does the prediction change?
*   **Ideal:** The prediction should be invariant to demographic attributes.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 "We don't collect race data"

*   **Defense:** "We can't be racist because we don't know their race!"
*   **Reality:** This is the *worst* defense.
    *   If you don't collect it, you can't test for bias.
    *   Regulators (and Plaintiff Attorneys) *will* infer it using Bayesian Surname Geocoding (BISG) and prove you are biased.
*   **Best Practice:** Infer race for testing purposes (Testing & Monitoring), but do not use it as a feature.

### 6.2 The Accuracy-Fairness Trade-off

*   **Truth:** Removing biased features (like Zip Code) often reduces model accuracy.
*   **Business Decision:** Are you willing to accept a 2% drop in fraud detection to ensure fairness? (The answer should be "Yes").

---

## 7. Advanced Topics & Extensions

### 7.1 Explainable AI (XAI) - SHAP Values

*   **Tool:** SHAP (SHapley Additive exPlanations).
*   **Usage:** Shows exactly *which* features drove the score for *this specific* claim.
*   **Benefit:** Allows the investigator to validate the reason ("Oh, it's because the car was insured yesterday", not "Because he lives in Detroit").

### 7.2 The EU AI Act

*   **Classification:** Insurance pricing and claims systems are "High-Risk AI".
*   **Requirements:**
    *   Data Governance.
    *   Record Keeping.
    *   Transparency.
    *   Human Oversight.
    *   Robustness & Accuracy.
*   **Penalty:** Up to 30M Euro or 6% of global turnover.

---

## 8. Regulatory & Governance Considerations

### 8.1 Model Governance Framework (SR 11-7)

*   **Standard:** Federal Reserve guidelines for model risk (adopted by insurance).
*   **Three Lines of Defense:**
    1.  **Model Developers:** Build and test.
    2.  **Model Validation:** Independent review (challenge the bias).
    3.  **Internal Audit:** Verify the process.

---

## 9. Practical Example

### 9.1 The "Credit Score" Debate

**Issue:** Credit score is highly correlated with race. It is also highly predictive of insurance loss.
**Regulation:**
*   Some states (CA, MA) ban credit score in Auto pricing.
*   Others allow it but require "Extraordinary Life Circumstance" exceptions.
**Fraud Context:**
*   Using credit score to detect fraud is extremely risky. A poor credit score does not make someone a criminal.
*   **Recommendation:** Exclude credit-based features from SIU referral models.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Fairness is not optional.** It is becoming law.
2.  **Proxy Discrimination** is the main technical challenge.
3.  **Transparency (XAI)** builds trust with investigators and regulators.

### 10.2 When to Use This Knowledge
*   **Model Validation:** When auditing a vendor's "Black Box" fraud score.
*   **Compliance:** Preparing for a Market Conduct Exam.

### 10.3 Critical Success Factors
1.  **Documentation:** If you didn't document the fairness test, it didn't happen.
2.  **Culture:** Data Scientists must feel empowered to say "This model is biased, we cannot deploy it."

### 10.4 Further Reading
*   **Cathy O'Neil:** "Weapons of Math Destruction".
*   **NAIC:** "Principles on Artificial Intelligence".

---

## Appendix

### A. Glossary
*   **BISG:** Bayesian Improved Surname Geocoding (Method to infer race).
*   **Protected Class:** A group protected by anti-discrimination laws (Race, Color, Religion, Sex, National Origin).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Disparate Impact** | $\frac{P(\hat{Y}=1|G=1)}{P(\hat{Y}=1|G=0)}$ | Fairness Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
