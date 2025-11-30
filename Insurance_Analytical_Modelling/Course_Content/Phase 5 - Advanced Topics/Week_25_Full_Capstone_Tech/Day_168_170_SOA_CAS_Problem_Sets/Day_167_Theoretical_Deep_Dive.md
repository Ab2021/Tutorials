# Model Risk Management & Documentation (Part 3) - Production Monitoring & Drift - Theoretical Deep Dive

## Overview
"The model was perfect on Day 1. On Day 100, it's a liability."
Models degrade. Not because the code rots, but because the world changes.
This day focuses on **Production Monitoring**: The dashboard you look at every morning to ensure your model hasn't gone rogue.
We cover **Data Drift**, **Concept Drift**, and the **PSI** metric.

---

## 1. Conceptual Foundation

### 1.1 The Drift Taxonomy

1.  **Data Drift (Covariate Shift):** $P(X)$ changes.
    *   *Example:* Inflation spikes. The distribution of "Vehicle Value" shifts right. The model has never seen a \$100k Toyota Camry before.
2.  **Concept Drift:** $P(Y|X)$ changes.
    *   *Example:* Covid lockdowns. The relationship between "Miles Driven" and "Accident Probability" changes (roads are empty, people drive faster).
3.  **Label Drift:** $P(Y)$ changes.
    *   *Example:* A sudden spike in fraud claims due to a new organized ring.

### 1.2 The "Silent Failure"

*   **Software Bug:** The code crashes (Exception). You know immediately.
*   **Model Decay:** The model runs perfectly, but the predictions become slightly less accurate every day. You don't know until you lose money.

---

## 2. Mathematical Framework

### 2.1 Population Stability Index (PSI)

The industry standard for measuring distributional shift.
$$ PSI = \sum_{i=1}^{B} (P_{actual, i} - P_{expected, i}) \times \ln\left(\frac{P_{actual, i}}{P_{expected, i}}\right) $$
*   **Buckets ($B$):** Usually deciles (10 buckets).
*   **Thresholds:**
    *   $PSI < 0.1$: No significant change.
    *   $0.1 \le PSI < 0.25$: Moderate change (Investigate).
    *   $PSI \ge 0.25$: Major shift (Retrain model).

### 2.2 Kullback-Leibler (KL) Divergence

*   **Definition:** A measure of how one probability distribution differs from a second, reference probability distribution.
*   **Relation to PSI:** $PSI$ is essentially the symmetric version of KL Divergence.

---

## 3. Theoretical Properties

### 3.1 Characteristic Stability Index (CSI)

*   **PSI:** Measures shift in the *Score* (Output).
*   **CSI:** Measures shift in the *Features* (Inputs).
*   **Workflow:**
    1.  Check PSI. If High $\rightarrow$
    2.  Check CSI for all features. Find the culprit (e.g., "Credit Score" distribution shifted).

### 3.2 Performance Estimation without Labels

*   **Problem:** In insurance, we don't know the "True Label" (Ultimate Loss) for years.
*   **Solution:** **CBPE (Confidence-Based Performance Estimation)**.
    *   If the model is well-calibrated, the average predicted probability should match the observed event rate.
    *   If the input data drifts into regions where the model is uncertain, the estimated performance drops.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Calculating PSI in Python

```python
import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    # Define breakpoints based on Expected (Training) data
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Calculate counts in each bucket
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]
    
    # Normalize to probabilities
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)
    
    return psi

# Example
train_scores = np.random.normal(0, 1, 1000)
prod_scores = np.random.normal(0.2, 1.2, 1000) # Drifted mean and variance

psi_score = calculate_psi(train_scores, prod_scores)
print(f"PSI: {psi_score:.4f}")
if psi_score > 0.25:
    print("ALERT: Model Drift Detected!")
```

### 4.2 Monitoring Dashboard (Concept)

*   **Daily View:**
    *   Traffic Volume (Did we get 0 quotes today?).
    *   Missing Value Rate (Did a data feed break?).
    *   PSI of Final Score.
*   **Weekly View:**
    *   CSI of Top 10 Features.
    *   Segment Analysis (Is the drift isolated to California?).

---

## 5. Evaluation & Validation

### 5.1 The "Retraining Trigger"

*   **Manual:** Actuary reviews PSI monthly and decides.
*   **Automated:** If $PSI > 0.2$ for 3 consecutive days, trigger the Airflow pipeline to retrain the model on the latest window.
*   **Risk:** Feedback Loops. If the model is biased, retraining on its own decisions reinforces the bias.

### 5.2 Window Strategy

*   **Fixed Window:** Train on Jan-Dec 2023.
*   **Sliding Window:** Train on last 12 months.
*   **Expanding Window:** Train on all history.
*   **Insurance Preference:** Sliding Window (usually 3-5 years) to capture recent trends while maintaining stability.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Seasonality vs. Drift

*   **Scenario:** Claims spike in December (Snow).
*   **False Alarm:** PSI spikes.
*   **Reality:** This is expected seasonality, not model decay.
*   **Fix:** Compare Dec 2024 vs. Dec 2023 (Year-over-Year), not Dec vs. Nov.

### 6.2 Data Quality vs. Data Drift

*   **Scenario:** "Age" variable shifts from Mean=40 to Mean=0.
*   **Diagnosis:**
    *   *Drift:* We started marketing to teenagers.
    *   *Quality:* The web form broke and defaults to 0.
*   **Action:** Always check Data Quality (Great Expectations) *before* checking Drift.

---

## 7. Advanced Topics & Extensions

### 7.1 Adversarial Drift

*   **Context:** Fraud Detection.
*   **Mechanism:** Fraudsters learn the model. They stop using "Suspicious IP Addresses".
*   **Result:** The model's performance drops, but PSI might look stable (because the fraudsters are mimicking normal users).
*   **Solution:** Active Learning / Anomaly Detection.

### 7.2 Shadow Models

*   **Strategy:** Run the "Challenger" model in production (shadow mode).
*   **Monitor:** If Challenger outperforms Champion for 2 weeks, swap them.

---

## 8. Regulatory & Governance Considerations

### 8.1 SS1/23 Requirement

*   **Mandate:** You must have a "Remediation Plan" for drift.
*   **Question:** "If the model breaks, what is the fallback?"
*   **Answer:** "We revert to the manual rating table from 2022."

---

## 9. Practical Example

### 9.1 The "Inflation Shock"

**Scenario:**
*   Auto Severity Model trained on 2018-2020 data.
*   2021: Used Car prices spike 30%.
*   **Impact:**
    *   Model predicts \$5,000 severity.
    *   Actual severity is \$6,500.
    *   PSI on "Vehicle Value" feature spikes.
**Action:**
*   **Short Term:** Apply a scalar adjustment (+30%) to the output.
*   **Long Term:** Retrain with recent data and add "CPI_Used_Cars" as a feature.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **PSI** is the smoke detector.
2.  **Data Drift** happens more often than Concept Drift.
3.  **Monitoring** is a continuous process, not a one-time check.

### 10.2 When to Use This Knowledge
*   **MLOps:** Setting up the production pipeline.
*   **Auditing:** "Show me your monitoring logs for last month."

### 10.3 Critical Success Factors
1.  **Alert Fatigue:** Don't alert on PSI > 0.1001. Tune the thresholds so you only wake up for real problems.
2.  **Root Cause Analysis:** Knowing *that* it drifted is easy. Knowing *why* is hard.

### 10.4 Further Reading
*   **Evidently AI:** "Machine Learning Monitoring" (Blog/Docs).
*   **NannyML:** "Confidence-Based Performance Estimation".

---

## Appendix

### A. Glossary
*   **Covariate Shift:** Change in the distribution of independent variables ($X$).
*   **Prior Probability Shift:** Change in the distribution of the target variable ($Y$).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PSI** | $\sum (P_a - P_e) \ln(P_a/P_e)$ | Drift Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
