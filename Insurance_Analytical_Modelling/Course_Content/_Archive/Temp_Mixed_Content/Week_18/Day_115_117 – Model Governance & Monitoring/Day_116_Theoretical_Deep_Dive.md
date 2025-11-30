# Model Governance & Monitoring (Part 2) - Model Monitoring & Drift Detection - Theoretical Deep Dive

## Overview
"A model is like a car. It needs regular maintenance."
You deployed the model. It works. But the world changes. Inflation rises, laws change, customer behavior shifts.
**Model Monitoring** is the dashboard that tells you when the "Check Engine" light is on.

---

## 1. Conceptual Foundation

### 1.1 The Two Types of Drift

1.  **Data Drift (Covariate Shift):** $P(X)$ changes.
    *   *Example:* Your model was trained on drivers aged 30-50. Suddenly, a marketing campaign attracts 18-year-olds. The input distribution has shifted.
2.  **Concept Drift:** $P(Y|X)$ changes.
    *   *Example:* Inflation hits. A repair that cost \$1000 last year now costs \$1500. The relationship between "Damage Severity" and "Cost" has changed.

### 1.2 Stability Indices

*   **PSI (Population Stability Index):** Measures Data Drift for the *Score* (Output).
*   **CSI (Characteristic Stability Index):** Measures Data Drift for the *Features* (Input).

---

## 2. Mathematical Framework

### 2.1 Calculating PSI

$$ \text{PSI} = \sum_{i=1}^{B} (A_i - E_i) \ln \left( \frac{A_i}{E_i} \right) $$

*   $B$: Number of bins (usually 10 deciles).
*   $A_i$: Actual % of population in bin $i$ (Current).
*   $E_i$: Expected % of population in bin $i$ (Training).
*   *Interpretation:*
    *   **< 0.1:** Stable (Green).
    *   **0.1 - 0.2:** Slight Drift (Amber).
    *   **> 0.2:** Significant Drift (Red). Action required.

### 2.2 Kullback-Leibler (KL) Divergence

*   PSI is essentially a symmetric version of KL Divergence.
*   It measures the "distance" between two probability distributions.

---

## 3. Theoretical Properties

### 3.1 Seasonality vs. Drift

*   **Scenario:** Claims frequency drops in Winter (fewer cars on road).
*   **Is it Drift?** No, it's seasonality.
*   **Fix:** Compare "This Month" vs "Same Month Last Year" (YoY), not "This Month" vs "Last Month".

### 3.2 Feedback Loops

*   **Scenario:** Your model predicts high risk for young drivers -> You charge them more -> They leave -> Your portfolio has fewer young drivers -> The model sees a shift.
*   **Self-Fulfilling Prophecy:** The model changes the world it observes.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Python Implementation (Manual PSI)

```python
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    # 1. Define Buckets based on Expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # 2. Calculate Counts
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # 3. Handle Zeros (to avoid log(0))
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # 4. Compute PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)
```

### 4.2 Evidently AI

*   Open-source library for drift detection.
*   Generates HTML reports showing "Drifted Features".

---

## 5. Evaluation & Validation

### 5.1 Performance Monitoring

*   **Ground Truth Delay:** In insurance, you don't know the "True Loss" for years.
*   **Proxy Metrics:**
    *   **Drift:** Immediate.
    *   **First Notice of Loss (FNOL):** Weeks.
    *   **Ultimate Loss:** Years.
*   *Strategy:* Monitor Drift daily. Monitor FNOL monthly. Monitor Ultimate annually.

### 5.2 Threshold Tuning

*   **Too Sensitive:** You retrain every week. (Expensive, unstable).
*   **Too Loose:** You miss a crisis.
*   *Best Practice:* Set thresholds based on historical volatility.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Retraining on Drifted Data**
    *   "Data has drifted. Let's retrain."
    *   *Risk:* If the drift is temporary (e.g., COVID lockdown), retraining locks in a temporary anomaly.
    *   *Fix:* Wait for stability or use a "Weighting" scheme.

2.  **Trap: Monitoring Averages Only**
    *   "Average Credit Score is still 700."
    *   *Reality:* The variance increased. You have more 800s and more 600s.
    *   *Fix:* Monitor the *Distribution* (PSI), not just the Mean.

---

## 7. Advanced Topics & Extensions

### 7.1 Adversarial Drift

*   **Fraud:** Fraudsters actively try to evade the model.
*   They probe the decision boundary.
*   *Detection:* Look for spikes in samples *just below* the rejection threshold.

### 7.2 Online Learning

*   Algorithms that update *continuously* (e.g., Contextual Bandits).
*   **Monitoring:** Requires real-time dashboards (Grafana).

---

## 8. Regulatory & Governance Considerations

### 8.1 SR 11-7 Ongoing Monitoring

*   "Validation is not a one-time event."
*   You must produce a **Quarterly Monitoring Report**.
    *   Contents: PSI, CSI, Gini Coefficient, Override Rates.

---

## 9. Practical Example

### 9.1 Worked Example: The Recession

**Scenario:**
*   **Model:** Credit Default Model trained in 2019 (Boom times).
*   **Event:** 2020 Recession.
*   **Monitoring:**
    *   **CSI (Unemployment):** Spikes. (Data Drift).
    *   **PSI (Score):** The model predicts massive defaults.
    *   **Reality:** Government stimulus checks prevent defaults. (Concept Drift - The relationship broke).
*   **Action:**
    *   The model is wrong.
    *   **Override:** Apply a "Scalar Adjustment" to lower the predicted default rate until the economy stabilizes.
    *   **Do NOT Retrain:** The data is chaotic.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **PSI** measures distributional shift.
2.  **Data Drift** is input change; **Concept Drift** is relationship change.
3.  **Monitoring** bridges the gap between "Deployed" and "Retired".

### 10.2 When to Use This Knowledge
*   **Production:** Every deployed model needs a monitor.
*   **Crisis Management:** When the world changes, look at the PSI.

### 10.3 Critical Success Factors
1.  **Automation:** Don't calculate PSI in Excel. Automate it.
2.  **Alerting:** Send an email when PSI > 0.2.

### 10.4 Further Reading
*   **NannyML:** "The Ultimate Guide to Model Monitoring".

---

## Appendix

### A. Glossary
*   **Covariate Shift:** Another name for Data Drift.
*   **Prior Probability Shift:** Change in the target variable distribution $P(Y)$.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PSI** | $\sum (A-E) \ln(A/E)$ | Drift Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
