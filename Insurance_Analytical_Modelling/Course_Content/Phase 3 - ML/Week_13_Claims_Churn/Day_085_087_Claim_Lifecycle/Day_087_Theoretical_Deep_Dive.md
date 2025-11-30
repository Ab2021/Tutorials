# Claim Lifecycle & Severity Development (Part 3) - Inflation, Trends & On-Leveling - Theoretical Deep Dive

## Overview
"A dollar today is not a dollar tomorrow."
In insurance, we price policies today to pay claims 5 years from now.
If we miss the **Inflation Trend**, our premiums will be insufficient.
This day focuses on **Economic Inflation**, **Social Inflation**, and the actuarial techniques of **Trending** and **On-Leveling**.

---

## 1. Conceptual Foundation

### 1.1 Types of Inflation

1.  **Economic Inflation (CPI):** General rise in prices.
2.  **Medical Inflation:** Rises faster than CPI. (Crucial for Bodily Injury).
3.  **Wage Inflation:** Affects Workers' Comp (Indemnity benefits linked to wages).
4.  **Social Inflation:** The increase in claims costs due to changing societal views, jury verdicts, and legal environments. (Hardest to predict).

### 1.2 The Trending Problem

*   **Data:** Historical claims from 2018-2023.
*   **Goal:** Predict claims for policies written in 2025.
*   **Gap:** We need to project 2018 claims $\to$ 2025 cost levels (7 years of trend).

---

## 2. Mathematical Framework

### 2.1 Exponential Trend Model

$$ Y_t = Y_0 e^{rt} $$
*   **$Y_t$:** Claim Severity at time $t$.
*   **$r$:** Annual Trend Rate (e.g., 0.05 for 5%).
*   **Fitting:** Log-linear regression on historical average severities. $\ln(Y_t) = \ln(Y_0) + rt$.

### 2.2 On-Leveling Premiums

*   **Concept:** We must also adjust historical *premiums* to current rates.
*   **Why?** If we raised rates by 10% in 2020, the 2019 premiums look artificially low compared to today's rate manual.
*   **Method:** Extension of Exposures. Recalculate what the 2019 policy *would have cost* if sold today.

---

## 3. Theoretical Properties

### 3.1 Leveraged Trend

*   **Scenario:** Excess of Loss layer (>\$100k).
*   **Effect:** A 5% trend on the Ground Up loss causes a >5% trend on the Excess layer.
*   **Formula:**
    $$ \text{Excess Trend} \approx \text{Ground Up Trend} \times \frac{\text{Mean Ground Up}}{\text{Mean Excess}} $$
    (Approximation: Leverage is always > 1).

### 3.2 Step vs. Continuous Trend

*   **Continuous:** Inflation happens every day.
*   **Step:** Law change (e.g., "Minimum Benefit increased on Jan 1st").
*   **Modeling:** Use dummy variables for Step changes, time variable for Continuous.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Estimating Trend (Python Statsmodels)

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Data: Quarterly Average Severity
# Date, AvgSev
df['Time_Index'] = np.arange(len(df)) # 0, 1, 2...

# Log-Linear Model
X = sm.add_constant(df['Time_Index'])
y = np.log(df['AvgSev'])

model = sm.OLS(y, X).fit()

# Extract Trend
slope = model.params['Time_Index']
annual_trend = np.exp(slope * 4) - 1 # Assuming quarterly data

print(f"Annual Severity Trend: {annual_trend:.2%}")
```

### 4.2 Detecting Trend Shifts (Chow Test)

*   **Question:** Did the trend change after COVID (2020)?
*   **Test:** Fit two separate lines (Pre-2020, Post-2020) and compare RSS (Residual Sum of Squares) vs. a single line.

---

## 5. Evaluation & Validation

### 5.1 Residual Analysis

*   **Check:** Plot residuals of the trend model over time.
*   **Pattern:** If residuals show a "U" shape, the trend is accelerating (Quadratic). If they show a sine wave, there is Seasonality.

### 5.2 External Validation

*   **Benchmark:** Compare your internal trend (5%) vs. Industry Indices (Fast Track, ISO, Masterson Index).
*   **Gap:** If you see 2% and the industry sees 8%, you might be under-reserving.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Mix Shift vs. Trend

*   **Illusion:** Average severity dropped by 10%.
*   **Reality:** We stopped writing high-risk sports cars and started writing minivans (Mix Shift).
*   **Fix:** Analyze trend on "Case Mix Adjusted" severity (using GLM residuals).

### 6.2 Calendar vs. Accident Year Trend

*   **Accident Year:** Trend in the *occurrence* of accidents (e.g., cars getting safer).
*   **Calendar Year:** Trend in the *payment* (e.g., inflation).
*   **Interaction:** Both happen simultaneously.

---

## 7. Advanced Topics & Extensions

### 7.1 Social Inflation Modeling

*   **Proxy:** Use "Attorney Representation Rate" as a leading indicator.
*   **Model:**
    $$ \text{Sev}_t = \beta_0 + \beta_1 \cdot \text{CPI}_t + \beta_2 \cdot \text{AttorneyRate}_t $$

### 7.2 Stochastic Trend

*   **Method:** Time Series (ARIMA).
*   **Forecast:** Instead of a fixed 5%, we predict "5% $\pm$ 2%".
*   **Impact:** Increases the width of the reserve distribution.

---

## 8. Regulatory & Governance Considerations

### 8.1 Rate Filing Justification

*   **Requirement:** You cannot just say "I feel like 10%". You must show the regression.
*   **Constraint:** Some states cap the allowable trend factor.

---

## 9. Practical Example

### 9.1 The "Lumber Price" Spike

**Scenario:** Property Insurance 2021.
**Event:** Lumber prices tripled.
**Model:** Standard CPI trend (2%) failed.
**Action:**
1.  Switch to a "Construction Cost Index" (CCI) for trending.
2.  Apply a "Shock Load" to the trend factor for 2021-2022.
**Result:** Premiums increased 15% to match the actual cost of rebuilding homes.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Trend** projects past to future.
2.  **On-Leveling** adjusts past premiums to present.
3.  **Social Inflation** is the silent killer of liability books.

### 10.2 When to Use This Knowledge
*   **Ratemaking:** Determining next year's base rate.
*   **Reserving:** Selecting LDFs (Tail factors often include inflation).

### 10.3 Critical Success Factors
1.  **Data Consistency:** Ensure definitions (e.g., "Paid Loss") haven't changed over the trend period.
2.  **External Indices:** Don't rely solely on internal data if your volume is low.

### 10.4 Further Reading
*   **Werner & Modlin:** "Basic Ratemaking" (Chapter on Trending).
*   **Swiss Re Sigma:** Reports on Social Inflation.

---

## Appendix

### A. Glossary
*   **Frequency Trend:** Change in claims per exposure.
*   **Severity Trend:** Change in cost per claim.
*   **Pure Premium Trend:** Combined effect ($\approx$ Freq Trend + Sev Trend).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Trend Factor** | $(1+r)^t$ | Projection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
