# Intro to Triangles & Claims Development - Theoretical Deep Dive

## Overview
This session introduces the fundamental data structure of non-life reserving: the Loss Triangle. We explore the mechanics of claims development, the difference between Paid and Incurred data, and the core mathematics of the Chain Ladder Method (CLM) to estimate Incurred But Not Reported (IBNR) reserves.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Loss Triangle:** A two-dimensional array organizing claims data by:
*   **Origin Period (Rows):** When the loss occurred (Accident Year) or the policy was written (Policy Year).
*   **Development Period (Columns):** How much time has passed since the origin (Age in months/years).

**The "Run-Off" Concept:**
Insurance claims are not settled instantly. A claim occurring in 2020 might be reported in 2021, partially paid in 2022, and settled in 2025.
*   **Short-Tail:** Property, Auto Physical Damage (settles quickly).
*   **Long-Tail:** Workers' Comp, Medical Malpractice (settles over decades).

**Key Terminology:**
*   **Paid Losses:** Cash actually sent to claimants.
*   **Case Reserves:** Estimates set by claims adjusters for known open files.
*   **Reported (Incurred) Losses:** Paid + Case Reserves.
*   **IBNR:** Incurred But Not Reported (Pure IBNR + IBNER).
*   **Ultimate Loss:** The final cost when all claims are closed.

### 1.2 Historical Context & Evolution

**Origin:**
*   **Chain Ladder:** The origins are obscure, but it became the standard "heuristic" in the mid-20th century.
*   **Bornhuetter-Ferguson (1972):** Introduced a method to stabilize the Chain Ladder for immature years.

**Evolution:**
*   **Deterministic vs. Stochastic:** Early methods produced a single point estimate. Modern methods (Mack, Bootstrap) produce a distribution of reserves.
*   **Data Granularity:** Moving from aggregate triangles to individual claim-level modeling (micro-reserving).

**Current State:**
*   **Standard:** The Chain Ladder on Paid and Incurred triangles is still the primary method for regulatory reporting (Schedule P in the US).
*   **Machine Learning:** Emerging use of ML to predict individual claim development, rolling up to the triangle level.

### 1.3 Why This Matters

**Business Impact:**
*   **Financial Statements:** Reserves are the largest liability on a P&C insurer's balance sheet. A 1% error can wipe out a year's profit.
*   **Pricing:** If you underestimate reserves today, you are underpricing tomorrow's policies (because you think costs are lower than they are).

**Regulatory Relevance:**
*   **Solvency:** Regulators require "Best Estimate" reserves plus a risk margin.
*   **Taxes:** The IRS (in the US) has specific discounting rules for tax reserves based on payment patterns.

---

## 2. Mathematical Framework

### 2.1 The Chain Ladder Method (CLM)

**Assumption:** "The past is prologue." The proportional growth of losses from age $t$ to age $t+1$ will be similar to historical averages.

**Step 1: Calculate Age-to-Age Factors (Link Ratios)**
$$ f_{i,j} = \frac{C_{i, j+1}}{C_{i, j}} $$
*   $C_{i,j}$: Cumulative loss for Accident Year $i$ at Age $j$.
*   $f_{i,j}$: The growth factor.

**Step 2: Select Development Factors (LDFs)**
*   Calculate averages (Simple, Volume-Weighted, Medial) of the historical link ratios for each column.
*   Select a factor $\lambda_j$ for each age $j$.

**Step 3: Calculate Cumulative LDF (CDF)**
$$ CDF_j = \lambda_j \times \lambda_{j+1} \times \dots \times \lambda_{Ult} $$
*   This factor projects known losses to Ultimate.

**Step 4: Project Ultimate Loss**
$$ U_i = C_{i, \text{latest}} \times CDF_{\text{latest}} $$

**Step 5: Calculate Reserve**
$$ R_i = U_i - \text{Paid}_i $$

### 2.2 Paid vs. Incurred Chain Ladder

*   **Paid CLM:** Projects ultimate payments based on payment history. Stable but slow to react.
*   **Incurred CLM:** Projects ultimate incurred based on reported history. Faster to react but volatile due to case reserve changes.
*   **Best Practice:** Do both. If they diverge, investigate why (Case Reserve Strengthening?).

### 2.3 IBNR Decomposition

$$ \text{Total Reserve} = \text{Ultimate} - \text{Paid} $$
$$ \text{Case Reserve} = \text{Reported} - \text{Paid} $$
$$ \text{IBNR Reserve} = \text{Total Reserve} - \text{Case Reserve} $$
$$ \text{IBNR Reserve} = \text{Ultimate} - \text{Reported} $$

*   **Note:** In actuarial shorthand, "IBNR" often means "Total IBNR" (Pure IBNR + IBNER).
    *   **Pure IBNR:** Claims not yet reported.
    *   **IBNER:** Incurred But Not Enough Reported (Development on known claims).

### 2.4 Tail Factors

What happens after the triangle ends? (e.g., Triangle goes to 10 years, but claims last 30 years).
*   **Tail Factor:** A multiplier $> 1.0$ applied to the final column.
*   **Estimation:** Curve fitting (Inverse Power Curve, Exponential Decay) to the LDFs.

---

## 3. Theoretical Properties

### 3.1 Mack's Assumptions (Stochastic CLM)

1.  $E[C_{i, j+1} | C_{i, 1}, \dots, C_{i, j}] = f_j C_{i, j}$. (Linearity).
2.  Independence between accident years.
3.  Variance of link ratios is proportional to $1/C_{i,j}$.

### 3.2 Bias

*   **Upward Bias:** If the LDFs are correlated (e.g., high inflation affects all years), the simple CLM underestimates the uncertainty.
*   **Leverage:** A small change in the Tail Factor causes a massive change in the total reserve.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Transaction Data:** ClaimID, DateOfLoss, TransactionDate, Amount, Type (Paid/Reserve).
*   **Aggregation:** Sum transactions into cells $(AY, Age)$.

### 4.2 Preprocessing Steps

**Step 1: Cumulative vs. Incremental**
*   Triangles are usually built as **Incremental** first (sum of payments in the quarter), then converted to **Cumulative** for Chain Ladder.

**Step 2: Handling Negatives**
*   Salvage & Subrogation can cause negative incremental payments.
*   Chain Ladder fails if Cumulative values are zero or negative.

### 4.3 Model Specification (Python Example)

Building a triangle and applying the Chain Ladder Method.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulated Incremental Paid Data (Accident Years 2020-2023)
# Format: [AY, DevYear, Amount]
data_raw = [
    [2020, 1, 500], [2020, 2, 300], [2020, 3, 100], [2020, 4, 50],
    [2021, 1, 550], [2021, 2, 320], [2021, 3, 110],
    [2022, 1, 600], [2022, 2, 350],
    [2023, 1, 650]
]

df = pd.DataFrame(data_raw, columns=['AY', 'Dev', 'Incr'])

# 1. Pivot to Triangle (Incremental)
tri_incr = df.pivot(index='AY', columns='Dev', values='Incr')
print("Incremental Triangle:")
print(tri_incr.fillna(''))

# 2. Convert to Cumulative
tri_cum = tri_incr.cumsum(axis=1)
print("\nCumulative Triangle:")
print(tri_cum.fillna(''))

# 3. Calculate Link Ratios (Age-to-Age Factors)
link_ratios = pd.DataFrame(index=tri_cum.index[:-1], columns=tri_cum.columns[:-1])

for col in range(1, 4):
    # Loss at t+1 / Loss at t
    link_ratios[col] = tri_cum[col+1] / tri_cum[col]

print("\nLink Ratios:")
print(link_ratios.fillna(''))

# 4. Select LDFs (Volume Weighted Average)
# Sum(C_t+1) / Sum(C_t) for all available years
selected_ldfs = []
for col in range(1, 4):
    # Filter valid pairs
    valid_rows = link_ratios[col].dropna().index
    sum_next = tri_cum.loc[valid_rows, col+1].sum()
    sum_curr = tri_cum.loc[valid_rows, col].sum()
    ldf = sum_next / sum_curr
    selected_ldfs.append(ldf)

print(f"\nSelected LDFs (1-2, 2-3, 3-4): {[round(x, 3) for x in selected_ldfs]}")

# 5. Calculate CDFs (Cumulative Development Factors) to Ultimate
# Assume Tail Factor = 1.0 (No development after year 4)
tail = 1.0
cdfs = [1.0] * 4
cdfs[3] = tail # Year 4 to Ult
cdfs[2] = selected_ldfs[2] * cdfs[3] # Year 3 to Ult
cdfs[1] = selected_ldfs[1] * cdfs[2] # Year 2 to Ult
cdfs[0] = selected_ldfs[0] * cdfs[1] # Year 1 to Ult

print(f"CDFs to Ultimate: {[round(x, 3) for x in cdfs]}")

# 6. Project Ultimate and IBNR
results = pd.DataFrame(index=tri_cum.index)
results['Latest_Cum'] = tri_cum.ffill(axis=1).iloc[:, -1]
# Map CDF based on age. 
# 2023 is age 1 (needs cdfs[0]), 2022 is age 2 (needs cdfs[1]), etc.
results['CDF'] = [cdfs[0], cdfs[1], cdfs[2], cdfs[3]] 
results['Ultimate'] = results['Latest_Cum'] * results['CDF']
results['IBNR'] = results['Ultimate'] - results['Latest_Cum']

print("\nReserve Estimates:")
print(results)
print(f"\nTotal IBNR Needed: {results['IBNR'].sum():,.2f}")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **LDFs:** The growth pattern (e.g., 1.50, 1.10, 1.02).
2.  **Ultimate Loss:** The forecasted final cost.
3.  **IBNR:** The liability to book.

**Interpretation:**
*   **High LDFs:** Indicates a "long tail" or slow reporting pattern.
*   **Volatile Link Ratios:** Indicates instability in claims processing or random large losses.

---

## 5. Evaluation & Validation

### 5.1 Diagnostics

*   **Residual Plots:** Plot (Actual - Expected) development. Look for trends (e.g., calendar year effects like inflation).
*   **Actual vs. Expected (AvE):** Check how well last year's projection predicted this year's payments.

### 5.2 Sensitivity Testing

*   "What if the Tail Factor is 1.05 instead of 1.00?"
*   "What if we exclude the outlier year 2021?"

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Calendar Year Inflation**
    *   **Issue:** High inflation increases payments in a diagonal (Calendar Year) direction.
    *   **Reality:** Standard Chain Ladder projects *Accident Year* trends. It misses diagonal shifts.
    *   **Fix:** Use the Berquist-Sherman method or adjust data for inflation before triangulation.

2.  **Trap: Changing Case Reserve Adequacy**
    *   **Issue:** If claims adjusters suddenly start putting up higher initial reserves.
    *   **Result:** Incurred LDFs will look artificially high. The projection will overestimate Ultimate.
    *   **Fix:** Use the Berquist-Sherman adjustment for case reserve adequacy.

### 6.2 Implementation Challenges

1.  **Sparse Triangles:**
    *   In new lines of business, the triangle is empty.
    *   **Solution:** Use industry benchmark LDFs (Bornhuetter-Ferguson method).

---

## 7. Advanced Topics & Extensions

### 7.1 Bornhuetter-Ferguson (BF) Method

*   **Philosophy:** "Don't trust the Chain Ladder for green years."
*   **Formula:** $R = \text{Expected Loss} \times (1 - 1/CDF)$.
*   **Result:** Blends the stability of an *a priori* loss ratio with the responsiveness of the data.

### 7.2 Bootstrapping

*   Resampling the residuals of the Chain Ladder to generate a distribution of reserves.
*   Provides the "Risk Margin" or "Ranges" (e.g., 75th percentile reserve).

---

## 8. Regulatory & Governance Considerations

### 8.1 Schedule P (US)

*   The statutory filing where US insurers must report 10 years of triangles for every line of business.
*   **Scrutiny:** Analysts look for "Reserve Takedowns" (releasing reserves to boost profit) or "Deficiency" (under-reserving).

### 8.2 Statement of Actuarial Opinion (SAO)

*   The Appointed Actuary must sign a legal document stating the reserves are "Reasonable."
*   Requires considering "Risk of Material Adverse Deviation" (RMAD).

---

## 9. Practical Example

### 9.1 Worked Example: Selecting LDFs

**Scenario:**
*   Link Ratios for Age 12-24: [1.50, 1.45, 1.80, 1.48].
*   **Problem:** One outlier (1.80).
*   **Decision:**
    *   *Simple Average:* $(1.5+1.45+1.8+1.48)/4 = 1.56$.
    *   *Volume Weighted:* Might be 1.55.
    *   *Excluding High/Low:* $(1.50+1.48)/2 = 1.49$.
*   **Judgment:** If the 1.80 was a one-off hurricane, exclude it. If it represents a new trend (worsening legal environment), keep it (or weight it higher).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Triangles** organize data by Accident Year and Age.
2.  **Chain Ladder** projects future growth based on past growth.
3.  **IBNR** is the difference between Ultimate and Reported.

### 10.2 When to Use This Knowledge
*   **Reserving:** Quarterly financial closing.
*   **M&A:** Due diligence (checking if the target company is under-reserved).
*   **Pricing:** Estimating the ultimate loss cost to charge in premiums.

### 10.3 Critical Success Factors
1.  **Data Quality:** Garbage in, garbage out. Check for missing claims or coding errors.
2.  **Segmentation:** Don't mix Property (short tail) with Liability (long tail) in one triangle.
3.  **Judgment:** The math is easy; selecting the factors is the art.

### 10.4 Further Reading
*   **Friedland:** "Estimating Unpaid Claims Using Basic Techniques" (The CAS "Gold Book").
*   **Mack:** "Distribution-free Calculation of the Standard Error of Chain Ladder Reserve Estimates".

---

## Appendix

### A. Glossary
*   **LDF:** Loss Development Factor.
*   **CDF:** Cumulative Development Factor.
*   **IBNR:** Incurred But Not Reported.
*   **AY:** Accident Year.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Link Ratio** | $C_{j+1} / C_j$ | Growth Factor |
| **Ultimate** | $C_{latest} \times CDF$ | Projection |
| **Reserve** | $Ult - Paid$ | Liability |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
