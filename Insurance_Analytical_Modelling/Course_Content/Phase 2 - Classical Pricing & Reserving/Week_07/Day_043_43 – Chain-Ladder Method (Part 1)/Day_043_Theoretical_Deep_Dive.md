# Reserving Fundamentals (Chain Ladder) - Theoretical Deep Dive

## Overview
We now enter the second half of Phase 2: **Reserving**. While Pricing asks "How much should we charge?", Reserving asks "How much do we owe?". The **Chain Ladder Method (CLM)** is the cornerstone of actuarial reserving. It relies on the assumption that past development patterns will predict future development. We explore **Run-off Triangles**, **Link Ratios**, and the calculation of **IBNR** (Incurred But Not Reported).

---

## 1. Conceptual Foundation

### 1.1 The Reserving Problem

**The Timeline of a Claim:**
1.  **Occurrence:** Accident happens (Jan 1).
2.  **Reporting:** Insurer is notified (Jan 15).
3.  **Payment:** Mechanic is paid (Feb 1).
4.  **Closure:** File closed (Feb 1).
5.  **Reopening:** Neck pain returns (June 1).

**The Liability:**
At any point in time (e.g., Dec 31), the insurer owes money for:
*   **Case Reserves:** Claims we know about but haven't fully paid.
*   **IBNR (Incurred But Not Reported):** Claims that happened but haven't been reported yet (Pure IBNR) + Future development on known claims (IBNER - Incurred But Not Enough Reported).

### 1.2 The Run-off Triangle

A matrix organizing data by **Accident Year** (Rows) and **Development Age** (Columns).
*   **Cell (i, j):** Cumulative Loss for Accident Year $i$ at Age $j$.
*   **The Shape:** We know the "Upper Triangle" (Past). We need to fill the "Lower Triangle" (Future).

### 1.3 The Chain Ladder Assumption

**"History Repeats Itself."**
*   If claims typically grow by 10% between Age 12 and Age 24, we assume the current Age 12 claims will also grow by 10%.
*   **Key Risk:** If the claims handling process changes (e.g., we pay claims faster now), the assumption breaks.

---

## 2. Mathematical Framework

### 2.1 Link Ratios (Age-to-Age Factors)

Let $C_{i,j}$ be the cumulative loss.
The **Individual Link Ratio** for Accident Year $i$ from Age $j$ to $j+1$ is:
$$ F_{i, j} = \frac{C_{i, j+1}}{C_{i, j}} $$

### 2.2 Selecting Development Factors (LDFs)

We need one factor $\hat{f}_j$ to represent the growth from Age $j$ to $j+1$.
Common selections:
*   **Simple Average:** $\frac{1}{n} \sum F_{i,j}$.
*   **Volume-Weighted Average:** $\frac{\sum C_{i, j+1}}{\sum C_{i, j}}$. (Preferred).
*   **Excluding High/Low:** Trimmed average to remove outliers.

### 2.3 The Projection

1.  **Cumulative LDF (CDF):** Product of age-to-age factors.
    $$ CDF_j = \hat{f}_j \times \hat{f}_{j+1} \times \dots \times \hat{f}_{ult} $$
2.  **Ultimate Loss ($U_i$):**
    $$ U_i = C_{i, current} \times CDF_{current} $$
3.  **Reserves ($R_i$):**
    $$ R_i = U_i - \text{Paid To Date}_i $$

---

## 3. Theoretical Properties

### 3.1 Mack's Model (Stochastic CLM)

Thomas Mack (1993) formalized the CLM as a regression model.
*   **Assumption 1:** $E[C_{i, j+1} | C_{i, j}] = f_j C_{i, j}$. (Linear relationship).
*   **Assumption 2:** Independence between accident years.
*   **Variance:** $\text{Var}(C_{i, j+1} | C_{i, j}) = \sigma_j^2 C_{i, j}$.

### 3.2 The Tail Factor

Triangles are finite (e.g., 10 years). Liability claims can last 30 years.
*   **Tail Factor:** Represents growth from Age 120 months to $\infty$.
*   **Estimation:** Curve fitting (Inverse Power Curve) or Industry Benchmarks.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Building a Triangle in Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Raw Data: Claim Transaction Log
data = pd.DataFrame({
    'AccidentYear': [2020, 2020, 2020, 2021, 2021, 2022],
    'DevelopmentLag': [12, 24, 36, 12, 24, 12],
    'IncrementalPaid': [100, 50, 10, 110, 60, 120]
})

# 1. Pivot to Triangle
triangle = data.pivot_table(
    index='AccidentYear', 
    columns='DevelopmentLag', 
    values='IncrementalPaid', 
    aggfunc='sum'
).cumsum(axis=1) # Cumulative Sum

print("Cumulative Triangle:")
print(triangle)

# 2. Calculate Link Ratios
link_ratios = pd.DataFrame()
cols = triangle.columns
for i in range(len(cols)-1):
    col_curr = cols[i]
    col_next = cols[i+1]
    # Ratio = Next / Curr
    link_ratios[f'{col_curr}-{col_next}'] = triangle[col_next] / triangle[col_curr]

print("\nLink Ratios:")
print(link_ratios)

# 3. Select LDFs (Volume Weighted Average)
selected_ldfs = []
for i in range(len(cols)-1):
    col_curr = cols[i]
    col_next = cols[i+1]
    # Sum(Next) / Sum(Curr)
    # Filter out NaNs (where Next is not yet observed)
    mask = triangle[col_next].notna()
    vwa = triangle.loc[mask, col_next].sum() / triangle.loc[mask, col_curr].sum()
    selected_ldfs.append(vwa)

print("\nSelected LDFs:", [f"{x:.3f}" for x in selected_ldfs])

# 4. Calculate Ultimate
# Assume Tail Factor = 1.0
cdfs = [np.prod(selected_ldfs[i:]) for i in range(len(selected_ldfs))]
# Add 1.0 for the last column
cdfs.append(1.0) 

latest_diagonal = triangle.ffill(axis=1).iloc[:, -1] # Simplified logic
# In reality, you pick the diagonal based on Age.
```

### 4.2 Diagnostic Plots

1.  **Residual Plot:** Plot $(C_{i, j+1} - f_j C_{i, j})$ against $C_{i, j}$.
    *   Should be random noise. Patterns indicate the Chain Ladder assumptions are violated.
2.  **Heatmap of Link Ratios:**
    *   Color code high/low ratios.
    *   **Calendar Year Effect:** A diagonal of "High" ratios suggests inflation or a change in payment speed.

---

## 5. Evaluation & Validation

### 5.1 Actual vs. Expected (A vs. E)

*   Project last year's reserve to today.
*   Compare to what actually happened.
*   **Adverse Development:** If Actual > Expected, we under-reserved.
*   **Redundancy:** If Actual < Expected, we over-reserved (profit release).

### 5.2 Sensitivity Testing

*   "What if the Tail Factor is 1.05 instead of 1.02?"
*   "What if we exclude the 2018 outlier?"
*   **Range of Reasonable Estimates:** Actuaries rarely give a single number. They give a range.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Leverage" Effect**
    *   **Issue:** Small changes in LDFs (e.g., 1.02 to 1.03) can cause massive changes in IBNR for recent years.
    *   **Reality:** The most recent years have the highest leverage because the CDF is large (e.g., 5.0).

2.  **Trap: Changing Mix of Business**
    *   **Issue:** If you merge "Auto Liability" (Slow) and "Auto Physical Damage" (Fast) into one triangle.
    *   **Result:** If the mix shifts towards Liability, the triangle will look like it's "slowing down," leading to massive over-projection if not handled.

### 6.2 Implementation Challenges

1.  **Negative Incremental Losses:**
    *   Salvage and Subrogation (S&S) can cause total paid to drop.
    *   Link ratios become $< 1$.
    *   If Cumulative becomes negative (rare), Chain Ladder fails (division by zero or sign flip).

---

## 7. Advanced Topics & Extensions

### 7.1 Bornhuetter-Ferguson (BF) Method

*   **Problem:** Chain Ladder is unstable for the most recent accident year (Age 12). A random large claim gets multiplied by 10.0.
*   **Solution:** Use an *A Priori* Loss Ratio for the IBNR piece.
*   *More on this in Day 44.*

### 7.2 Berquist-Sherman Adjustments

*   Adjusting the triangle for changes in:
    *   Claim Settlement Rate (Speedups/Slowdowns).
    *   Case Reserve Adequacy (Strengthening/Weakening).

---

## 8. Regulatory & Governance Considerations

### 8.1 Statement of Actuarial Opinion (SAO)

*   The Appointed Actuary must sign a legal document stating reserves are "Reasonable."
*   **Personal Liability:** Actuaries can be sued if reserves are grossly insufficient due to negligence.

### 8.2 Solvency II / IFRS 17

*   Requires **Discounting** of reserves (Time Value of Money).
*   Requires a **Risk Margin** (Cost of Capital).

---

## 9. Practical Example

### 9.1 Worked Example: IBNR Calculation

**Data:**
*   Accident Year 2023.
*   Paid to Date: \$10M.
*   Selected LDF (12-Ult): 2.50.

**Calculation:**
1.  **Ultimate Loss:** $10M \times 2.50 = \$25M$.
2.  **Total Reserve:** $25M - 10M = \$15M$.
3.  **Split:**
    *   If Case Reserves = \$8M.
    *   **IBNR** = Total Reserve - Case Reserve = $15M - 8M = \$7M$.

**Interpretation:**
*   We expect to pay \$8M on known claims and \$7M on unknown/development claims.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Triangles** are the map.
2.  **LDFs** are the compass.
3.  **IBNR** is the destination.

### 10.2 When to Use This Knowledge
*   **Quarterly Reserving:** The heartbeat of the insurance finance team.
*   **M&A:** Valuing the liabilities of a target company.

### 10.3 Critical Success Factors
1.  **Smooth the Data:** Don't chase every wiggle in the triangle.
2.  **Know the Story:** Did the Claims Department hire 50 new adjusters? That changes the triangle.
3.  **Validate:** Check the implied Loss Ratios.

### 10.4 Further Reading
*   **Friedland:** "Estimating Unpaid Claims Using Basic Techniques".
*   **Mack (1993):** "Distribution-free Calculation of the Standard Error".

---

## Appendix

### A. Glossary
*   **Development Age:** Months since the beginning of the accident year.
*   **Valuation Date:** The date the data snapshot was taken.
*   **Incremental vs. Cumulative:** Daily payments vs. Total-to-date.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Link Ratio** | $C_{j+1} / C_j$ | Development |
| **Ultimate** | $C_{curr} \times \Pi f_j$ | Projection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
