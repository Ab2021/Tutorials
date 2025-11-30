# Non-life Reserving Case Study (Part 1) - Theoretical Deep Dive

## Overview
Reserving is more than just applying the Chain Ladder method. It is an end-to-end process that starts with raw data extraction and ends with a signed Statement of Actuarial Opinion (SAO). This session covers the **Reserving Cycle**, **Data Diagnostics**, and the art of **Method Selection**.

---

## 1. Conceptual Foundation

### 1.1 The Reserving Cycle

1.  **Data Extraction:** Pulling Paid, Incurred, and Claim Counts from the warehouse.
2.  **Reconciliation:** Ensuring Data = General Ledger.
3.  **Diagnostics:** Checking for changes in mix, settlement speed, or case reserving strength.
4.  **Method Selection:** Choosing the right tool (CL, BF, Cape Cod) for the job.
5.  **Selection:** Picking the final Ultimate Loss for each Accident Year.
6.  **Reporting:** IBNR = Ultimate - Reported.

### 1.2 Data Issues & Cleaning

*   **Missing Values:** "0" vs. "Null". (Is the claim closed with \$0 pay, or is the data missing?).
*   **Outliers:** A single \$10M claim can distort the LDFs.
    *   *Action:* Remove the large loss, project the "Attritional" losses, and add a separate "Large Loss Load".
*   **Coding Changes:** "Bodily Injury" was Code 1 in 2010 and Code 5 in 2020. You must map them.

### 1.3 Method Selection Framework

| Method | When to Use | Weakness |
| :--- | :--- | :--- |
| **Chain Ladder (Paid)** | Stable payment patterns. | Unstable if settlement speed changes. |
| **Chain Ladder (Incurred)** | Stable case reserving. | Distorted by "Case Strengthening". |
| **Bornhuetter-Ferguson** | Immature years (Green). | Relies on the "A Priori" Loss Ratio. |
| **Cape Cod** | Immature years (Data-driven). | Needs a stable trend. |
| **Frequency-Severity** | When inflation is high. | Requires reliable count data. |

---

## 2. Mathematical Framework

### 2.1 The "A Priori" in BF

$$ U_{BF} = \text{Actual} + (1 - \frac{1}{LDF}) \times \text{Expected Loss} $$
*   Where does "Expected Loss" come from?
    *   **Pricing LR:** The loss ratio assumed when the policy was sold.
    *   **Industry LR:** Benchmarks (e.g., Schedule P).
    *   **Trended Pure Premium:** Last year's ultimate $\times$ Trend.

### 2.2 Cape Cod (Stanard-Bühlmann)

*   Calculates the "Expected Loss" from the data itself.
*   $$ ELR = \frac{\sum \text{Used Loss}}{\sum \text{Used Premium}} $$
*   *Used Premium:* Premium adjusted for development ($\text{Premium} \times \frac{1}{LDF}$).
*   *Advantage:* More responsive than BF, more stable than CL.

---

## 3. Theoretical Properties

### 3.1 The "Leverage" Effect

*   In the Chain Ladder method, the LDFs are multiplicative.
*   A small change in the 12-24 factor affects *all* subsequent years.
*   **Tail Factor:** The most leveraged assumption. A 1.05 tail vs. a 1.10 tail can double the IBNR.

### 3.2 Case Reserve Adequacy

*   **Metric:** `Average Case Reserve = Total Case / Open Counts`.
*   **Trend:** If Average Case Reserve is growing faster than inflation, the claims department is strengthening reserves.
*   **Impact:** Incurred LDFs will increase. Incurred CL will over-project.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Automated Diagnostics (Python)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data: Triangle of Paid Losses
triangle = pd.DataFrame({
    '12': [100, 110, 120, 130],
    '24': [150, 160, 180, np.nan],
    '36': [180, 200, np.nan, np.nan],
    '48': [210, np.nan, np.nan, np.nan]
}, index=[2020, 2021, 2022, 2023])

# 1. Calculate Age-to-Age Factors (LDFs)
ldfs = pd.DataFrame()
for col in range(len(triangle.columns)-1):
    curr = triangle.iloc[:, col]
    next_ = triangle.iloc[:, col+1]
    ldfs[f'{triangle.columns[col]}-{triangle.columns[col+1]}'] = next_ / curr

print("Age-to-Age Factors:")
print(ldfs)

# 2. Diagnostic: Check for Stability (Coefficient of Variation)
# If CV > 10%, the method is dangerous.
cv = ldfs.std() / ldfs.mean()
print("\nvolatility (CV) of LDFs:")
print(cv)

# 3. Diagnostic: Settlement Rate (Paid / Ultimate)
# Assume we have an initial view of Ultimate
initial_ultimate = pd.Series([220, 230, 250, 270], index=triangle.index)
paid_to_ult = triangle['12'] / initial_ultimate
print("\nPaid to Ultimate (at 12 months):")
print(paid_to_ult)
# If 2023 is much lower than 2020, settlement has slowed down.
```

### 4.2 The "Method Selection" Script

```python
# Pseudo-code for a selection algorithm
def select_ultimate(ay, maturity, paid_cl, incurred_cl, bf):
    if maturity < 12:
        return bf # Too green for CL
    elif maturity > 60:
        return paid_cl # Stable, case reserves are gone
    else:
        # Blend
        return 0.5 * paid_cl + 0.5 * incurred_cl

# Apply to each row
# results['Selected'] = results.apply(lambda x: select_ultimate(x['AY'], x['Age'], ...), axis=1)
```

---

## 5. Evaluation & Validation

### 5.1 Actual vs. Expected (AvE)

*   **Test:** Last quarter, we predicted \$10M of payments.
*   **Actual:** We paid \$12M.
*   **Ratio:** 120%.
*   **Diagnosis:**
    *   Is it timing? (A payment happened 1 day early).
    *   Is it severity? (Claims are settling for more).
    *   Is it frequency? (More claims reported).

### 5.2 The Hindsight Test

*   Recalculate the reserves as of 2018 using only 2018 data.
*   Compare to the *current* view of 2018 Ultimate.
*   **Reserve Deficiency:** If the 2018 estimate was \$100M and now it's \$150M, we were deficient.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Calendar Year" Inflation**
    *   **Issue:** LDFs assume inflation is constant.
    *   **Reality:** Inflation spikes in 2022.
    *   **Result:** Historical LDFs (from 2015-2021) are too low.
    *   **Fix:** Use the **Berquist-Sherman** adjustment or explicit inflation loading.

2.  **Trap: Changing Mix of Business**
    *   **Issue:** Writing more "High Deductible" policies.
    *   **Result:** Small claims disappear. LDFs increase (because only large claims remain).
    *   **Fix:** Segment the data (Gross vs. Net of Deductible).

### 6.2 Implementation Challenges

1.  **Sparse Triangles:**
    *   Excess Lines might have 3 claims in the whole triangle.
    *   **Solution:** Use Industry Benchmarks (ISO/NCCI) for the LDFs. Do not rely on own data.

---

## 7. Advanced Topics & Extensions

### 7.1 Roll-Forward Methods

*   Instead of re-running the full triangle every month (Quarterly Reserving), we "Roll Forward" the previous quarter's result.
*   $IBNR_{t} = IBNR_{t-1} - \text{Paid}_{month} + \text{ExpectedIncurred}_{month}$.
*   *Risk:* You drift away from reality if you don't re-set often.

### 7.2 Individual Claim Reserving (ICR)

*   Using Machine Learning to predict the ultimate of *each open claim* based on its description, injury code, and lawyer.
*   Aggregating these up to get the Total Reserve.
*   *Status:* Emerging, but standard methods (CL) are still the primary for financial reporting.

---

## 8. Regulatory & Governance Considerations

### 8.1 Statement of Actuarial Opinion (SAO)

*   The Appointed Actuary must sign a legal document.
*   **Opinion Types:**
    *   **Reasonable:** Reserves are within a range.
    *   **Deficient:** Reserves are too low.
    *   **Redundant:** Reserves are too high.
    *   **Qualified:** "I can't tell because the data is bad."

---

## 9. Practical Example

### 9.1 Worked Example: The "Berquist-Sherman" Adjustment

**Scenario:**
*   New Claims Manager hired in 2021.
*   Strategy: "Close claims faster!"
*   **Data:** Paid LDFs explode (because payments are accelerated).
*   **Standard CL:** Projects massive ultimates (thinking the high payments mean high losses).
*   **Adjustment:**
    1.  Adjust the *past* Paid Triangle to the *current* settlement speed.
    2.  Re-calculate LDFs on the adjusted triangle.
    3.  Result: Lower LDFs, reasonable Ultimate.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Clean the Data** before you model.
2.  **Select Methods** based on maturity and stability.
3.  **Diagnose** changes in the environment (Inflation, Settlement).

### 10.2 When to Use This Knowledge
*   **Quarterly Close:** The bread and butter of reserving actuaries.
*   **M&A:** Due diligence on a target's reserves.

### 10.3 Critical Success Factors
1.  **Consistency:** Don't change methods every quarter just to smooth the result.
2.  **Documentation:** Write down *why* you picked the 5-year average LDF.
3.  **Communication:** Explain the "Why" to the CFO, not just the "What".

### 10.4 Further Reading
*   **Friedland:** "Estimating Unpaid Claims Using Basic Techniques".
*   **Berquist & Sherman:** "Loss Reserve Adequacy Testing".

---

## Appendix

### A. Glossary
*   **Green:** Immature accident year.
*   **Seasoned:** Mature accident year.
*   **Prior Year Development (PYD):** Change in estimate for old years.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Cape Cod ELR** | $\sum L / \sum (P/LDF)$ | Expected Loss |
| **AvE Ratio** | $Actual / Expected$ | Validation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
