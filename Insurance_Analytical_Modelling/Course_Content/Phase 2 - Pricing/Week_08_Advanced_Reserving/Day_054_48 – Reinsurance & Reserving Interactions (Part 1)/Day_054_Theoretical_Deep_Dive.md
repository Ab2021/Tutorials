# Reinsurance & Reserving Interactions (Part 1) - Theoretical Deep Dive

## Overview
An insurer's balance sheet has two sides: **Gross** (what we owe the claimant) and **Net** (what we actually pay after reinsurance). The difference is the **Reinsurance Recoverable**, which is an Asset. This session covers the mechanics of **Quota Share** and **Excess of Loss (XoL)**, and the critical **Credit Risk** associated with that asset.

---

## 1. Conceptual Foundation

### 1.1 Gross vs. Net

*   **Gross Reserve:** The total liability to the policyholder.
*   **Ceded Reserve:** The amount the reinsurer owes us.
*   **Net Reserve:** Gross - Ceded.
*   **Accounting Rule:** We usually book Gross Liabilities and Ceded Assets separately (no netting on the balance sheet), to show the Credit Risk.

### 1.2 Types of Reinsurance

1.  **Quota Share (QS):** Proportional.
    *   "We cede 40% of everything."
    *   *Reserving:* Simple. Net = 60% of Gross. LDFs are usually the same.
2.  **Excess of Loss (XoL):** Non-Proportional.
    *   "We cede losses > \$1M."
    *   *Reserving:* Complex. Net LDFs are lower than Gross LDFs (because Net is capped).

### 1.3 The "Netting Down" Fallacy

*   **Mistake:** Calculating Gross Ultimate, then subtracting Reinsurance *paid to date* to get Net Reserve.
*   **Reality:** You must project **Gross Ultimate** and **Ceded Ultimate** separately.
*   **Why?** Reinsurance recoveries often happen *years* after the gross payment (due to reporting lags and proof of loss).

---

## 2. Mathematical Framework

### 2.1 Modeling Excess of Loss (XoL)

Let $X$ be the Gross Loss. The Net Loss $Y$ is:
$$ Y = \min(X, R) $$
where $R$ is the Retention.

**Impact on LDFs:**
*   Gross LDFs are driven by the *tail* (large claims developing).
*   Net LDFs are driven by the *attritional* claims (capped at $R$).
*   **Rule:** Net LDFs $\le$ Gross LDFs. (Usually).
*   *Warning:* If the retention is indexed (e.g., \$1M in 2010, \$2M in 2020), you cannot use a single triangle.

### 2.2 Reinsurance Recoverables

$$ \text{Recoverable} = (\text{Gross Ult} - \text{Gross Paid}) - (\text{Net Ult} - \text{Net Paid}) $$
*   This is the asset on the balance sheet.
*   It includes IBNR Recoverables (recoveries on claims that haven't happened yet).

---

## 3. Theoretical Properties

### 3.1 Credit Risk (Bad Debt)

*   If the Reinsurer goes insolvent, the Recoverable becomes 0.
*   **Provision for Reinsurance Bad Debt:**
    $$ \text{Provision} = \text{Recoverable} \times \text{PD} \times \text{LGD} $$
    *   PD: Probability of Default (based on Rating, e.g., AA).
    *   LGD: Loss Given Default (usually 50%).

### 3.2 Commutations

*   **Definition:** The Reinsurer pays a lump sum to the Insurer to exit the contract.
*   **Impact:**
    *   Recoverable $\to$ Cash.
    *   Credit Risk $\to$ 0.
    *   *Reserving:* You must remove the commuted claims from the Ceded triangle, or the history will look distorted.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Constructing the Net Triangle (Python)

```python
import numpy as np
import pandas as pd

# Data: Gross Triangle and Individual Large Losses
gross_triangle = pd.DataFrame({
    '12': [100, 110, 120],
    '24': [150, 160, np.nan],
    '36': [180, np.nan, np.nan]
}, index=[2020, 2021, 2022])

# Reinsurance: XoL $50 xs $50 (Retention=50, Limit=50)
# We need individual claim data to do this perfectly, 
# but often we only have "Ceded Paid" triangles.

ceded_triangle = pd.DataFrame({
    '12': [10, 20, 30],
    '24': [40, 50, np.nan],
    '36': [60, np.nan, np.nan]
}, index=[2020, 2021, 2022])

# Calculate Net Triangle
net_triangle = gross_triangle - ceded_triangle

print("Gross Triangle:")
print(gross_triangle)
print("\nNet Triangle:")
print(net_triangle)

# Calculate LDFs
def calculate_ldfs(tri):
    ldfs = []
    cols = tri.columns
    for i in range(len(cols)-1):
        # Sum of col i+1 / Sum of col i (excluding missing)
        mask = ~tri[cols[i+1]].isna()
        sum_curr = tri.loc[mask, cols[i]].sum()
        sum_next = tri.loc[mask, cols[i+1]].sum()
        ldfs.append(sum_next / sum_curr)
    return ldfs

gross_ldfs = calculate_ldfs(gross_triangle)
net_ldfs = calculate_ldfs(net_triangle)

print(f"\nGross LDF (12-24): {gross_ldfs[0]:.3f}")
print(f"Net LDF (12-24): {net_ldfs[0]:.3f}")

# Interpretation:
# Net LDF should be lower. If Net LDF > Gross LDF, check data.
# (Could happen if recoveries are slow to be paid).
```

### 4.2 Credit Risk Calculation

```python
# Inputs
recoverable = 5000000 # $5M
reinsurer_rating = 'A-'
# S&P Default Rates (Hypothetical)
default_rates = {'AAA': 0.001, 'AA': 0.002, 'A': 0.005, 'BBB': 0.02}

pd = default_rates.get(reinsurer_rating[:1], 0.01) # Simple mapping
lgd = 0.50 # Standard assumption

bad_debt_provision = recoverable * pd * lgd
net_asset_value = recoverable - bad_debt_provision

print(f"Recoverable: ${recoverable:,.0f}")
print(f"Bad Debt Provision: ${bad_debt_provision:,.0f}")
print(f"Net Asset Value: ${net_asset_value:,.0f}")
```

---

## 5. Evaluation & Validation

### 5.1 The "Gross-to-Net" Ratio

*   Plot `Net Ultimate / Gross Ultimate` by Accident Year.
*   **Expectation:** Should be stable (e.g., 80%).
*   **Shock:** If it drops to 50% in 2023, did we buy more reinsurance? Or are we over-estimating recoveries?

### 5.2 Consistency Check

*   `Gross IBNR` must be $\ge$ `Net IBNR`.
*   If `Net IBNR` > `Gross IBNR`, you are projecting negative ceded reserves (paying the reinsurer?).
*   *Exception:* Sliding Scale Commissions or Profit Shares can cause weird net effects.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Indexation Clauses**
    *   **Issue:** The retention is \$1M *indexed for inflation*.
    *   **Reality:** In 2023, the retention is \$1.5M.
    *   **Result:** The Ceded layer is shrinking. The Net layer is growing (leveraged inflation).

2.  **Trap: "Inuring" Reinsurance**
    *   **Issue:** You have Fac, then QS, then XoL.
    *   **Order:** Fac pays first. Then QS applies to the remainder. Then XoL applies to the QS net.
    *   **Error:** Applying them in the wrong order gives the wrong Net.

### 6.2 Implementation Challenges

1.  **Bordereaux Quality:**
    *   Reinsurers get data via "Bordereaux" (spreadsheets).
    *   Often aggregated or delayed.
    *   **Result:** Ceded triangles are "slower" than Gross. You must adjust the Ceded LDFs to match the Gross reporting speed.

---

## 7. Advanced Topics & Extensions

### 7.1 Transfer Pricing (Internal Reinsurance)

*   Global insurers use "Internal Re" to move capital.
*   **Reserving:** The "Ceding Company" books a recoverable. The "Assuming Company" books a liability.
*   **Consolidation:** They must cancel out exactly. If they don't, the Group accounts are wrong.

### 7.2 LPT (Loss Portfolio Transfer)

*   Selling the reserves to a run-off specialist (e.g., Berkshire Hathaway).
*   **Accounting:** You pay \$1B to transfer \$1.2B of reserves.
*   **Gain:** You book a \$200M gain (deferred).
*   **Reserving:** The reserves are gone from the Net balance sheet.

---

## 8. Regulatory & Governance Considerations

### 8.1 Schedule F (US Statutory)

*   The "Penalty Box" for slow-paying reinsurers.
*   If a reinsurer is slow (> 90 days past due), you must write off the recoverable (Provision for Unauthorized Reinsurance).
*   **Impact:** Hits Surplus directly.

---

## 9. Practical Example

### 9.1 Worked Example: The Hurricane Catastrophe

**Scenario:**
*   Gross Loss: \$100M (Hurricane).
*   Reinsurance:
    *   QS: 50%.
    *   XoL: \$10M xs \$10M (on the 50% net?).
    *   *Contract says:* XoL applies to the "Subject Premium" (Gross).
*   **Calculation:**
    1.  Gross: \$100M.
    2.  XoL Recovery: (\$100M - \$10M) capped at \$10M? No, usually XoL is "per event". Let's say XoL pays \$90M.
    3.  Net after XoL: \$10M.
    4.  QS Recovery: 50% of \$10M = \$5M.
    5.  Final Net: \$5M.
*   **Lesson:** The *order of application* changes everything. Read the slip.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Gross - Ceded = Net.**
2.  **XoL** leverages the Net volatility.
3.  **Credit Risk** is real.

### 10.2 When to Use This Knowledge
*   **Financial Reporting:** Every quarter.
*   **Capital Modeling:** Reinsurance is the main capital relief tool.

### 10.3 Critical Success Factors
1.  **Read the Contracts:** Don't guess the retention.
2.  **Match the Triangles:** Ensure Gross and Ceded data use the same cutoff dates.
3.  **Watch the Counterparty:** A cheap reinsurer is expensive if they don't pay.

### 10.4 Further Reading
*   **Patrik (2001):** "Reinsurance" (CAS Syllabus).
*   **Schedule F Instructions:** NAIC.

---

## Appendix

### A. Glossary
*   **Retention:** The deductible for the insurer.
*   **Cession:** The amount passed to the reinsurer.
*   **Bordereau:** Report sent to reinsurer.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Net Reserve** | $R_{gross} - R_{ceded}$ | Liability |
| **Provision** | $Rec \times PD \times LGD$ | Credit Risk |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
