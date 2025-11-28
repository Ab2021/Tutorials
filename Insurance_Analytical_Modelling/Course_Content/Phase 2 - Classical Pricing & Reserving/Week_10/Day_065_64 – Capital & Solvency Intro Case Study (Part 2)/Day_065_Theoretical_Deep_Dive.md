# Solvency II & IFRS 17 (Part 2) - Theoretical Deep Dive

## Overview
IFRS 17 is the biggest change in insurance accounting in 20 years. It replaces "Revenue = Premium" with "Revenue = Service Provided". This session covers the **General Measurement Model (GMM)**, the **Premium Allocation Approach (PAA)**, and the concept of the **Contractual Service Margin (CSM)**.

---

## 1. Conceptual Foundation

### 1.1 The "Unearned Profit" Concept

*   **Old World (IFRS 4):** Book profit when premium is received (mostly).
*   **New World (IFRS 17):** Profit is stored in a bucket called **CSM** and released slowly as you provide coverage.
*   **Goal:** Comparability. A 10-year policy should recognize profit over 10 years, not all in Year 1.

### 1.2 The Three Models

1.  **GMM (General Measurement Model):** The default. For Long-Term Life & Reinsurance.
2.  **PAA (Premium Allocation Approach):** Simplified. For Short-Term Non-Life (< 1 year). Similar to UPR.
3.  **VFA (Variable Fee Approach):** For Unit-Linked / With-Profits business.

### 1.3 The Balance Sheet (GMM)

$$ Liability = BEL + RA + CSM $$
*   **BEL (Best Estimate Liability):** PV of Cash Flows.
*   **RA (Risk Adjustment):** Value of uncertainty (similar to Risk Margin).
*   **CSM (Contractual Service Margin):** Unearned Profit.

---

## 2. Mathematical Framework

### 2.1 Initial Recognition (GMM)

At Day 1:
$$ CSM_0 = -(PV_{inflows} - PV_{outflows} - RA) $$
*   If $CSM > 0$: We store it and release it later.
*   If $CSM < 0$: The contract is **Onerous**. We book a Loss immediately. (No negative CSM allowed).

### 2.2 Subsequent Measurement (Roll-Forward)

$$ CSM_{end} = CSM_{start} + \text{Interest} + \Delta \text{CashFlows} - \text{Amortization} $$
*   **Interest:** Accretion at the "Locked-in Rate".
*   **Delta CF:** Changes in future assumptions (e.g., mortality) adjust the CSM, not the P&L. (Smoothing).
*   **Amortization:** Release to P&L based on "Coverage Units".

### 2.3 PAA Eligibility

*   Can use PAA if Coverage Period $\le$ 1 year.
*   OR if PAA $\approx$ GMM (tested via "LRC Materiality Test").

---

## 3. Theoretical Properties

### 3.1 The OCI Option

*   **Problem:** Assets are Market Value (Volatile). Liabilities are Discounted (Volatile).
*   **Mismatch:** If Asset duration $\neq$ Liability duration, P&L is volatile.
*   **Solution:** Put the "Interest Rate Swing" into **OCI (Other Comprehensive Income)** instead of P&L.
*   *Result:* Stable Net Income, Volatile Equity.

### 3.2 Coverage Units

*   How do we release the CSM?
*   **Term Life:** Coverage Unit = Sum Assured.
*   **Annuity:** Coverage Unit = Fund Value or Annuity Payment.
*   **Release:** $Amortization = CSM \times \frac{Units_{current}}{Units_{current} + Units_{future}}$.

---

## 4. Modeling Artifacts & Implementation

### 4.1 CSM Engine (Python)

```python
import numpy as np
import pandas as pd

# Inputs
pv_premiums = 1000
pv_claims = 600
pv_expenses = 100
risk_adjustment = 50

# 1. Initial Recognition
fulfilment_cash_flows = pv_claims + pv_expenses + risk_adjustment
net_inflow = pv_premiums - fulfilment_cash_flows

if net_inflow > 0:
    csm = net_inflow
    loss_component = 0
    print(f"Profitable Group. CSM Created: {csm}")
else:
    csm = 0
    loss_component = -net_inflow
    print(f"Onerous Group. Loss Booked: {loss_component}")

# 2. Amortization (Simplistic)
coverage_units = [100, 100, 100, 100, 100] # 5 Year Policy
total_units = sum(coverage_units)
amortization_profile = []

remaining_csm = csm
for unit in coverage_units:
    # Release proportion of remaining units
    # Note: IFRS 17 uses (Current / (Current + Future))
    # Here we simplify to straight line for demo
    release = csm * (unit / total_units)
    amortization_profile.append(release)

print("CSM Release Profile:", amortization_profile)
```

### 4.2 PAA vs. GMM Comparison

*   **Scenario:** 3-Year Construction Bond.
*   **GMM:** Discount cash flows, calc CSM, release over 3 years.
*   **PAA:** Hold UPR. If loss ratio spikes in Year 2, check for Onerousness.
*   *Key Difference:* GMM updates assumptions every quarter. PAA ignores future assumptions unless onerous.

---

## 5. Evaluation & Validation

### 5.1 The "Unlock" Check

*   If Mortality assumption worsens by \$10M.
*   **IFRS 4:** P&L hit of -\$10M.
*   **IFRS 17:** CSM reduces by \$10M. P&L hit is \$0. (Unless CSM goes negative).
*   **Validation:** Check that the CSM movement exactly offsets the BEL change.

### 5.2 Transition Approaches

*   How to calculate CSM for a policy sold in 1990?
*   **Full Retrospective:** Re-run history as if IFRS 17 always existed. (Hard).
*   **Modified Retrospective:** Use approximations.
*   **Fair Value:** $CSM = FairValue - BEL$.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Reinsurance Mismatch**
    *   **Issue:** You have an Onerous underlying contract (Loss) and a Reinsurance contract (Gain).
    *   **Rule:** You can recognize the Reinsurance Gain immediately to offset the Loss. (Amendment 2020).

2.  **Trap: Discount Rates**
    *   **GMM:** Locked-in rate for CSM accretion. Current rate for BEL.
    *   **Difference:** Goes to OCI (if option selected) or P&L (Finance Expense).

### 6.2 Implementation Challenges

1.  **Data Volume:**
    *   IFRS 17 requires grouping by "Annual Cohort".
    *   Instead of 1 Life Portfolio, you have 30 Cohorts (1990-2020).
    *   **Result:** Data explosion.

---

## 7. Advanced Topics & Extensions

### 7.1 VFA (Variable Fee Approach)

*   For Unit-Linked business.
*   The insurer is essentially an asset manager earning a fee.
*   **CSM absorbs Financial Variance too.** (Unlike GMM where financial variance goes to P&L/OCI).

### 7.2 Experience Variance

*   **Premiums:** If you receive more premium than expected $\rightarrow$ CSM increases.
*   **Claims:** If you pay more claims than expected $\rightarrow$ P&L Hit (Experience Variance).
*   *Why?* Past is P&L. Future is CSM.

---

## 8. Regulatory & Governance Considerations

### 8.1 KPI Changes

*   **"Premium" is no longer Revenue.**
*   **New KPI:** "Insurance Service Result" (Release of CSM + RA + Exp Load).
*   **Impact:** Analysts need to be retrained. Growth in sales doesn't immediately boost the top line.

---

## 9. Practical Example

### 9.1 Worked Example: The "Onerous" Test

**Scenario:**
*   Group of Term Life policies.
*   Pricing Error: Premium is too low.
*   **Calculation:** $PV(Prem) < PV(Claims) + RA$.
*   **Result:** Net Outflow.
*   **Action:**
    1.  Set CSM = 0.
    2.  Book "Loss Component" (LC) on Balance Sheet.
    3.  Hit P&L immediately with the loss.
    4.  Track LC over time until it reverses.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **CSM** is the profit buffer.
2.  **GMM** is the main model, **PAA** is the simplification.
3.  **Cohorts** prevent cross-subsidization between generations.

### 10.2 When to Use This Knowledge
*   **Financial Reporting:** Producing the IFRS 17 P&L.
*   **Investor Relations:** Explaining why profit is stable despite volatility.

### 10.3 Critical Success Factors
1.  **Granularity:** Grouping contracts correctly (Portfolio, Cohort, Profitability).
2.  **Traceability:** Audit trail from Cash Flow to Journal Entry.
3.  **Systems:** You cannot do IFRS 17 in Excel. (Need SAS, Moody's, etc.).

### 10.4 Further Reading
*   **IASB:** "IFRS 17 Standard and Basis for Conclusions".
*   **Deloitte:** "IFRS 17 Insurance Contracts - A Pocket Guide".

---

## Appendix

### A. Glossary
*   **LRC:** Liability for Remaining Coverage.
*   **LIC:** Liability for Incurred Claims.
*   **LC:** Loss Component.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **CSM Roll-forward** | $CSM_{start} + Int + \Delta CF - Amort$ | Measurement |
| **Insurance Revenue** | $ExpClaims + RA_{rel} + CSM_{rel}$ | P&L |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
