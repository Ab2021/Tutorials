# Solvency II & IFRS 17 (Part 1) - Theoretical Deep Dive

## Overview
Solvency II is the gold standard for risk-based capital regulation. It moved the industry from "Factor-Based" (Solvency I) to "Risk-Based" capital. This session covers the **Three Pillars**, the **Standard Formula** for SCR, and the **ORSA** process.

---

## 1. Conceptual Foundation

### 1.1 The Three Pillars

1.  **Pillar 1 (Quantitative):** How much capital must you hold?
    *   **SCR (Solvency Capital Requirement):** 99.5% VaR over 1 year.
    *   **MCR (Minimum Capital Requirement):** 85% VaR. (Below this, you lose your license).
    *   **Own Funds:** Available Capital (Assets - Liabilities).
2.  **Pillar 2 (Qualitative):** Governance and Risk Management.
    *   **ORSA:** Own Risk and Solvency Assessment.
    *   **Key Functions:** Actuarial, Risk, Compliance, Audit.
3.  **Pillar 3 (Disclosure):** Reporting to Public and Regulator.
    *   **SFCR:** Solvency and Financial Condition Report (Public).
    *   **RSR:** Regular Supervisory Report (Private).

### 1.2 The Balance Sheet

*   **Assets:** Market Value.
*   **Liabilities:** Best Estimate + Risk Margin (Technical Provisions).
*   **Own Funds:** Assets - Liabilities.
*   **Ratio:** Own Funds / SCR. (Target > 150%).

---

## 2. Mathematical Framework

### 2.1 The Standard Formula (SCR)

The SCR is built using a correlation matrix approach.
$$ SCR_{total} = \sqrt{\sum_{i,j} Corr_{i,j} \cdot SCR_i \cdot SCR_j} $$
*   **Modules:**
    *   **Market Risk:** Equity, Interest Rate, Spread, Property, Currency.
    *   **Counterparty Default Risk:** Reinsurer failure.
    *   **Life Underwriting Risk:** Mortality, Longevity, Lapse, Expense.
    *   **Non-Life Underwriting Risk:** Premium & Reserve, Catastrophe.
    *   **Health Underwriting Risk.**
    *   **Operational Risk.**

### 2.2 Aggregation Example

*   $SCR_{mkt} = 100$.
*   $SCR_{life} = 100$.
*   Correlation = 0.25.
*   $SCR_{total} = \sqrt{100^2 + 100^2 + 2 \cdot 0.25 \cdot 100 \cdot 100} = \sqrt{25000} \approx 158$.
*   **Diversification Benefit:** $200 - 158 = 42$.

### 2.3 MCR Calculation

*   Linear formula based on Premiums and Reserves.
*   **Corridor:** $25\% SCR \le MCR \le 45\% SCR$.

---

## 3. Theoretical Properties

### 3.1 The "1-in-200 Year" Event

*   SCR corresponds to the 99.5% VaR.
*   **Meaning:** The capital should be sufficient to survive a disaster that happens once every 200 years.
*   *Critique:* Is a 1-year horizon enough? (Run-off risk).

### 3.2 Loss Absorbing Capacity (LAC)

*   **LAC-DT (Deferred Taxes):** If you lose \$100M, you pay less tax in the future. This is an asset.
*   **LAC-TP (Technical Provisions):** If you lose money, you can cut discretionary bonuses (Life Insurance).

---

## 4. Modeling Artifacts & Implementation

### 4.1 SCR Calculation (Python)

```python
import numpy as np
import pandas as pd

# Inputs (Stand-alone Capital Charges)
scr_market = 50
scr_default = 10
scr_life = 40
scr_nonlife = 60
scr_health = 0

# Correlation Matrix (Simplified Level 1)
# Market, Default, Life, NonLife, Health
corr_matrix = np.array([
    [1.00, 0.25, 0.25, 0.25, 0.25],
    [0.25, 1.00, 0.25, 0.25, 0.25],
    [0.25, 0.25, 1.00, 0.00, 0.25], # Life/NonLife usually 0 corr
    [0.25, 0.25, 0.00, 1.00, 0.25],
    [0.25, 0.25, 0.25, 0.25, 1.00]
])

scrs = np.array([scr_market, scr_default, scr_life, scr_nonlife, scr_health])

# Matrix Multiplication: sqrt(x' R x)
bsrc = np.sqrt(scrs.T @ corr_matrix @ scrs) # Basic Solvency Capital Requirement

op_risk = 0.03 * bsrc # Operational Risk (Simplified)
adj = -5 # Loss Absorbing Capacity

total_scr = bsrc + op_risk + adj

print(f"BSCR: {bsrc:.2f}")
print(f"Total SCR: {total_scr:.2f}")
print(f"Sum of Parts: {np.sum(scrs):.2f}")
print(f"Diversification Benefit: {np.sum(scrs) - bsrc:.2f}")
```

### 4.2 ORSA Stress Testing

*   **Scenario:** "Pandemic + Market Crash".
*   **Impact:**
    *   Life Claims increase (Mortality Risk).
    *   Assets drop (Market Risk).
    *   Own Funds drop significantly.
*   **Outcome:** Does the Solvency Ratio stay above 100%? If not, we need a recovery plan.

---

## 5. Evaluation & Validation

### 5.1 QRTs (Quantitative Reporting Templates)

*   Massive spreadsheets (XBRL) sent to the regulator.
*   **S.06.02:** List of every single asset.
*   **S.17.01:** Non-Life Technical Provisions.
*   **Validation:** Regulators run automated checks (e.g., "Assets = Liabilities + Equity").

### 5.2 Internal Model Approval Process (IMAP)

*   If the Standard Formula doesn't fit (e.g., you write unique risks), you can build your own **Internal Model**.
*   **Hurdle:** Must pass the "Use Test" (Management actually uses it) and rigorous statistical quality tests.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Look-Through**
    *   **Issue:** You hold a Mutual Fund.
    *   **Rule:** You cannot just treat it as "Equity". You must "Look Through" to the underlying assets (Cash, Bonds, Derivatives) and calculate capital on each.

2.  **Trap: Contract Boundaries**
    *   **Issue:** Including future premiums that are not legally binding.
    *   **Rule:** Only include cash flows that are "contractually guaranteed".

### 6.2 Implementation Challenges

1.  **Data Granularity:**
    *   Standard Formula requires detailed data (e.g., duration of every bond).
    *   **Solution:** Data Warehousing (Day 56) is critical.

---

## 7. Advanced Topics & Extensions

### 7.1 Matching Adjustment (MA)

*   If you hold bonds to maturity to match annuity liabilities.
*   You can ignore the "Spread Risk" (Market volatility).
*   **Benefit:** Increases the Discount Rate, lowers Liabilities, increases Capital.
*   *Note:* Critical for UK Annuity writers.

### 7.2 Volatility Adjustment (VA)

*   A simpler version of MA.
*   Adds a spread to the discount curve during times of market stress to prevent pro-cyclicality.

---

## 8. Regulatory & Governance Considerations

### 8.1 The Actuarial Function Holder (AFH)

*   A specific role required by law.
*   Must produce an annual "Actuarial Function Report" (AFR) commenting on:
    *   Reliability of Technical Provisions.
    *   Underwriting Policy.
    *   Reinsurance Arrangements.

---

## 9. Practical Example

### 9.1 Worked Example: The "Double Hit"

**Scenario:**
*   Insurer holds Corporate Bonds and writes Annuities.
*   **Event:** Credit Spreads widen (Bonds drop in value).
*   **Standard Accounting:** Assets drop, Liabilities stay same. Insolvent.
*   **Solvency II (with VA):** Assets drop. Discount Rate increases (due to VA). Liabilities drop.
*   **Result:** Solvency Ratio is stable. The regime works.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **SCR** is the 1-in-200 year capital buffer.
2.  **Standard Formula** uses correlations.
3.  **ORSA** is the internal view of risk.

### 10.2 When to Use This Knowledge
*   **Capital Management:** Deciding how much dividend to pay.
*   **Product Design:** Designing capital-light products.

### 10.3 Critical Success Factors
1.  **Data Quality:** Look-through data is hard to get.
2.  **Governance:** The Board must understand the SCR.
3.  **Reporting Speed:** QRTs are due 5 weeks after quarter-end.

### 10.4 Further Reading
*   **EIOPA:** "The Underlying Assumptions in the Standard Formula".
*   **Sandstrom:** "Solvency: Models, Assessment and Regulation".

---

## Appendix

### A. Glossary
*   **QRT:** Quantitative Reporting Template.
*   **RSR:** Regular Supervisory Report.
*   **UFR:** Ultimate Forward Rate (for discounting).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **SCR Aggregation** | $\sqrt{x'Rx}$ | Capital Calculation |
| **Solvency Ratio** | $OwnFunds / SCR$ | KPI |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
