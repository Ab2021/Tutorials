# Life Pricing & Reserving Case Study (Part 2) - Theoretical Deep Dive

## Overview
In the past, Life Reserves were just a formula ($A_x - P \ddot{a}_x$). Today, under Solvency II and IFRS 17, we use **Gross Premium Valuation (GPV)**. This involves projecting realistic cash flows for 40+ years using **Best Estimate Assumptions (BEA)** and adding a **Risk Margin**. This session covers the modern valuation framework.

---

## 1. Conceptual Foundation

### 1.1 Net Premium vs. Gross Premium Valuation

*   **Net Premium Valuation (NPV):**
    *   *Assumptions:* Locked-in at issue (Prudent).
    *   *Premium:* Net Premium only.
    *   *Purpose:* US Statutory (historically).
*   **Gross Premium Valuation (GPV):**
    *   *Assumptions:* Current Best Estimate (Realistic).
    *   *Premium:* Gross Premium (Actual cash collected).
    *   *Expenses:* Explicitly modeled.
    *   *Purpose:* Solvency II, IFRS 17, Embedded Value.

### 1.2 The Building Blocks of GPV

$$ Reserve = PV(\text{Benefits}) + PV(\text{Expenses}) - PV(\text{Gross Premiums}) $$
*   If $Reserve < 0$, it means the policy is profitable (Asset).
*   **Flooring:** Usually, we cannot hold a negative reserve (or we floor it at the Surrender Value).

### 1.3 Risk Margin (The "Buffer")

*   Since GPV uses *Best Estimate* assumptions (50/50 chance of being right), we need a buffer for uncertainty.
*   **Cost of Capital Method:**
    $$ RM = \sum_{t=0}^{\infty} \frac{SCR_t \times CoC}{(1+r)^{t+1}} $$
    *   $SCR_t$: Solvency Capital Requirement in year $t$.
    *   $CoC$: Cost of Capital (e.g., 6%).

---

## 2. Mathematical Framework

### 2.1 The Cash Flow Projection

For each year $t$:
1.  **Inflow:** $Premium_t \times P(\text{Persistency})$.
2.  **Outflow (Death):** $SumAssured \times P(\text{Death})$.
3.  **Outflow (Surrender):** $CSV_t \times P(\text{Surrender})$.
4.  **Outflow (Expense):** $Maintenance + Inflation$.
5.  **Net CF:** Inflow - Outflow.

### 2.2 Discounting

*   We discount the Net CF using the **Risk-Free Yield Curve** (e.g., EIOPA curve).
*   *Long Tail:* For liabilities > 20 years, the curve is extrapolated (Smith-Wilson method).

### 2.3 Best Estimate Assumptions (BEA)

*   **Mortality:** Base Table $\times$ Improvement Factor.
*   **Lapse:** Dynamic Lapse Function (Sensitive to interest rates).
    *   *Formula:* $Lapse = Base \times f(MarketRate - CreditedRate)$.
*   **Expense:** Per Policy + \% of Premium.

---

## 3. Theoretical Properties

### 3.1 Negative Reserves & Lapse Risk

*   If a policy is very profitable (GPV is negative), the insurer *loses money* if the policyholder lapses.
*   **Mass Lapse Scenario:** If 50% of people quit, the "Asset" disappears.
*   **Reserve Floor:** Setting Reserve = $\max(GPV, CSV)$ protects against this.

### 3.2 Interest Rate Sensitivity (Duration)

*   Life liabilities have very long duration (20+ years).
*   **Asset-Liability Mismatch:** If Assets have duration 10 and Liabilities have duration 20, a drop in interest rates kills the insurer.
*   **Hedging:** Using Swaps/Swaptions to match duration.

---

## 4. Modeling Artifacts & Implementation

### 4.1 GPV Model in Python

```python
import numpy as np
import pandas as pd

# Inputs
age = 40
term = 20
sum_assured = 100000
gross_premium = 500 # Annual
expense_per_policy = 50
inflation = 0.02
interest_rate = 0.03
lapse_rate = 0.05
mortality_rate = 0.002 # Flat for simplicity

# Arrays
years = np.arange(1, term + 1)
n_pols = np.zeros(term + 1)
n_pols[0] = 1000 # Start with 1000 policies

# Projection Loop
cash_flows = []

for t in range(term):
    # Decrements
    deaths = n_pols[t] * mortality_rate
    lapses = (n_pols[t] - deaths) * lapse_rate
    survivors = n_pols[t] - deaths - lapses
    n_pols[t+1] = survivors
    
    # Cash Flows
    premium_in = n_pols[t] * gross_premium
    death_out = deaths * sum_assured
    expense_out = n_pols[t] * expense_per_policy * (1 + inflation)**t
    
    net_cf = premium_in - death_out - expense_out
    cash_flows.append(net_cf)

# Discounting
pv_cf = 0
for t in range(term):
    pv_cf += cash_flows[t] / (1 + interest_rate)**(t+1)

# Reserve (Per Policy)
# Note: This is the Value at Time 0 (Issue). 
# Usually we calculate Reserve at Time T (Future).
reserve_per_pol = -pv_cf / 1000 

print(f"PV of Future Profits (Time 0): ${pv_cf:,.0f}")
print(f"Reserve per Policy (Time 0): ${reserve_per_pol:.2f}")
# If negative, it means it's profitable at issue.
```

### 4.2 Sensitivity Testing Script

```python
def run_sensitivity(shock_mortality=0.0, shock_lapse=0.0, shock_interest=0.0):
    # ... (Same logic as above but with shocked inputs)
    # mortality_rate = base_mortality * (1 + shock_mortality)
    # ...
    return pv_cf

base_pv = run_sensitivity()
mort_up_pv = run_sensitivity(shock_mortality=0.10)
lapse_up_pv = run_sensitivity(shock_lapse=0.20)
rate_down_pv = run_sensitivity(shock_interest=-0.01)

print(f"Base PV: {base_pv:,.0f}")
print(f"Mortality +10%: {mort_up_pv:,.0f} (Impact: {mort_up_pv - base_pv:,.0f})")
print(f"Lapse +20%: {lapse_up_pv:,.0f} (Impact: {lapse_up_pv - base_pv:,.0f})")
print(f"Interest -1%: {rate_down_pv:,.0f} (Impact: {rate_down_pv - base_pv:,.0f})")
```

---

## 5. Evaluation & Validation

### 5.1 Analysis of Change (AoC)

*   Why did the reserve change from \$100M to \$105M?
*   **Steps:**
    1.  **New Business:** +5M.
    2.  **Unwind of Discount:** +3M (Time passing).
    3.  **Experience Variance:** -2M (More deaths than expected).
    4.  **Assumption Change:** -1M (Updated mortality table).
*   **Check:** The sum must explain the total change.

### 5.2 Roll-Forward Validation

*   $Res_t = Res_{t-1} \times (1+i) + P - Claims - Expenses$.
*   Does the model output match this recursive formula?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Double Counting Risk**
    *   **Issue:** Using a "Prudent" mortality table AND adding a Risk Margin.
    *   **Rule:** GPV uses *Best Estimate* (Mean). Risk Margin handles the deviation. Don't pad the assumptions.

2.  **Trap: Contract Boundaries**
    *   **Issue:** Can we project premiums for 50 years?
    *   **IFRS 17 Rule:** Only if the insurer *cannot* re-price the policy. If we can raise rates next year, the "Contract Boundary" ends there.

### 6.2 Implementation Challenges

1.  **Runtime:**
    *   Projecting 1 million policies for 50 years (monthly steps) = 600 million calculations.
    *   **Solution:** Model Point Compression (Grouping similar policies).

---

## 7. Advanced Topics & Extensions

### 7.1 Stochastic GPV (Variable Annuities)

*   If benefits depend on the S&P 500 (Guarantees).
*   We cannot use a deterministic rate.
*   **Method:** Run 1,000 Economic Scenarios (ESG). Calculate GPV for each. Take the average.

### 7.2 IFRS 17 CSM (Contractual Service Margin)

*   Under IFRS 17, we don't book the profit at day 1.
*   **CSM:** A liability bucket that holds the unearned profit. It is released slowly over time.
*   $Liability = BEL + RM + CSM$. (BEL = Best Estimate Liability).

---

## 8. Regulatory & Governance Considerations

### 8.1 Assumption Governance

*   Changing the Lapse Assumption by 1% can release \$50M of profit.
*   **Control:** Assumption changes must be approved by the "Assumption Committee" and often the Board.
*   **Backtesting:** You must prove your assumption matches recent history.

---

## 9. Practical Example

### 9.1 Worked Example: The "Lapse Shock"

**Scenario:**
*   Company sells "Term to 100" (very cheap).
*   Assumption: 4% lapse rate forever.
*   **Reality:** At age 80, people stop lapsing (because the value is high). Lapse drops to 0.5%.
*   **Impact:** The insurer holds way too little reserve.
*   **Result:** Massive reserve strengthening required (Billions). (This actually happened to major insurers).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **GPV** is realistic, **NPV** is prudent.
2.  **Best Estimate Assumptions** drive the valuation.
3.  **Risk Margin** covers the uncertainty.

### 10.2 When to Use This Knowledge
*   **Solvency II:** Calculating Technical Provisions.
*   **Embedded Value:** Valuing a life insurance company for M&A.

### 10.3 Critical Success Factors
1.  **Data Quality:** If the "Date of Birth" is wrong, the reserve is wrong.
2.  **Economic Scenarios:** Using a robust ESG (Economic Scenario Generator).
3.  **Model Governance:** Locking down the code so actuaries can't "tweak" the result.

### 10.4 Further Reading
*   **Milliman:** "Life Insurance Valuation in the US".
*   **EIOPA:** "Technical Specifications for Solvency II".

---

## Appendix

### A. Glossary
*   **BEL:** Best Estimate Liability.
*   **CSV:** Cash Surrender Value.
*   **AoC:** Analysis of Change.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **GPV** | $PV(Out) - PV(In)$ | Valuation |
| **Risk Margin** | $CoC \times \sum SCR$ | Solvency |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
