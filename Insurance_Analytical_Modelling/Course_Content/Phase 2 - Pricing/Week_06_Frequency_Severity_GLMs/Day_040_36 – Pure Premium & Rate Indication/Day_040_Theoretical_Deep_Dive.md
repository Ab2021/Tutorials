# Pure Premium & Rate Indication - Theoretical Deep Dive

## Overview
We have built sophisticated GLMs for Frequency, Severity, and Pure Premium. Now, we must translate these statistical outputs into a commercial price tag: the **Indicated Rate**. This session bridges the gap between Actuarial Science and Business Strategy, covering the **Loss Ratio Method**, **Pure Premium Method**, **Expense Loading**, and the critical analysis of **Rate Dislocation**.

---

## 1. Conceptual Foundation

### 1.1 The Goal of Ratemaking

**Fundamental Equation:**
$$ \text{Premium} = \text{Losses} + \text{Expenses} + \text{Profit} $$

**The Indication:**
An actuarial calculation that tells us how much we *should* change our rates to achieve our target profit.
*   **Indication = +10%:** We need to raise rates by 10%.
*   **Indication = -5%:** We can lower rates by 5% (to gain market share).

### 1.2 Two Methods

1.  **Pure Premium Method:**
    *   Builds the rate from scratch.
    *   Used for **New Lines of Business** or when current rates are obsolete.
    *   Formula: $R = \frac{PP + F}{1 - V - Q}$.

2.  **Loss Ratio Method:**
    *   Adjusts the current rate.
    *   Used for **Existing Lines** (Rate Reviews).
    *   Formula: $\text{Change} = \frac{\text{Actual LR}}{\text{Target LR}} - 1$.

### 1.3 Rate Dislocation

**The Problem:**
Even if the *overall* indication is +0%, individual customers might see huge swings.
*   Customer A: +50% (New model says they are risky).
*   Customer B: -30% (New model says they are safe).

**Dislocation Analysis:**
Measuring the "shock" to the book. High dislocation leads to:
*   **Retention Drop:** Customers with +50% will leave.
*   **Adverse Selection:** Customers with -30% will stay (and we make less money on them).

---

## 2. Mathematical Framework

### 2.1 The Fundamental Insurance Equation

$$ P = L + E_F + (V \cdot P) + (Q \cdot P) $$
*   $P$: Premium (Rate).
*   $L$: Expected Loss (Pure Premium).
*   $E_F$: Fixed Expenses (Rent, Salaries).
*   $V$: Variable Expense % (Commissions, Taxes).
*   $Q$: Profit Provision % (Cost of Capital).

**Solving for P:**
$$ P (1 - V - Q) = L + E_F $$
$$ P = \frac{L + E_F}{1 - V - Q} $$

### 2.2 Loss Ratio Method

$$ \text{Indicated Change} = \frac{\text{Projected Loss Ratio}}{\text{Target Loss Ratio}} - 1 $$

*   **Projected LR:** $\frac{\text{Trended Ultimate Losses}}{\text{On-Level Earned Premium}}$.
*   **Target LR:** $1 - V - Q - \frac{E_F}{P}$. (The portion of premium available to pay losses).

### 2.3 Expense Loading

*   **Fixed ($E_F$):** A flat dollar amount per policy (e.g., \$25 for policy issuance).
*   **Variable ($V$):** A percentage (e.g., 15% Commission + 2% Premium Tax).
*   **Trend:** Expenses inflate too! We must trend $E_F$.

---

## 3. Theoretical Properties

### 3.1 On-Leveling Premiums

**The Problem:**
We are comparing 2022 losses to 2022 premiums. But we raised rates by 5% in 2023.
*   If we don't adjust, the 2022 premiums look "too low" compared to today's rates.

**The Fix (Parallelogram Method):**
Adjust historical premiums to the *Current Rate Level* (CRL).
*   "If we had charged today's rates back in 2022, how much would we have collected?"
*   This ensures the Indication measures the *adequacy of current rates*, not past rates.

### 3.2 Profit Provision ($Q$)

*   **Underwriting Profit:** Premium - Loss - Expense.
*   **Investment Income:** Insurers invest the "Float" (premium held before claims are paid).
*   **Target:** If Investment Income is high (5%), we might accept an Underwriting Loss ($Q = -2\%$). If rates are low (1%), we need Underwriting Profit ($Q = +3\%$).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Calculating the Indication (Python)

```python
import numpy as np
import pandas as pd

# Inputs
projected_losses = 55_000_000  # Trended & Developed
earned_premium_raw = 60_000_000
current_rate_level_factor = 1.05 # We took a 5% increase recently
fixed_expenses = 5_000_000
variable_expense_pct = 0.20 # 20% (Comm + Tax)
profit_load = 0.05 # 5% Target Profit

# 1. On-Level Premium
on_level_premium = earned_premium_raw * current_rate_level_factor
print(f"On-Level Premium: ${on_level_premium:,.0f}")

# 2. Variable Permissible Loss Ratio (VPLR)
# The % of premium available for Losses + Fixed Expenses
vplr = 1.0 - variable_expense_pct - profit_load
print(f"Variable Permissible LR: {vplr:.1%}")

# 3. Indicated Rate Change (Loss Ratio Method)
# Formula: (Loss + Fixed Exp) / (Prem * VPLR) - 1
# Or: (Loss Ratio + Fixed Exp Ratio) / VPLR - 1
# Let's do Total Need vs Total Available

total_need = projected_losses + fixed_expenses
total_available = on_level_premium * vplr # This is what covers Loss+Fixed

indication = (total_need / total_available) - 1
# Wait, standard formula:
# Indication = (Proj LR + Fixed Exp Ratio) / (1 - Var Exp - Profit) - 1
proj_lr = projected_losses / on_level_premium
fixed_exp_ratio = fixed_expenses / on_level_premium

indication_standard = (proj_lr + fixed_exp_ratio) / (1 - variable_expense_pct - profit_load) - 1

print(f"Projected Loss Ratio: {proj_lr:.1%}")
print(f"Fixed Exp Ratio: {fixed_exp_ratio:.1%}")
print(f"Indicated Rate Change: {indication_standard:.1%}")

# Interpretation:
# If Indication is +8.3%, we need to raise rates by 8.3%.
```

### 4.2 Dislocation Analysis (The "Impact" Plot)

1.  Calculate `OldPremium` and `NewPremium` for every current policy.
2.  Calculate `% Change = New / Old - 1`.
3.  **Histogram:** Plot the distribution of % Change.
4.  **Capping:** If the histogram shows many customers at +50%, we might impose a **Rate Cap** of +15% per renewal.

### 4.3 Off-Balancing

If we cap rates at +15%, we lose money (we wanted +50%).
*   **Off-Balance Factor:** We must raise the *Base Rate* slightly for everyone else to recover the lost revenue from the capped customers.
*   *It's a zero-sum game.*

---

## 5. Evaluation & Validation

### 5.1 Retention Modeling

*   Build a GLM to predict `RenewalProbability`.
*   Feature: `RateChange`.
*   **Elasticity:** How sensitive are customers to price hikes?
    *   "For every 1% rate increase, retention drops by 0.5%."
*   **Optimization:** Find the rate change that maximizes *Total Future Profit* (Premium $\times$ Retention - Loss).

### 5.2 Competitive Analysis

*   **Win Rate:** Compare your New Rate to Competitors (using a rater tool like Quadrant/Insurify).
*   If your Indication is +10% but you are already \$200 more expensive than Geico, taking the +10% might be suicide.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Ignoring Fixed Expenses**
    *   **Issue:** Treating all expenses as variable (%).
    *   **Result:** You overcharge high-premium policies (they pay too much for "rent") and undercharge low-premium policies.

2.  **Trap: Double Trending**
    *   **Issue:** Trending losses to 2025, but using 2022 premiums.
    *   **Fix:** You must trend losses *and* on-level premiums to the same future point.

### 6.2 Implementation Challenges

1.  **Data Quality:**
    *   "On-Leveling" requires a perfect history of every rate change for the last 3 years.
    *   **Solution:** Extension of Exposures (re-rating every historical policy) is accurate but computationally expensive. Parallelogram is an approximation.

---

## 7. Advanced Topics & Extensions

### 7.1 Lifetime Value (LTV) Pricing

*   Instead of targeting a 5% profit *this year*, target a 5% profit over the *customer's life*.
*   **New Business Penalty:** We might write new business at a loss (Indication -10%) to acquire them, knowing retention is high.
*   **Price Optimization:** Using elasticity to charge each customer the maximum they are willing to pay (Regulatory Warning: "Price Optimization" is banned in many states).

### 7.2 Tiered Rating

*   Creating "Underwriting Companies" (Tier A, Tier B, Tier C).
*   Each tier has a different Base Rate.
*   Allows for wider price segmentation than a single rating plan.

---

## 8. Regulatory & Governance Considerations

### 8.1 Unfair Discrimination

*   You cannot charge different rates based on Race, Religion, or (in some states) Credit Score or Gender.
*   **Disparate Impact:** Even if you don't use Race, if "Zip Code" correlates with Race, your rate change might be challenged.

### 8.2 Rate Capping Rules

*   Some states (e.g., California, New York) have strict rules on how much you can raise rates at once (e.g., "No more than 25%").

---

## 9. Practical Example

### 9.1 Worked Example: The "Death Spiral"

**Scenario:**
*   Indication is +20%.
*   We take the +20%.
*   **Result:** The safest, most price-sensitive customers leave (Retention drops).
*   The remaining pool is riskier.
*   Next year's Indication is +25%.
*   We take it. More safe customers leave.
*   **End Game:** Insolvency.

**Solution:**
*   Take +5% this year. Tighten underwriting (stop writing bad risks). Accept a short-term loss to stabilize the book.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Indication** is the math; **Selection** is the business decision.
2.  **On-Leveling** makes history comparable to today.
3.  **Dislocation** measures the pain to the customer.

### 10.2 When to Use This Knowledge
*   **Rate Reviews:** The annual ritual of every insurance product manager.
*   **Product Launch:** Setting the initial price tag.

### 10.3 Critical Success Factors
1.  **Balance:** Profit vs. Growth vs. Retention.
2.  **Granularity:** Don't just look at the statewide indication. Look at it by Territory and Vehicle Type.
3.  **Compliance:** If the state says "No," the math doesn't matter.

### 10.4 Further Reading
*   **Werner & Modlin:** "Basic Ratemaking" (The Bible).
*   **CAS Exam 5 Syllabus.**

---

## Appendix

### A. Glossary
*   **On-Level:** Adjusted to current rates.
*   **Permissible Loss Ratio (PLR):** The break-even loss ratio.
*   **Extension of Exposures:** Re-rating historical policies to calculate CRL.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Pure Premium Rate** | $(PP+F)/(1-V-Q)$ | New Business |
| **Loss Ratio Indication** | $LR_{proj} / LR_{target} - 1$ | Rate Review |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
