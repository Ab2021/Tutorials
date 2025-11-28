# Introduction to Ratemaking (Part 1) - Theoretical Deep Dive

## Overview
Ratemaking (Pricing) is the process of setting the price of the insurance product *before* we know the cost of goods sold. Unlike manufacturing, where cost is known, insurance cost is stochastic. This session covers the **Fundamental Insurance Equation**, the components of a rate (Loss, Expense, Profit), and the core metrics (Frequency, Severity, Pure Premium).

---

## 1. Conceptual Foundation

### 1.1 The Fundamental Insurance Equation

$$ \text{Premium} = \text{Losses} + \text{Expenses} + \text{Profit} $$
*   **Losses:** The expected claim payments (Pure Premium).
*   **Expenses:**
    *   **LAE (Loss Adjustment Expense):** Cost to settle claims (Adjusters, Lawyers).
    *   **Underwriting Expense:** Commissions, Taxes, Overhead.
*   **Profit:** The target return on capital (Underwriting Profit).

### 1.2 Pure Premium vs. Gross Premium

*   **Pure Premium (Burning Cost):** The amount needed to pay losses only.
    $$ PP = \frac{\text{Total Losses}}{\text{Exposure}} = \text{Frequency} \times \text{Severity} $$
*   **Gross Premium (Rate):** The amount charged to the customer.
    $$ GP = \frac{PP + \text{Fixed Expenses}}{1 - \text{Variable Expense Ratio} - \text{Profit Load}} $$

### 1.3 Frequency and Severity

*   **Frequency ($F$):** Claims per Exposure. (e.g., 5 claims per 100 cars = 0.05).
*   **Severity ($S$):** Average Cost per Claim. (e.g., \$5,000).
*   **Relationship:** $PP = F \times S$.
    *   *Why split them?* Inflation affects Severity. Road safety affects Frequency. They move independently.

---

## 2. Mathematical Framework

### 2.1 The Loss Ratio Method

Used to adjust *existing* rates.
$$ \text{Indicated Change} = \frac{\text{Actual Loss Ratio}}{\text{Target Loss Ratio}} - 1 $$
*   **Actual LR:** (Losses + LAE) / Current Premium.
*   **Target LR:** (1 - Expense Ratio - Profit Load).
*   *Example:* Actual LR = 70%. Target LR = 60%. Change = 70/60 - 1 = +16.7%.

### 2.2 The Pure Premium Method

Used to calculate rates from scratch (New Products).
$$ \text{Indicated Rate} = \frac{\text{Pure Premium} + \text{Fixed Expense}}{1 - \text{Variable Expense} - \text{Profit}} $$
*   Does not rely on current premiums.

### 2.3 Exposure Bases

*   **Auto:** Car-Year.
*   **Workers Comp:** \$100 of Payroll.
*   **General Liability:** \$1,000 of Sales or Square Footage.
*   **Requirement:** The exposure should be proportional to the risk.

---

## 3. Theoretical Properties

### 3.1 Law of Large Numbers

*   Ratemaking relies on the LLN.
*   $\bar{X} \to \mu$ as $n \to \infty$.
*   **Credibility:** If $n$ is small, $\bar{X}$ is volatile. We must blend it with a complement (Industry Data).

### 3.2 Homogeneity vs. Credibility

*   **Homogeneity:** We want groups of risks that are similar (e.g., 16-year-old drivers).
*   **Credibility:** But if the group is too small, we have no data.
*   **Trade-off:** We group risks to get credibility, but lose homogeneity. (e.g., Grouping 16-18 year olds).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Calculating Indicated Rates (Python)

```python
import numpy as np
import pandas as pd

# Inputs
losses = 5000000
lae = 500000 # 10% of losses
earned_premium = 7000000
exposures = 10000

# Expense Assumptions
fixed_expense_per_policy = 50
variable_expense_ratio = 0.20 # Commissions + Taxes
profit_load = 0.05

# 1. Loss Ratio Method
total_loss_lae = losses + lae
actual_lr = total_loss_lae / earned_premium
target_lr = 1.0 - variable_expense_ratio - profit_load - (fixed_expense_per_policy * exposures / earned_premium)
# Note: Fixed expenses are usually handled as a ratio in LR method, or converted.
# Simplified Target LR: 1 - Total Expense Ratio - Profit
total_expense_ratio = (fixed_expense_per_policy * exposures / earned_premium) + variable_expense_ratio
target_lr_simple = 1.0 - total_expense_ratio - profit_load

indicated_change = (actual_lr / target_lr_simple) - 1

print(f"Actual LR: {actual_lr:.1%}")
print(f"Target LR: {target_lr_simple:.1%}")
print(f"Indicated Change: {indicated_change:.1%}")

# 2. Pure Premium Method
pure_premium = total_loss_lae / exposures
indicated_rate = (pure_premium + fixed_expense_per_policy) / (1 - variable_expense_ratio - profit_load)

print(f"\nPure Premium: ${pure_premium:.2f}")
print(f"Indicated Rate: ${indicated_rate:.2f}")
print(f"Current Average Rate: ${earned_premium/exposures:.2f}")
```

### 4.2 Frequency/Severity Analysis

```python
claim_count = 500
frequency = claim_count / exposures
severity = total_loss_lae / claim_count

print(f"\nFrequency: {frequency:.3f} claims per exposure")
print(f"Severity: ${severity:,.0f} per claim")
print(f"Check PP: {frequency * severity:.2f}")
```

---

## 5. Evaluation & Validation

### 5.1 The "Off-Balance"

*   If you change rates by +10%, do you get +10% premium?
*   **Distribution Shift:** If customers leave (Adverse Selection), you might get +5% premium but +15% losses.
*   **Validation:** Retrospective testing of rate changes.

### 5.2 Competitor Analysis

*   Actuarial Indication: +20%.
*   Market: Competitors are cutting rates.
*   **Decision:** Management might cap the increase at +5% and accept a lower profit (or loss) to save market share. (The "Underwriting Cycle").

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Written vs. Earned**
    *   **Issue:** Using Written Premium in the Loss Ratio.
    *   **Reality:** Losses match *Earned* Premium (exposure provided). Written Premium includes future exposure.
    *   **Rule:** Always use Earned Premium for Ratemaking.

2.  **Trap: Calendar vs. Accident Year**
    *   **Issue:** Using Calendar Year losses for pricing.
    *   **Reality:** CY includes reserve changes from old years. Pricing should be based on *Accident Year* (current accident cost).

### 6.2 Implementation Challenges

1.  **Matching:**
    *   Ensuring the Losses in the numerator match the Exposures in the denominator.
    *   *Example:* If you exclude "Mass" state losses, you must exclude "Mass" exposures.

---

## 7. Advanced Topics & Extensions

### 7.1 Generalized Linear Models (GLMs)

*   We don't just calculate one rate. We calculate a rate for every risk.
*   **GLM:** $Rate = Base \times \text{AgeFactor} \times \text{ZipFactor} \dots$
*   This is the standard for Personal Lines (Day 58).

### 7.2 Tiering

*   Placing customers into "Tiers" (Standard, Preferred, Ultra-Preferred).
*   Essentially a simplified GLM.

---

## 8. Regulatory & Governance Considerations

### 8.1 Rate Filings

*   In the US, you must file your rates with the DOI (Department of Insurance).
*   **Justification:** You must prove the rate is "Not Excessive, Not Inadequate, and Not Unfairly Discriminatory."
*   **Profit Provision:** Regulators often cap the profit load (e.g., 5%).

---

## 9. Practical Example

### 9.1 Worked Example: The Lemonade Stand

**Scenario:**
*   Selling Lemonade Insurance (spill coverage).
*   Exposure: 1000 cups sold.
*   Losses: 50 spills @ \$2 each = \$100.
*   Expense: \$50 for the stand.
*   Profit: Want 10%.

**Calculation:**
*   Pure Premium = \$100 / 1000 = \$0.10.
*   Fixed Expense = \$50 / 1000 = \$0.05.
*   Variable Expense = 0 (for simplicity).
*   Rate = (0.10 + 0.05) / (1 - 0.10) = 0.15 / 0.90 = \$0.167.
*   **Price:** 17 cents per cup.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Premium** pays for Losses, Expenses, and Profit.
2.  **Loss Ratio Method** adjusts; **Pure Premium Method** builds.
3.  **Earned Premium** is the denominator.

### 10.2 When to Use This Knowledge
*   **Product Management:** Setting base rates.
*   **Underwriting:** Understanding the target loss ratio.

### 10.3 Critical Success Factors
1.  **Data Quality:** Garbage in, garbage rates.
2.  **Trend Selection:** Past losses must be trended to the future policy period.
3.  **Expense Allocation:** Don't burden a cheap product with high fixed costs.

### 10.4 Further Reading
*   **Werner & Modlin:** "Basic Ratemaking" (CAS Syllabus).
*   **McClenahan:** "Ratemaking".

---

## Appendix

### A. Glossary
*   **Exposure:** The unit of risk (Car-Year).
*   **Loading:** Adding expenses/profit to the pure premium.
*   **Relativity:** The factor for a specific variable (e.g., Age 25 = 1.50).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Pure Premium** | $L / E$ | Base Cost |
| **Gross Rate** | $(PP+F)/(1-V-Q)$ | Final Price |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
