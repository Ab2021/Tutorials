# Customer Churn & Retention (Part 2) - Customer Lifetime Value (CLV) - Theoretical Deep Dive

## Overview
"Not all customers are created equal."
Some pay premiums for 20 years and never claim. Others claim on Day 1 and leave.
**Customer Lifetime Value (CLV)** is the North Star metric that combines Profitability, Retention, and Growth potential into a single number.
This day focuses on **Calculating CLV**, **Predictive CLV**, and **Value-Based Segmentation**.

---

## 1. Conceptual Foundation

### 1.1 The CLV Equation

$$ CLV = \sum_{t=1}^{T} \frac{(Premium_t - Loss_t - Expense_t) \times Retention_t}{(1 + d)^t} - AcquisitionCost $$
*   **Premium:** Future premiums (requires inflation model).
*   **Loss:** Expected losses (requires risk model).
*   **Retention:** Probability of staying (requires churn model).
*   **d:** Discount rate.

### 1.2 Historical vs. Predictive CLV

*   **Historical Value:** What have they paid *so far*? (Sunk cost). Good for rewarding loyalty.
*   **Predictive Value:** What *will* they pay? (Future potential). Good for decision making.
*   **Focus:** We care about Predictive CLV (pCLV).

---

## 2. Mathematical Framework

### 2.1 The Buy 'Til You Die (BTYD) Models

*   **Context:** Usually for non-contractual settings (Retail), but adapted for Insurance cross-sell.
*   **Pareto/NBD:** Models "Recency" and "Frequency" of interactions.
*   **BG/NBD:** Beta-Geometric / Negative Binomial.
    *   Probability of being "Alive" (Active).
    *   Expected number of future transactions.

### 2.2 Machine Learning for pCLV

*   **Approach:** Regression.
*   **Target:** Sum of Margins over next 3 years.
*   **Features:**
    *   Current Margin.
    *   Churn Probability.
    *   Cross-sell propensity (likelihood to buy Home + Auto).
    *   Demographics (Income, Home Value).

---

## 3. Theoretical Properties

### 3.1 The "Zero" CLV Problem

*   **Issue:** Many customers have negative CLV (Loss Ratio > 100%).
*   **Strategy:**
    *   **Re-underwrite:** Raise rates or drop coverage.
    *   **Cross-subsidize:** Maybe they have a profitable Life policy? (Household view).

### 3.2 The Time Horizon

*   **Choice:** 3 years? 5 years? Lifetime?
*   **Insurance:** Usually 3-5 years. "Lifetime" is too uncertain due to market changes.
*   **Discounting:** Future money is worth less. $d \approx 10\%$ (Cost of Capital).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Simple CLV Calculation (Python)

```python
import pandas as pd

# Inputs
avg_premium = 1000
loss_ratio = 0.70
expense_ratio = 0.20
retention_rate = 0.85
discount_rate = 0.10
years = 5

margin = avg_premium * (1 - loss_ratio - expense_ratio) # $100
clv = 0

for t in range(1, years + 1):
    prob_alive = retention_rate ** t
    discount_factor = (1 + discount_rate) ** t
    clv += (margin * prob_alive) / discount_factor

print(f"5-Year CLV: ${clv:.2f}")
```

### 4.2 Predictive CLV (Random Forest)

```python
from sklearn.ensemble import RandomForestRegressor

# Target: Actual 3-year profit (calculated from historical data)
# Features: Tenure, Premium, Claims, Churn_Score
X = df[['tenure', 'premium', 'claims_cost', 'churn_prob']]
y = df['future_3yr_profit']

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict for current customers
df['predicted_clv'] = model.predict(X_current)
```

---

## 5. Evaluation & Validation

### 5.1 Decile Analysis

*   **Check:** Rank customers by pCLV.
*   **Validation:** Wait 1 year. Did the top decile actually generate the most profit?
*   **Gini:** Calculate Gini coefficient for Profit concentration. (Usually 20% of customers generate 150% of profit, while the bottom 20% destroy value).

### 5.2 Stability

*   **Check:** Does a customer's CLV score jump wildly from month to month?
*   **Goal:** Stable scores unless a major event happens (Claim, Rate Change).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Claim" Paradox

*   **Paradox:** A customer who just had a claim has a *lower* historical profit, but might have a *higher* retention (fear of switching) and *higher* premium (surcharge).
*   **Result:** Their pCLV might actually go *up* after a claim (if the surcharge > expected future loss).
*   **Ethics:** Is it fair to target them?

### 6.2 Cost Allocation

*   **Issue:** How to allocate "Fixed Expenses" (CEO salary) to individual customers?
*   **Method:** Marginal Expense vs. Fully Loaded Expense.
    *   For CLV, usually use **Marginal Expense** (Variable costs).

---

## 7. Advanced Topics & Extensions

### 7.1 Household CLV

*   **Concept:** Sum of CLV for all policies in a household (Auto + Home + Umbrella).
*   **Network Effect:** If the son leaves, the parents might leave too.
*   **Model:** Graph-based CLV.

### 7.2 Dynamic Pricing with CLV

*   **Strategy:** Offer a discount equal to 10% of CLV to acquire a high-value customer.
*   **Constraint:** Regulatory limits on "Unfair Discrimination".

---

## 8. Regulatory & Governance Considerations

### 8.1 Fairness

*   **Risk:** CLV models often correlate with Income/Credit.
*   **Bias:** Targeting high CLV might mean excluding low-income neighborhoods (Redlining).
*   **Audit:** Test CLV distribution across protected classes.

---

## 9. Practical Example

### 9.1 The "Cross-Sell" Engine

**Scenario:** We want to sell Umbrella policies.
**Targeting:**
1.  **Traditional:** Anyone with a Home and Auto policy.
2.  **CLV-Based:** Customers with High pCLV *and* High Asset Wealth (need for Umbrella).
**Action:** Marketing spend focused only on the High CLV segment.
**Result:** Conversion rate doubled. ROI tripled.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **CLV** = Profit $\times$ Retention.
2.  **Predictive** is better than Historical.
3.  **Segmentation** is the primary use case.

### 10.2 When to Use This Knowledge
*   **Strategy:** Deciding which markets to enter.
*   **Service:** Prioritizing call center queues (High CLV goes first).

### 10.3 Critical Success Factors
1.  **Integration:** CLV is useless in a spreadsheet. It must be in the CRM (Salesforce).
2.  **Long-term view:** Management must accept that acquiring High CLV customers costs more upfront.

### 10.4 Further Reading
*   **Fader & Hardie:** "Probability Models for Customer-Base Analysis".
*   **Kumar:** "Customer Lifetime Value".

---

## Appendix

### A. Glossary
*   **CAC:** Customer Acquisition Cost.
*   **Churn Rate:** $1 - Retention Rate$.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Simple CLV** | $\frac{Margin \times Retention}{1 + Discount - Retention}$ | Infinite Horizon Approx |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
