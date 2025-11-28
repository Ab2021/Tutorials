# Life Pricing & Reserving Case Study (Part 3) - Theoretical Deep Dive

## Overview
How do we know if a Life Insurance product is actually making money? We can't wait 40 years to find out. We use **Profit Testing** to project the cash flows and calculate the **IRR**, **Profit Margin**, and **Embedded Value (EV)**. This session covers the metrics that CFOs care about.

---

## 1. Conceptual Foundation

### 1.1 The Profit Signature

*   **Definition:** A vector of expected profits for each year of the policy's life.
*   **Time 0:** Usually negative (Acquisition Strain).
    *   *Why?* Commission + Underwriting > First Premium.
*   **Time 1+:** Positive (Renewal Profits).
*   **Goal:** The PV of future profits must exceed the initial loss.

### 1.2 Internal Rate of Return (IRR)

*   The discount rate that makes $NPV(\text{Profit Signature}) = 0$.
*   **Hurdle Rate:** Insurers typically demand an IRR of 10-15%.
*   **Capital:** The "Investment" is the initial strain + Solvency Capital held.

### 1.3 Embedded Value (EV)

*   **Shareholder Value:** EV = Adjusted Net Asset Value (ANAV) + Value of In-Force (VIF).
*   **VIF:** PV(Future Profits) - Cost of Capital.
*   **Meaning:** If the company stopped selling today, how much cash would it generate?

---

## 2. Mathematical Framework

### 2.1 The Profit Vector Calculation

For year $t$:
$$ Pr_t = (P_t - E_t - D_t)(1+i) - (V_t - V_{t-1}(1+i)) $$
*   $P_t$: Premium.
*   $E_t$: Expense.
*   $D_t$: Expected Death Benefit ($q_{x+t} \times S$).
*   $V_t$: Reserve at end of year.
*   **Interpretation:** Profit = Cash In - Cash Out - Increase in Reserve.

### 2.2 Value of New Business (VNB)

$$ VNB = \sum_{t=0}^{\infty} \frac{Pr_t \times P(\text{InForce}_t)}{(1+r)^t} $$
*   **Metric:** VNB Margin = VNB / PV(Premiums).
*   **Good Margin:** 5-10% for Savings products, 20%+ for Protection products.

### 2.3 Cost of Capital (CoC)

*   We must hold capital ($SCR$). We invest it at risk-free rate ($r$). Shareholders demand ($r + 6\%$).
*   **Frictional Cost:** The 6% spread is the "Cost of Capital".
*   $CoC_t = SCR_t \times 6\%$.
*   This reduces the VIF.

---

## 3. Theoretical Properties

### 3.1 The "Hockey Stick"

*   A typical profit signature looks like a hockey stick:
    *   Year 1: $-\$1000$ (Loss).
    *   Year 2: $+\$200$.
    *   Year 3: $+\$250$.
*   **Breakeven Year:** When does cumulative profit turn positive? (Usually year 5-7).

### 3.2 Sensitivity of VNB

*   VNB is highly sensitive to:
    *   **Discount Rate:** Since profits are far in the future.
    *   **Acquisition Cost:** Since it hits Time 0 directly.
    *   **Lapse Rate:** If people leave before the acquisition cost is recovered, VNB is destroyed.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Profit Testing Model (Python)

```python
import numpy as np
import numpy_financial as npf # For IRR

# Inputs
premium = 1000
acquisition_cost = 1200 # > Premium!
maintenance_cost = 50
reserve_increase = [500, 550, 600, 650, 0] # 5 Year Product
claims = [100, 110, 120, 130, 1000] # Maturity at end
interest = 0.04
hurdle_rate = 0.10

# Profit Vector Calculation
profits = []
# Year 0 (Issue)
profit_0 = premium - acquisition_cost - reserve_increase[0]
profits.append(profit_0)

# Years 1-4
for t in range(1, 5):
    # Simplified: Profit = (Prem - Exp - Claim) + Interest - DeltaReserve
    # Assuming Reserve earns interest
    op_profit = (premium - maintenance_cost - claims[t])
    inv_income = (reserve_increase[t-1] + op_profit) * interest
    delta_res = reserve_increase[t] - reserve_increase[t-1]
    
    # Total Profit
    # This is a simplified "Statutory Profit" view
    profit_t = op_profit + inv_income - delta_res
    profits.append(profit_t)

print("Profit Signature:", [int(p) for p in profits])

# IRR
irr = npf.irr(profits)
print(f"IRR: {irr:.1%}")

# VNB (at Hurdle Rate)
vnb = npf.npv(hurdle_rate, profits)
print(f"VNB (at 10%): ${vnb:.0f}")

# Decision
if irr > hurdle_rate:
    print("Product is Profitable.")
else:
    print("Product Destroys Value.")
```

### 4.2 Analysis of Change (VNB)

*   **Step 1:** Calculate VNB with last year's assumptions.
*   **Step 2:** Change Interest Rate. (Economic Variance).
*   **Step 3:** Change Expenses. (Operating Variance).
*   **Step 4:** Change Sales Volume. (Volume Variance).
*   **Waterfall Chart:** Visualizing how we got from VNB 2022 to VNB 2023.

---

## 5. Evaluation & Validation

### 5.1 The "Zeroization" Check

*   If we set the Discount Rate = Investment Return, does NPV = 0?
*   (Assuming no profit loading).
*   This validates the math of the model.

### 5.2 Market Consistent EV (MCEV)

*   Instead of using a fixed discount rate (10%), we use the Risk-Free Rate.
*   But we allow for the "Cost of Non-Hedgeable Risks" (CNHR).
*   **Trend:** Moving away from "Traditional EV" to MCEV or IFRS 17 Equity.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Ignoring Capital**
    *   **Issue:** Calculating IRR on "Cash Flows" but ignoring the Solvency Capital required to back the product.
    *   **Reality:** You must invest capital. This lowers the IRR.
    *   **Fix:** Include "Change in Required Capital" in the profit signature.

2.  **Trap: Cross-Subsidies**
    *   **Issue:** Product A is profitable, Product B is loss-making. Total is OK.
    *   **Risk:** Competitors will cherry-pick Product A. You are left with Product B.

### 6.2 Implementation Challenges

1.  **Expense Allocation:**
    *   How much of the CEO's salary should be allocated to this specific Term Life policy?
    *   **Marginal vs. Full Costing:** Pricing often uses Marginal. EV must use Full.

---

## 7. Advanced Topics & Extensions

### 7.1 Macro-Pricing

*   Instead of pricing per policy, we price the "Project".
*   Includes overhead, marketing campaigns, and IT build costs.
*   **Metric:** "Time to Payback" (e.g., 5 years).

### 7.2 Dynamic Policyholder Behavior

*   **Assumption:** Lapse = 5%.
*   **Dynamic:** If we raise premiums, Lapse = 20%.
*   **Model:** We need a "Reaction Function" in the projection loop.

---

## 8. Regulatory & Governance Considerations

### 8.1 EV Audit

*   Embedded Value is often audited by external actuaries (Milliman, Oliver Wyman).
*   **Disclosure:** Published in the Annual Report (Supplementary Information).
*   **Investors:** Hedge funds look at "Price / EV" ratio to value insurers. (Typically 0.8x to 1.2x).

---

## 9. Practical Example

### 9.1 Worked Example: The "New Business Strain"

**Scenario:**
*   Insurer sells massive volume of new business in Q4.
*   **Accounting Result:** Huge Loss (because of acquisition costs).
*   **EV Result:** Huge Profit (VNB is positive).
*   **Communication:** CFO must explain to shareholders: "We lost money because we grew too fast. But value was created."

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Profit Signature** shows the timing of cash flows.
2.  **IRR** measures the return on capital.
3.  **VNB** measures the value added by sales.

### 10.2 When to Use This Knowledge
*   **Pricing:** Setting the premium to hit the 12% IRR target.
*   **Reporting:** Calculating the EV for the Annual Report.

### 10.3 Critical Success Factors
1.  **Strain Management:** Don't sell so much that you run out of cash.
2.  **Assumption Monitoring:** If experience deviates from pricing, update the VNB immediately.
3.  **Capital Efficiency:** Design products that use less capital (Unit-Linked) to boost IRR.

### 10.4 Further Reading
*   **CFO Forum:** "Market Consistent Embedded Value Principles".
*   **SOA:** "Understanding Profitability in Life Insurance".

---

## Appendix

### A. Glossary
*   **ANAV:** Adjusted Net Asset Value.
*   **VIF:** Value of In-Force.
*   **Strain:** The initial cash outflow.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **VNB** | $PV(Profits_{new})$ | Growth Metric |
| **EV** | $ANAV + VIF$ | Valuation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
