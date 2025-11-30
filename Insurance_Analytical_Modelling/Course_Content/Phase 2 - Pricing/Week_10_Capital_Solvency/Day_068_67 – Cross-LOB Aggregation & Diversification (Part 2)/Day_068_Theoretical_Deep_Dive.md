# Cross-LOB Aggregation & Diversification (Part 2) - Theoretical Deep Dive

## Overview
Reinsurance is not just about protection; it's about **Capital Efficiency**. By ceding tail risk, we lower the SCR. This session covers **Reinsurance Optimization**, the **Efficient Frontier**, and how to structure a program to maximize ROE.

---

## 1. Conceptual Foundation

### 1.1 The Trade-off

*   **Cost:** Reinsurance Premium (ROP).
*   **Benefit:**
    1.  **Volatility Reduction:** Stabilizes P&L.
    2.  **Capital Relief:** Reduces SCR (Solvency Capital Requirement).
*   **Optimization Goal:** Maximize $ROE = \frac{Net Profit}{Net Capital}$.

### 1.2 Types of Structures

1.  **Quota Share (QS):** Proportional.
    *   *Effect:* Reduces Mean and Volatility equally. Good for "Volume" relief.
2.  **Excess of Loss (XoL):** Non-Proportional.
    *   *Effect:* Chops off the tail. Massive Capital Relief. High cost relative to expected loss.
3.  **Stop Loss:** Aggregate protection.
    *   *Effect:* Protects the bottom line. Very expensive.

### 1.3 The Efficient Frontier

*   Plot **Retained Earnings (Y-axis)** vs. **Retained Volatility (X-axis)**.
*   We want to be on the "Top Left" (High Return, Low Risk).
*   Reinsurance moves us along this curve.

---

## 2. Mathematical Framework

### 2.1 Capital Relief Calculation

$$ SCR_{net} = SCR_{gross} - \Delta SCR_{reins} $$
*   $\Delta SCR_{reins}$ is not just the limit purchased. It depends on the *correlation* with other risks.
*   **Counterparty Risk:** Buying reinsurance *increases* Credit Risk SCR. This offsets some benefit.

### 2.2 The Cost of Capital (CoC) Approach

*   Is the reinsurance expensive?
*   **Test:** If $Premium > E[Loss] + CoC \times \Delta SCR$, it destroys value.
*   **Implied CoC:** What CoC makes the deal breakeven? If it's < Insurer's WACC, buy it.

---

## 3. Theoretical Properties

### 3.1 Vertical vs. Horizontal

*   **Vertical:** Increasing the Limit (e.g., \$100M xs \$10M $\rightarrow$ \$200M xs \$10M).
    *   *Impact:* Protects against Severity (1-in-200 year event).
*   **Horizontal:** Increasing the Reinstatements (e.g., 1 @ 100% $\rightarrow$ Unlimited).
    *   *Impact:* Protects against Frequency (Multiple hurricanes in one year).

### 3.2 Basis Risk

*   **Parametric Reinsurance:** Pays if "Wind Speed > 100mph".
*   **Indemnity Reinsurance:** Pays if "Your Loss > \$10M".
*   **Basis Risk:** Wind speed is high, but your loss is low (or vice versa). Solvency II penalizes Basis Risk.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Optimization Script (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

# Inputs
gross_losses = np.random.lognormal(10, 1, 10000) # 10,000 simulations
capital_charge = 0.06 # 6% Cost of Capital

# Structure: XoL (Retention R, Limit L)
def apply_xol(losses, retention, limit):
    ceded = np.minimum(np.maximum(losses - retention, 0), limit)
    net = losses - ceded
    premium = np.mean(ceded) * 1.5 # 50% Loading
    return net, premium

# Grid Search
retentions = [10000, 20000, 50000, 100000]
results = []

for R in retentions:
    net_losses, prem = apply_xol(gross_losses, R, 1000000)
    
    # Calculate Capital (VaR 99.5%)
    scr_net = np.percentile(net_losses, 99.5) - np.mean(net_losses)
    
    # Calculate Total Cost
    # Cost = Expected Net Loss + Reins Premium + Cost of Capital * SCR
    total_cost = np.mean(net_losses) + prem + capital_charge * scr_net
    
    results.append((R, total_cost, scr_net))

# Output
print(f"{'Retention':<10} {'Total Cost':<15} {'SCR':<15}")
for res in results:
    print(f"{res[0]:<10.0f} {res[1]:<15.0f} {res[2]:<15.0f}")

# Plot
x = [r[2] for r in results] # SCR
y = [r[1] for r in results] # Cost
plt.scatter(x, y)
plt.xlabel("Required Capital (SCR)")
plt.ylabel("Total Economic Cost")
plt.title("Reinsurance Optimization Frontier")
plt.show()
```

### 4.2 The "RoL vs. PoL" Chart

*   **RoL (Rate on Line):** Premium / Limit.
*   **PoL (Probability of Loss):** Prob(Loss > Attachment).
*   **Chart:** Plot RoL vs. PoL.
*   **Hard Market:** RoL is high relative to PoL.
*   **Soft Market:** RoL is cheap. Buy more cover.

---

## 5. Evaluation & Validation

### 5.1 Transfer of Risk Test

*   Accounting rules (IFRS/GAAP) require "Significant Risk Transfer".
*   **10-10 Rule:** 10% chance of a 10% loss. (Day 55).
*   **ERD (Expected Reinsurer Deficit):** Must be > 1%.

### 5.2 Counterparty Limits

*   Don't buy 100% of your cover from one reinsurer (Munich Re).
*   **Diversify Panel:** Use Swiss Re, SCOR, Hannover Re.
*   **Collateral:** Require letters of credit if the reinsurer is unrated.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Buying to the Mean**
    *   **Issue:** Buying reinsurance because "it pays back every year".
    *   **Reality:** That's "Dollar Swapping". Reinsurance is for the Tail, not the Mean.
    *   **Fix:** Increase retention to stop swapping dollars.

2.  **Trap: Ignoring Reinstatements**
    *   **Issue:** Assuming the limit is available forever.
    *   **Reality:** After a loss, you might have to pay another premium to "Reload" the limit.

### 6.2 Implementation Challenges

1.  **Modeling Complex Structures:**
    *   "Annual Aggregate Deductible with an Inner Limit per Event".
    *   Requires Monte Carlo. Analytical formulas fail.

---

## 7. Advanced Topics & Extensions

### 7.1 Catastrophe Bonds (ILS)

*   Instead of a Reinsurer, you sell a Bond to investors.
*   If a Hurricane hits, investors lose their principal (you keep the money).
*   **Benefit:** Fully collateralized (No credit risk). Diversifies sources of capacity.

### 7.2 Sidecars

*   A special purpose vehicle (SPV) that takes a Quota Share of your book.
*   Investors put cash in.
*   Used to fund growth in a Hard Market.

---

## 8. Regulatory & Governance Considerations

### 8.1 Net Retention Guidelines

*   Board must approve the "Maximum Net Retention".
*   Example: "We will not retain more than 5% of Surplus in any single event."

---

## 9. Practical Example

### 9.1 Worked Example: The "Capital Arbitrage"

**Scenario:**
*   Insurer CoC = 10%.
*   Reinsurer CoC = 6% (Diversified Global Player).
*   **Deal:** Insurer buys XoL.
*   **Math:** Insurer pays \$1M premium to save \$10M of capital.
*   **Savings:** \$10M * 10% = \$1M.
*   **Result:** Breakeven. But if Reinsurer charges \$0.8M, it's an arbitrage. Value is created.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Optimize** based on Economic Cost, not just Premium.
2.  **XoL** is best for Capital Relief.
3.  **Credit Risk** must be factored in.

### 10.2 When to Use This Knowledge
*   **Renewal Season:** Deciding what to buy for Jan 1.
*   **Capital Planning:** Using reinsurance to avoid raising equity.

### 10.3 Critical Success Factors
1.  **Data:** Reinsurers need good data to give good prices.
2.  **Relationships:** Reinsurance is a relationship business.
3.  **Holistic View:** Don't buy silos. Buy for the Group.

### 10.4 Further Reading
*   **Froot:** "The Financing of Catastrophe Risk".
*   **Swiss Re:** "Sigma Reports on Reinsurance".

---

## Appendix

### A. Glossary
*   **ROP:** Rate on Line / Reinsurance Premium.
*   **GNPI:** Gross Net Premium Income (Base for pricing).
*   **Burn Cost:** Historical average loss.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Economic Cost** | $E[L] + Prem + CoC \cdot SCR$ | Optimization |
| **Rate on Line** | $Premium / Limit$ | Pricing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
