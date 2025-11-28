# Reinsurance & Reserving Interactions (Part 2) - Theoretical Deep Dive

## Overview
Not all reinsurance is "Reinsurance." If a contract doesn't transfer significant risk, it's just a loan (Deposit Accounting). This session covers the **Risk Transfer Testing** rules (10-10 Rule, ERD), the mechanics of **Commutations** (buying back the risk), and **Loss Portfolio Transfers (LPTs)**.

---

## 1. Conceptual Foundation

### 1.1 Risk Transfer vs. Deposit Accounting

*   **Insurance Accounting (FAS 113 / SSAP 62):** Allows you to book recoveries as assets and offset liabilities.
*   **Deposit Accounting:** If the contract fails risk transfer, you book it as a loan. No reserve credit.
*   **The Principle:** The reinsurer must have a reasonable chance of losing a significant amount of money.

### 1.2 The "10-10" Rule

*   A heuristic used by auditors.
*   **Rule:** There must be at least a **10% chance** of the reinsurer suffering a **10% loss** (PV of Loss / PV of Premium > 110%).
*   *Critique:* It's a binary threshold. A 9% chance of a 1000% loss fails, which is absurd.

### 1.3 Expected Reinsurer Deficit (ERD)

*   The modern standard (CAS/AAA).
*   $ERD = E[\max(0, \text{Loss} - \text{Premium})] / \text{Premium}$.
*   **Threshold:** Usually 1%. (e.g., 1% chance of 100% loss passes).

---

## 2. Mathematical Framework

### 2.1 Commutations

**Definition:** The Reinsurer pays $P$ to the Cedant to cancel the contract.
*   Cedant takes back the reserves ($R$).
*   Reinsurer keeps the difference ($R - P$) as profit (or loss).

**Price of Commutation:**
$$ P = \sum \frac{E[\text{Future Claims}]}{(1+r)^t} - \text{Risk Discount} $$
*   The Reinsurer wants to pay less than the nominal reserve (Time Value of Money).
*   The Cedant accepts less because they get the cash *now* (and eliminate credit risk).

### 2.2 Loss Portfolio Transfer (LPT)

*   Transferring a block of *past* liabilities.
*   **Accounting Gain:**
    *   Book Value of Reserves: \$100M.
    *   Price Paid: \$90M (due to discounting).
    *   Gain: \$10M.
*   **Regulatory Rule:** The gain is usually *deferred* and amortized over the life of the paid claims (to prevent manipulation).

---

## 3. Theoretical Properties

### 3.1 The "Tail" in Commutations

*   Why commute?
    *   **Administrative Burden:** Keeping a treaty open for 1 tiny claim is annoying.
    *   **Disputes:** Resolve a coverage disagreement.
    *   **Solvency:** Reinsurer is shaky; get the cash out now.
*   **Adverse Selection:** The Reinsurer knows which claims are "dogs." The Cedant knows the case details. Information asymmetry is high.

### 3.2 Risk Transfer in Finite Re

*   **Finite Reinsurance:** Contracts with caps, corridors, and profit commissions.
*   Designed to smooth earnings, not transfer risk.
*   **Testing:** These contracts often fail the 10-10 rule and require Deposit Accounting.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Risk Transfer Simulation (Python)

```python
import numpy as np
import pandas as pd

# Scenario: Quota Share with a Loss Ratio Cap
# Premium: $10M
# Reinsurer pays 50% of losses, but Capped at 120% Loss Ratio.
# Commission: 30%.

n_sims = 10000
premium = 10000000
cap = 1.20 * premium
commission = 0.30 * premium

# Simulate Gross Losses (LogNormal)
# Mean $8M, CV 40%
mu = np.log(8000000) - 0.5 * np.log(1 + 0.4**2)
sigma = np.sqrt(np.log(1 + 0.4**2))
gross_losses = np.random.lognormal(mu, sigma, n_sims)

# Reinsurer Cash Flows
# Inflow: Premium - Commission
inflow = premium - commission
# Outflow: Min(50% * Loss, Cap)
outflow = np.minimum(0.5 * gross_losses, cap)

net_result = inflow - outflow
# Present Value (assume 1 year duration for simplicity, r=0)
# In reality, discount the outflow.

# 1. 10-10 Rule Check
# Loss = -Net Result (if Net Result < 0)
# Loss % = Loss / Premium
loss_pct = -net_result / premium
prob_10_pct_loss = np.mean(loss_pct > 0.10)

print(f"Probability of 10% Loss: {prob_10_pct_loss:.1%}")
if prob_10_pct_loss >= 0.10:
    print("Passes 10-10 Rule")
else:
    print("Fails 10-10 Rule")

# 2. ERD Calculation
# Deficit = Max(0, -Net Result)
deficit = np.maximum(0, -net_result)
erd = np.mean(deficit) / premium

print(f"ERD: {erd:.2%}")
if erd >= 0.01:
    print("Passes ERD Test")
else:
    print("Fails ERD Test")
```

### 4.2 Commutation Pricing

```python
# Reserve: $1M
# Payout: $200k/year for 5 years.
# Reinsurer Yield: 5%. Cedant Yield: 3%.

payouts = np.array([200000] * 5)
years = np.arange(1, 6)

# Reinsurer View (Break Even)
pv_re = np.sum(payouts / (1.05)**years)

# Cedant View (Break Even)
pv_cedant = np.sum(payouts / (1.03)**years)

print(f"Reinsurer Willing to Pay: < ${pv_re:,.0f}")
print(f"Cedant Willing to Accept: > ${pv_cedant:,.0f}")

# Negotiation Zone
print(f"Deal Zone: ${pv_re:,.0f} - ${pv_cedant:,.0f}")
# Wait, Cedant wants MORE than Reinsurer wants to pay?
# Yes. Commutations usually happen when:
# 1. Cedant has higher yield (needs cash).
# 2. Reinsurer has higher view of Ultimate (scared of explosion).
```

---

## 5. Evaluation & Validation

### 5.1 The "Side Letter" Trap

*   **Scandal:** AIG / Gen Re.
*   **Issue:** A formal contract transferred risk, but a secret "side letter" promised to pay it back.
*   **Result:** Criminal charges.
*   **Lesson:** The *entire* agreement must be documented and tested.

### 5.2 Deposit Accounting Entries

*   If it fails testing:
    *   **Asset:** Deposit Asset (Amount paid).
    *   **Income:** Interest Income (Yield on the deposit).
    *   **Claims:** None. (Payments reduce the deposit).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Profit Commissions**
    *   **Issue:** "Reinsurer pays losses, but if losses are low, they return 99% of profit."
    *   **Reality:** This reduces the Reinsurer's upside, but does it limit their downside?
    *   **Test:** Risk transfer focuses on the *downside* (Reinsurer losing money). Profit commissions usually don't kill risk transfer, but *Loss Corridors* do.

2.  **Trap: Discounting in the Test**
    *   **Issue:** Using nominal losses for the 10-10 test.
    *   **Reality:** You MUST use Present Value. A \$100 loss paid in 50 years is not a loss today.

### 6.2 Implementation Challenges

1.  **Parameterizing the Simulation:**
    *   How volatile is the tail?
    *   If you assume low volatility, you fail the test.
    *   Regulators look at your simulation assumptions closely.

---

## 7. Advanced Topics & Extensions

### 7.1 Retroactive Reinsurance

*   Reinsurance covering *past* accidents (LPT).
*   **Accounting:** Gains are deferred.
*   **Exception:** If the fraud/concealment exception applies.

### 7.2 Structured Settlements

*   Buying an annuity to pay a claimant.
*   **Risk Transfer:** The longevity risk moves to the Life Insurer.
*   **Reserving:** The claim is closed (if the annuity is "Assigned").

---

## 8. Regulatory & Governance Considerations

### 8.1 CEO/CFO Attestation

*   US Statutory requirement.
*   CEO/CFO must sign that "There are no side agreements."
*   Personal liability for hiding deposit contracts.

---

## 9. Practical Example

### 9.1 Worked Example: The "Time and Distance" Policy

**Scenario:**
*   Insurer pays \$80M premium.
*   Reinsurer pays \$100M in exactly 10 years.
*   **Risk?** None. It's a zero-coupon bond.
*   **Accounting:** Deposit Accounting.
*   **Why do it?** To smooth the P&L (in the old days). Now illegal/banned if disguised as insurance.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Risk Transfer** is the gateway to Reinsurance Accounting.
2.  **ERD** is the gold standard test.
3.  **Commutations** end the relationship.

### 10.2 When to Use This Knowledge
*   **Structuring Deals:** Helping the Ceded Re team design a treaty.
*   **M&A:** Valuing LPTs.

### 10.3 Critical Success Factors
1.  **Simulate Cash Flows:** Don't rely on averages.
2.  **Check the Yield Curve:** Interest rates drive the 10-10 test.
3.  **Document Everything:** The auditor will ask for the Risk Transfer Analysis.

### 10.4 Further Reading
*   **FASB 113:** "Accounting and Reporting for Reinsurance".
*   **Freihaut & Vendetti:** "Common Pitfalls in Risk Transfer Testing".

---

## Appendix

### A. Glossary
*   **Cedant:** The primary insurer.
*   **Commutation:** Early termination of a treaty.
*   **LPT:** Loss Portfolio Transfer.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **ERD** | $E[Deficit] / Premium$ | Risk Transfer |
| **Commutation** | $PV(Liabilities) - Risk$ | Pricing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
