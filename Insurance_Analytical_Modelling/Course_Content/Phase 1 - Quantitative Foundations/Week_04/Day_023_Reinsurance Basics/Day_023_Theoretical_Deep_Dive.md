# Reinsurance Basics - Theoretical Deep Dive

## Overview
This session covers Reinsurance, the "insurance for insurers." We explore the fundamental structures (Treaty vs. Facultative), the mathematical mechanics of Proportional (Quota Share, Surplus Share) and Non-Proportional (Excess of Loss, Stop Loss) arrangements, and the pricing methodologies (Experience vs. Exposure Rating) used to transfer risk and manage capital.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Reinsurance:** A transaction where one insurer (the *Cedant*) transfers a portion of its risk portfolio to another insurer (the *Reinsurer*) in exchange for a premium.

**Core Objectives:**
1.  **Capacity:** Allows the cedant to write larger risks (e.g., a $500M building) than its own capital would permit.
2.  **Stabilization:** Smooths out volatility in financial results (e.g., capping losses in a bad year).
3.  **Catastrophe Protection:** Protects against single events affecting many policies (e.g., Hurricane).
4.  **Capital Relief:** Reduces required capital (Solvency II / RBC) by transferring tail risk.

**Key Terminology:**
*   **Retention:** The amount of risk the cedant keeps.
*   **Cession:** The amount transferred to the reinsurer.
*   **Retrocession:** When a reinsurer buys reinsurance for itself.
*   **Attachment Point:** The threshold where reinsurance coverage begins (like a deductible).

### 1.2 Historical Context & Evolution

**Origin:**
*   **14th Century:** Marine reinsurance in Italy.
*   **19th Century:** Cologne Re (1846) and Swiss Re (1863) established as dedicated professional reinsurers.

**Evolution:**
*   **Proportional Era:** Early reinsurance was mostly Quota Share (partnerships).
*   **Excess of Loss:** As modeling improved, insurers preferred XoL to retain high-frequency small losses and only cede large shocks.
*   **ILS (Insurance Linked Securities):** Cat Bonds and Sidecars entered the market in the 1990s, allowing capital markets to act as reinsurers.

**Current State:**
*   **Alternative Capital:** Hedge funds and pension funds provide collateralized reinsurance.
*   **Soft vs. Hard Markets:** Reinsurance pricing cycles are global and volatile, driven by recent catastrophe activity.

### 1.3 Why This Matters

**Business Impact:**
*   **Solvency:** Without reinsurance, a small insurer could be bankrupted by one large fire or lawsuit.
*   **Earnings Smoothing:** Publicly traded insurers use reinsurance to avoid missing earnings targets due to weather events.

**Regulatory Relevance:**
*   **Credit for Reinsurance:** Regulators allow insurers to hold less reserves if they have valid reinsurance.
*   **Counterparty Risk:** If the reinsurer goes bust, the cedant is still liable to the original policyholder.

---

## 2. Mathematical Framework

### 2.1 Structural Classification

| Type | Description | Example |
| :--- | :--- | :--- |
| **Treaty** | Automatic coverage for a whole portfolio. | "All auto policies written in 2024." |
| **Facultative** | Negotiated risk-by-risk. | "This specific oil rig." |

### 2.2 Proportional Reinsurance

The reinsurer shares a % of premiums and a % of losses.

**1. Quota Share (QS):**
*   **Mechanism:** Fixed % (e.g., 20%) ceded on *every* risk.
*   **Math:**
    $$ \text{Ceded Loss} = \alpha \times \text{Gross Loss} $$
    $$ \text{Ceded Premium} = \alpha \times \text{Gross Premium} \times (1 - \text{Ceding Commission}) $$
*   **Use Case:** New companies needing capital relief; entering a new line of business.

**2. Surplus Share:**
*   **Mechanism:** Cedes only the amount above a "Line" (Retention). The % varies by risk size.
*   **Math:**
    $$ \text{Retained Amount} = \min(\text{Line}, \text{Sum Insured}) $$
    $$ \text{Cession \%} = \frac{\text{Sum Insured} - \text{Retained Amount}}{\text{Sum Insured}} $$
*   **Use Case:** Property insurance (homogeneous retention, heterogeneous limits).

### 2.3 Non-Proportional Reinsurance (Excess of Loss - XoL)

The reinsurer pays only if the loss exceeds a retention ($R$).

**1. Per Risk XoL (Working Layer):**
*   **Mechanism:** Applies to each individual claim.
*   **Math:**
    $$ \text{Reinsurer Pays} = \min(\text{Limit}, \max(0, \text{Loss} - R)) $$
*   **Use Case:** Capping large liability claims.

**2. Catastrophe XoL (Cat Cover):**
*   **Mechanism:** Applies to the *aggregate* of all claims from a single event.
*   **Math:**
    $$ \text{Reinsurer Pays} = \min(\text{Limit}, \max(0, \sum \text{Losses} - R)) $$
*   **Use Case:** Hurricane, Earthquake.

**3. Stop Loss (Aggregate XoL):**
*   **Mechanism:** Applies to the total annual loss ratio.
*   **Math:** Pays if Loss Ratio > 100%.
*   **Use Case:** Protecting the bottom line (rarely sold now due to moral hazard).

### 2.4 Reinsurance Pricing Methods

**1. Experience Rating (Burning Cost):**
*   Used when the cedant has credible history.
*   **Burning Cost:** $\frac{\text{Historical Losses in Layer}}{\text{Historical Premium}}$.
*   **Loading:** Rate = Burning Cost $\times$ LDF $\times$ Trend / (1 - Expense - Profit).

**2. Exposure Rating:**
*   Used when history is sparse (e.g., high excess layers).
*   **Curves:** Uses industry severity curves (Swiss Re Curves, MBBEFD curves) to estimate the % of loss falling into the layer.
*   **Formula:**
    $$ E[\text{Layer Loss}] = E[\text{Total Loss}] \times [G(R + L) - G(R)] $$
    Where $G(x)$ is the Limited Expected Value function relative to the total mean.

---

## 3. Theoretical Properties

### 3.1 The Reinsurance Theorem (Borch)

*   **Optimal Arrangement:** Under certain conditions, the optimal reinsurance structure (Pareto-optimal) is a **Stop Loss** contract.
*   **Why not ubiquitous?** Moral hazard. If the reinsurer pays everything above a 100% loss ratio, the insurer stops underwriting carefully.

### 3.2 Stability vs. Cost

*   **Proportional:** High stability, high cost (giving away profit on good risks).
*   **Non-Proportional:** Lower cost (only paying for tail protection), higher volatility (retaining all attritional losses).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Bordereau:** A detailed report sent to reinsurers listing risks (for Proportional) or large claims (for XoL).
*   **Triangles:** Gross vs. Net development triangles to assess the impact of reinsurance on reserves.

### 4.2 Preprocessing Steps

**Step 1: As-If Adjustments**
*   Restating historical large losses as if current reinsurance limits were in place.

**Step 2: Indexation**
*   Adjusting historical retentions for inflation (Stability Clause).

### 4.3 Model Specification (Python Example)

Simulating Gross vs. Net results under Quota Share and XoL.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation Parameters
np.random.seed(42)
n_sims = 10000
lambda_freq = 10 # Avg claims per year
mu_sev = 100000 # Avg severity (Lognormal)
sigma_sev = 1.5

# Reinsurance Structures
# 1. Quota Share: 20% Cession
qs_percent = 0.20

# 2. Excess of Loss: 500k xs 100k
xol_retention = 100000
xol_limit = 500000

results = []

for i in range(n_sims):
    # Simulate Frequency
    n_claims = np.random.poisson(lambda_freq)
    
    if n_claims == 0:
        results.append([0, 0, 0, 0])
        continue
        
    # Simulate Severity
    losses = np.random.lognormal(np.log(mu_sev), sigma_sev, n_claims)
    
    gross_loss = np.sum(losses)
    
    # Apply QS
    qs_recovery = gross_loss * qs_percent
    net_qs = gross_loss - qs_recovery
    
    # Apply XoL (Per Risk)
    xol_recoveries = np.minimum(xol_limit, np.maximum(0, losses - xol_retention))
    total_xol_recovery = np.sum(xol_recoveries)
    net_xol = gross_loss - total_xol_recovery
    
    results.append([gross_loss, net_qs, net_xol, total_xol_recovery])

df = pd.DataFrame(results, columns=['Gross', 'Net_QS', 'Net_XoL', 'XoL_Rec'])

print("Summary Statistics (Means):")
print(df.mean().apply(lambda x: f"${x:,.0f}"))

print("\nStandard Deviation (Volatility):")
print(df.std().apply(lambda x: f"${x:,.0f}"))

print("\n99th Percentile (Tail Risk):")
print(df.quantile(0.99).apply(lambda x: f"${x:,.0f}"))

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(df['Gross'], bins=50, alpha=0.5, label='Gross')
plt.hist(df['Net_XoL'], bins=50, alpha=0.5, label='Net (XoL)')
plt.title('Gross vs. Net Loss Distribution (XoL)')
plt.legend()
plt.show()

# Interpretation:
# QS reduces Mean and Volatility proportionally.
# XoL reduces Volatility and Tail Risk significantly, but has small impact on Mean (unless losses are huge).
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Net Loss Distribution:** The profile of risk remaining with the cedant.
2.  **Ceded Premium:** The cost of the protection.
3.  **Reinsurance Efficiency:** Reduction in Capital / Ceded Premium.

**Interpretation:**
*   **XoL Efficiency:** Often viewed as more efficient capital management because it specifically targets the capital-consuming tail events.

---

## 5. Evaluation & Validation

### 5.1 Reinstatements

*   **Concept:** In XoL, if the limit is exhausted by a claim, the cedant often pays a "Reinstatement Premium" to restore the limit for the rest of the year.
*   **Modeling:** Must include this contingent cost in the pricing model.

### 5.2 Basis Risk (Cat Bonds)

*   **Issue:** ILS often trigger on *industry* losses or *parametric* triggers (wind speed), not the cedant's *actual* losses.
*   **Risk:** Cedant has a huge loss, but the bond doesn't trigger (Basis Risk).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Stacking Limits**
    *   **Issue:** Assuming XoL and QS work independently.
    *   **Reality:** Order matters. Usually, "Gross" -> "Inures to Benefit of" -> "Net".
    *   *Standard:* XoL usually applies to the *Net* of QS (or vice versa depending on contract).

2.  **Trap: ALAE Treatment**
    *   **Issue:** Are legal expenses (ALAE) included in the limit?
    *   **Reality:** "Pro Rata" (shared) vs. "Within Limits" (eats the limit). Huge impact on pricing.

### 6.2 Implementation Challenges

1.  **IBNR Allocation:**
    *   Allocating bulk IBNR to specific reinsurance treaties is difficult because we don't know which specific claims will breach the retention.

---

## 7. Advanced Topics & Extensions

### 7.1 Finite Reinsurance

*   Deals that transfer little timing/underwriting risk and are mostly for financial engineering (smoothing earnings).
*   **Regulation:** Highly scrutinized (and often banned) if risk transfer is insufficient (< 10% probability of 10% loss).

### 7.2 Optimization

*   Finding the optimal retention $R$ and limit $L$ to minimize the cost of capital + ceded premium, subject to a risk appetite constraint (e.g., Prob(Insolvency) < 0.5%).

---

## 8. Regulatory & Governance Considerations

### 8.1 Schedule F (US Statutory)

*   A massive schedule in the Annual Statement detailing every reinsurer, ceded premium, and recoverable.
*   **Penalty:** If a reinsurer is unauthorized or slow to pay, the cedant takes a penalty to surplus.

### 8.2 Transfer of Risk Test

*   Auditors (and FASB 113) require proving that "it is reasonably possible that the reinsurer may realize a significant loss."

---

## 9. Practical Example

### 9.1 Worked Example: Exposure Rating (Pareto)

**Scenario:**
*   Total Expected Loss: $10M.
*   Profile: Commercial Liability.
*   Layer: $4M xs $1M.
*   Curve: Pareto with $\alpha = 2.0$.

**Calculation:**
1.  **CDF:** $F(x) = 1 - (\frac{\theta}{x+\theta})^\alpha$. Let's assume $\theta$ is calibrated such that Mean = $10k (per claim).
    *   *Actually, Exposure Rating uses "Exposure Factors" ($G(x)$).*
    *   Let's use a simplified table lookup logic.

    *   $G(1M) = 85\%$ (85% of total losses are below 1M).
    *   $G(5M) = 95\%$ (95% of total losses are below 5M).

2.  **Loss in Layer:**
    *   Losses below 5M = 95%.
    *   Losses below 1M = 85%.
    *   Losses in Layer (1M to 5M) = $95\% - 85\% = 10\%$.

3.  **Expected Layer Loss:**
    $$ 10\% \times \$10M = \$1M $$

4.  **Premium:**
    *   Expected Loss: $1M.
    *   Expense/Profit Load (e.g., 1.4x): $1.4M Premium.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Treaty vs. Fac:** Portfolio vs. Individual.
2.  **QS vs. XoL:** Sharing from dollar one vs. Protecting the tail.
3.  **Pricing:** Experience (History) vs. Exposure (Curves).

### 10.2 When to Use This Knowledge
*   **Capital Management:** Reducing volatility.
*   **Pricing:** Pricing the "Net" cost of a product.
*   **Corporate Strategy:** Deciding how much risk to retain.

### 10.3 Critical Success Factors
1.  **Understand the Contract:** "Inuring," "Reinstatement," "Hours Clause" (for Cats).
2.  **Check Counterparty Credit:** Reinsurance is useless if they don't pay.
3.  **Model the Tail:** XoL pricing is pure tail estimation.

### 10.4 Further Reading
*   **Clark:** "Basics of Reinsurance Pricing" (CAS Study Note).
*   **Swiss Re:** "Proportional and Non-Proportional Reinsurance".

---

## Appendix

### A. Glossary
*   **Cedant:** The primary insurer buying reinsurance.
*   **Retention:** The deductible for the insurer.
*   **Bordereau:** Report of risks/claims.
*   **Retrocession:** Reinsurance for reinsurers.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **QS Cession** | $\alpha \times \text{Gross}$ | Proportional |
| **XoL Recovery** | $\max(0, \min(L, X-R))$ | Non-Prop |
| **Exposure Factor** | $\frac{E[X \wedge L]}{E[X]}$ | Pricing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
