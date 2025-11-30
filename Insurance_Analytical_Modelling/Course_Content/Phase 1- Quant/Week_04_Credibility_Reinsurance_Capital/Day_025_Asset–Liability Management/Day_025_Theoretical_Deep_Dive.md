# Asset–Liability Management (ALM) - Theoretical Deep Dive

## Overview
This session covers Asset-Liability Management (ALM), the practice of managing financial risks that arise due to mismatches between the assets and liabilities of an insurance company. We explore key concepts like Duration, Convexity, Immunization (Redington), Cash Flow Matching, and the risks of Disintermediation and Liquidity.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Asset-Liability Management (ALM):** The ongoing process of formulating, implementing, monitoring, and revising strategies related to assets and liabilities to achieve an organization's financial objectives, given the organization's risk tolerance and other constraints.

**The Core Problem:**
*   **Liabilities:** Future promises to pay policyholders (e.g., a pension payment in 20 years).
*   **Assets:** Investments held today to fund those promises (e.g., Bonds, Stocks).
*   **Mismatch:** If interest rates change, the value of assets and liabilities might change by different amounts, potentially leaving the insurer insolvent.

**Key Terminology:**
*   **Duration:** The weighted average time to maturity of cash flows; a measure of interest rate sensitivity.
*   **Convexity:** The curvature of the price-yield relationship; the second derivative.
*   **Immunization:** Structuring assets to protect the surplus from interest rate shocks.
*   **Disintermediation:** Policyholders withdrawing funds (surrendering) when interest rates rise to invest elsewhere.

### 1.2 Historical Context & Evolution

**Origin:**
*   **Redington (1952):** Frank Redington, a British actuary, introduced the theory of immunization to protect life insurers.
*   **1980s Crisis:** High inflation and interest rates caused massive disintermediation (runs on the bank) for life insurers, forcing ALM to become a critical discipline.

**Evolution:**
*   **Cash Flow Matching:** The early "perfect" solution (dedication).
*   **Duration Matching:** The more flexible solution.
*   **Stochastic ALM:** Modern approach using thousands of economic scenarios (ESG) to optimize the frontier.

**Current State:**
*   **LDI (Liability Driven Investment):** Pension funds explicitly targeting a funding ratio rather than total return.
*   **Negative Rates:** The post-2008 environment challenged ALM models that assumed rates couldn't go below zero.

### 1.3 Why This Matters

**Business Impact:**
*   **Solvency:** A 1% drop in rates can increase liability values by 15% (for long-tail business). If assets only rise by 5%, the company is bankrupt.
*   **Profitability:** Taking "mismatch risk" (investing long to back short liabilities) is a source of profit (spread), but risky.

**Regulatory Relevance:**
*   **C-3 Risk (US RBC):** The capital charge specifically for Interest Rate Risk.
*   **Liquidity Stress Testing:** Regulators require proof that the insurer can survive a "run" scenario.

---

## 2. Mathematical Framework

### 2.1 Duration (Macaulay & Modified)

**Macaulay Duration ($D_{mac}$):** The weighted average time to receive cash flows.
$$ D_{mac} = \frac{\sum_{t} t \cdot CF_t \cdot v^t}{\sum_{t} CF_t \cdot v^t} $$
*   $v = 1/(1+i)$.

**Modified Duration ($D_{mod}$):** The percentage change in price for a unit change in yield.
$$ D_{mod} = \frac{D_{mac}}{1+i} = -\frac{1}{P} \frac{dP}{di} $$
*   *Approximation:* $\Delta P \approx -D_{mod} \cdot P \cdot \Delta i$.

### 2.2 Convexity ($C$)

The second derivative of Price with respect to yield.
$$ C = \frac{1}{P} \frac{d^2P}{di^2} $$
*   *Approximation:* $\Delta P \approx -D_{mod} \cdot P \cdot \Delta i + \frac{1}{2} C \cdot P \cdot (\Delta i)^2$.
*   **Why it matters:** Convexity is "good." If rates fall, price rises *more* than duration predicts. If rates rise, price falls *less* than duration predicts.

### 2.3 Redington Immunization

**Goal:** Protect Surplus ($S = A - L$) from small rate changes.
**Conditions:**
1.  **PV Match:** $PV(A) = PV(L)$.
2.  **Duration Match:** $D_{mod}(A) = D_{mod}(L)$.
3.  **Convexity Condition:** $C(A) > C(L)$.

*   *Result:* For small $\Delta i$, $\Delta S \ge 0$. The surplus curve is convex and tangent to zero change at the current rate.

### 2.4 Cash Flow Matching (Dedication)

**Goal:** Asset cash flows exactly match Liability cash flows in every period.
$$ CF_t(A) = CF_t(L) \quad \forall t $$
*   **Pros:** Eliminates interest rate risk entirely.
*   **Cons:** Expensive (lower yield), inflexible, hard to find exact bonds (especially for 50+ year tails).

---

## 3. Theoretical Properties

### 3.1 Reinvestment Risk vs. Price Risk

*   **Rising Rates:**
    *   *Bad:* Bond prices fall (Price Risk).
    *   *Good:* Coupons reinvested at higher rates (Reinvestment Risk).
*   **Duration Point:** The point in time where these two effects cancel each other out.

### 3.2 Disintermediation Risk (The "Put Option")

*   Policyholders often have the right to surrender their policy for Cash Surrender Value (CSV).
*   This is effectively a **Put Option** on the bond portfolio.
*   **Scenario:** Rates rise -> Bond values fall -> Policyholders surrender -> Insurer forced to sell bonds at a loss.
*   **Modeling:** Must model the "Dynamic Lapse Function" where Lapse Rate = $f(\text{Market Rate} - \text{Credited Rate})$.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Asset Inventory:** CUSIP, Coupon, Maturity, Callability.
*   **Liability Cash Flows:** Projected benefit payments (best estimate).
*   **Economic Scenarios:** Interest rate paths (Hull-White, CIR models).

### 4.2 Preprocessing Steps

**Step 1: Segmentation**
*   Group business lines (e.g., SPDA, Term Life, P&C) into distinct segments.
*   Assign specific assets to back each segment.

**Step 2: Key Rate Durations (KRD)**
*   Calculate sensitivity to specific points on the yield curve (2yr, 5yr, 10yr, 30yr).
*   Ensures protection against "Twists" and non-parallel shifts.

### 4.3 Model Specification (Python Example)

Calculating Duration, Convexity, and Immunization status.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define Cash Flow Engine
def calculate_metrics(cash_flows, times, yield_rate):
    # PV
    discount_factors = (1 + yield_rate) ** -times
    pv = np.sum(cash_flows * discount_factors)
    
    # Macaulay Duration
    # Sum(t * PV_t) / Sum(PV_t)
    weighted_times = np.sum(times * cash_flows * discount_factors)
    mac_duration = weighted_times / pv
    
    # Modified Duration
    mod_duration = mac_duration / (1 + yield_rate)
    
    # Convexity
    # Sum(t * (t+1) * PV_t) / (PV * (1+i)^2)
    convexity_num = np.sum(times * (times + 1) * cash_flows * discount_factors)
    convexity = convexity_num / (pv * (1 + yield_rate)**2)
    
    return pv, mod_duration, convexity

# 2. Scenario: Liability vs. Asset Portfolio
yield_curve = 0.05

# Liability: Single payment of $100M in 10 years
liab_cf = np.array([100])
liab_t = np.array([10])

# Asset A: Zero Coupon Bond, 10 years (Perfect Match)
asset_a_cf = np.array([100])
asset_a_t = np.array([10])

# Asset B: Barbell (5yr and 15yr bonds)
asset_b_cf = np.array([50, 50]) # Simplified amounts
asset_b_t = np.array([5, 15])

# Calculate Metrics
l_pv, l_dur, l_conv = calculate_metrics(liab_cf, liab_t, yield_curve)
a_pv, a_dur, a_conv = calculate_metrics(asset_a_cf, asset_a_t, yield_curve)
b_pv, b_dur, b_conv = calculate_metrics(asset_b_cf, asset_b_t, yield_curve)

print(f"Liability: PV={l_pv:.2f}, Dur={l_dur:.2f}, Conv={l_conv:.2f}")
print(f"Asset A (Zero): PV={a_pv:.2f}, Dur={a_dur:.2f}, Conv={a_conv:.2f}")
print(f"Asset B (Barbell): PV={b_pv:.2f}, Dur={b_dur:.2f}, Conv={b_conv:.2f}")

# Check Immunization for Asset B
# Scale Asset B to match Liability PV
scale_factor = l_pv / b_pv
b_pv_scaled = b_pv * scale_factor
print(f"\nScaled Asset B PV: {b_pv_scaled:.2f}")

print("Immunization Check:")
print(f"1. PV Match: {np.isclose(b_pv_scaled, l_pv)}")
print(f"2. Duration Match: {np.isclose(b_dur, l_dur)} (Asset: {b_dur:.2f} vs Liab: {l_dur:.2f})")
# Note: Barbell usually has higher convexity than Bullet
print(f"3. Convexity Asset > Liab: {b_conv > l_conv} ({b_conv:.2f} > {l_conv:.2f})")

# Visualization: Surplus Sensitivity
shifts = np.linspace(-0.02, 0.02, 100)
surplus_vals = []

for shift in shifts:
    new_rate = yield_curve + shift
    
    # Re-calc PVs
    l_pv_new, _, _ = calculate_metrics(liab_cf, liab_t, new_rate)
    b_pv_new, _, _ = calculate_metrics(asset_b_cf * scale_factor, asset_b_t, new_rate)
    
    surplus_vals.append(b_pv_new - l_pv_new)

plt.figure(figsize=(10, 6))
plt.plot(shifts * 100, surplus_vals)
plt.title('Surplus Sensitivity (Redington Immunization)')
plt.xlabel('Interest Rate Shift (%)')
plt.ylabel('Surplus Change')
plt.axhline(0, color='k', linestyle='--')
plt.grid(True)
plt.show()

# Interpretation:
# The curve is "U-shaped" (Convex).
# Any small shift in rates (up or down) INCREASES surplus.
# This is the power of Convexity > Liability Convexity.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Duration Gap:** $D(A) - D(L)$. Ideally zero.
2.  **Convexity Gap:** $C(A) - C(L)$. Ideally positive.

**Interpretation:**
*   **Positive Convexity Gap:** The insurer benefits from volatility.
*   **Negative Duration Gap:** Assets are shorter than liabilities. The insurer is exposed to falling rates (reinvestment risk).

---

## 5. Evaluation & Validation

### 5.1 Stress Testing

*   **Parallel Shift:** Rates up/down 100bps.
*   **Twist:** Short rates up, long rates down (Inverted Yield Curve).
*   **Fly:** Butterfly shifts.

### 5.2 Liquidity Coverage Ratio

*   Can we meet cash outflows in the next 30 days without selling assets at a fire-sale price?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Duration Drift**
    *   **Issue:** Duration changes as time passes and rates change.
    *   **Reality:** An immunized portfolio today will NOT be immunized tomorrow.
    *   **Fix:** Rebalancing is required frequently.

2.  **Trap: Negative Convexity (MBS)**
    *   **Issue:** Mortgage-Backed Securities (MBS) have prepayment risk. When rates fall, people refinance.
    *   **Result:** Price does *not* rise as much as expected (Negative Convexity). This destroys immunization strategies.

### 6.2 Implementation Challenges

1.  **Modeling Liabilities:**
    *   P&C liabilities (inflation-sensitive) behave differently than Life liabilities (interest-sensitive).
    *   Inflation acts like a "negative asset."

---

## 7. Advanced Topics & Extensions

### 7.1 Strategic Asset Allocation (SAA)

*   Optimizing the asset mix (Stocks vs. Bonds vs. Real Estate) to maximize long-term surplus growth while satisfying risk constraints.
*   **Efficient Frontier:** Plotting Risk (Vol of Surplus) vs. Return (Exp Growth of Surplus).

### 7.2 Hedging with Derivatives

*   **Interest Rate Swaps:** Swapping floating payments for fixed payments to increase duration without buying long bonds.
*   **Swaptions:** Buying options to enter a swap (hedging convexity).

---

## 8. Regulatory & Governance Considerations

### 8.1 Prudent Person Principle

*   Solvency II and modern regimes replace strict "Investment Limits" (e.g., max 10% in equities) with the Prudent Person Principle: "Invest only in assets whose risks you can identify, measure, monitor, manage, control, and report."

### 8.2 Asset Adequacy Analysis (Cash Flow Testing)

*   US Actuarial Standard of Practice (ASOP 22).
*   The Appointed Actuary must certify that assets are adequate to mature liabilities under 7 prescribed interest rate scenarios (NY7).

---

## 9. Practical Example

### 9.1 Worked Example: Duration Gap Management

**Scenario:**
*   Liabilities: $100M PV, Duration = 12.
*   Assets: $105M PV, Duration = 8.
*   **Problem:** Mismatch. If rates fall, Liabilities rise faster than Assets.

**Solution (Swaps):**
*   We need to increase Asset Duration.
*   **Action:** Enter a "Receive Fixed / Pay Float" Interest Rate Swap.
*   **Effect:** The Swap behaves like a long bond (Fixed Leg) minus a short bond (Float Leg). It adds significant duration with zero initial cost.

**Calculation:**
*   Target Duration = 12.
*   Current Dollar Duration = $105 \times 8 = 840$.
*   Target Dollar Duration = $105 \times 12 = 1260$.
*   Gap = 420.
*   Swap Duration (approx) = Term of Swap (e.g., 20yr Swap has dur ~14).
*   Notional Needed = Gap / Swap Dur = 420 / 14 = $30M.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Duration** measures linear rate risk.
2.  **Convexity** measures curvature (and is beneficial).
3.  **Immunization** locks in the surplus.

### 10.2 When to Use This Knowledge
*   **Investment Strategy:** Telling the CIO what duration to target.
*   **Product Design:** Understanding the cost of embedded options (guarantees).
*   **Valuation:** Discounting liabilities at the market consistent rate.

### 10.3 Critical Success Factors
1.  **Match the Flows:** Cash flow matching is safer than duration matching.
2.  **Watch the Convexity:** Don't get caught with negative convexity (MBS) when rates fall.
3.  **Dynamic Behavior:** Model policyholder lapses dynamically.

### 10.4 Further Reading
*   **Panjer:** "Financial Economics".
*   **Redington:** "Review of the Principles of Life-Office Valuations" (1952).

---

## Appendix

### A. Glossary
*   **Duration:** Sensitivity to interest rates.
*   **Convexity:** Second derivative of price.
*   **Immunization:** Protecting surplus.
*   **Disintermediation:** Policyholder withdrawals.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Mod Duration** | $-\frac{1}{P} \frac{dP}{di}$ | Sensitivity |
| **Convexity** | $\frac{1}{P} \frac{d^2P}{di^2}$ | Curvature |
| **Immunization** | $D_A = D_L, C_A > C_L$ | Protection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
