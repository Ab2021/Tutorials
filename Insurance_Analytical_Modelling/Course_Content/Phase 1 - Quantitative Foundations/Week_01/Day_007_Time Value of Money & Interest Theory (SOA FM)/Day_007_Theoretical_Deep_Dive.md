# Time Value of Money & Interest Theory - Theoretical Deep Dive

## Overview
This session covers the fundamental principles of time value of money (TVM) and interest theory, which underpin all financial mathematics in actuarial science. These concepts are essential for SOA Exam FM, CAS Exam 2, pricing insurance products, valuing liabilities, and investment analysis. We explore present value, future value, annuities, effective vs. nominal interest rates, and their applications in insurance.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Time Value of Money (TVM):** The principle that money available today is worth more than the same amount in the future due to its earning potential. A dollar today can be invested to earn interest, making it worth more than a dollar received in the future.

**Interest:** The compensation paid for the use of money over time. It represents the time value of money in quantitative terms.

**Key Terminology:**
- **Principal (P):** Initial amount of money invested or borrowed
- **Interest Rate (i):** Rate at which money grows per period (e.g., 5% per year)
- **Accumulation Function (a(t)):** Amount to which $1 grows after time $t$
- **Discount Factor (v):** Present value of $1 due in one period; $v = 1/(1+i)$
- **Present Value (PV):** Current worth of a future cash flow
- **Future Value (FV):** Amount to which a current investment will grow
- **Annuity:** Series of equal payments at regular intervals
- **Effective Interest Rate:** Actual annual rate accounting for compounding
- **Nominal Interest Rate:** Stated annual rate before considering compounding frequency

### 1.2 Historical Context & Evolution

**Origin:**
- **Ancient Times:** Interest has been charged since Mesopotamia (~3000 BC); Code of Hammurabi (1750 BC) regulated interest rates
- **Medieval Period:** Religious prohibitions on usury (charging interest); led to creative financial instruments
- **17th Century:** Compound interest formalized; Edmond Halley (1693) used compound interest in life tables

**Evolution:**
- **1700s-1800s:** Actuaries developed annuity formulas for pension and life insurance valuation
- **1900s:** Interest theory became standardized in actuarial exams
- **1970s-Present:** Financial calculators and computers enabled complex cash flow analysis; derivatives pricing (Black-Scholes) extended interest theory

**Current State:**
Modern actuarial practice uses:
- **Deterministic Interest:** Fixed rates for pricing and reserving (conservative)
- **Stochastic Interest:** Random interest rates for risk management (Economic Scenario Generators)
- **Negative Interest Rates:** Recent phenomenon in some economies (e.g., Japan, EU)

### 1.3 Why This Matters

**Business Impact:**
- **Pricing:** Life insurance premiums are present value of future benefits minus present value of future premiums
- **Reserving:** Reserves are present value of future claim payments
- **Investment:** Insurers invest premiums; returns depend on interest rates
- **Profitability:** Profit emerges as the difference between actual and assumed interest rates

**Regulatory Relevance:**
- **Statutory Reserves:** Use prescribed interest rates (conservative, e.g., 3-4%)
- **IFRS 17:** Discount rates based on yield curves (market-consistent)
- **Solvency II:** Risk-free rates from EIOPA; interest rate risk is a major SCR component

**Industry Adoption:**
- **Life Insurance:** Critical for pricing, reserving, and embedded value calculations
- **Annuities:** Entire product is based on time value of money
- **P&C Insurance:** Less critical (short-tail), but still used for discounting reserves
- **Pensions:** Defined benefit obligations are present value of future payments

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1. **Assumption: Interest Rates are Known and Constant**
   - **Description:** We assume a fixed interest rate $i$ for all periods
   - **Implication:** Simplifies calculations; enables closed-form formulas
   - **Real-world validity:** Violated in practice; interest rates fluctuate (yield curves, economic cycles)

2. **Assumption: No Transaction Costs or Taxes**
   - **Description:** Money can be invested or borrowed at rate $i$ without friction
   - **Implication:** Present value calculations are symmetric
   - **Real-world validity:** Real markets have bid-ask spreads, taxes, and fees

3. **Assumption: Compound Interest (Unless Stated Otherwise)**
   - **Description:** Interest earns interest (reinvestment assumption)
   - **Implication:** Exponential growth over time
   - **Real-world validity:** Standard assumption; simple interest is rare (only for very short periods)

4. **Assumption: Payments Occur at Specific Times**
   - **Description:** Cash flows occur at discrete points (end of year, beginning of year)
   - **Implication:** Timing of payments matters significantly
   - **Real-world validity:** Generally valid; continuous payments are approximations

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $i$ | Effective annual interest rate | 0.05 (5%) |
| $i^{(m)}$ | Nominal annual rate compounded $m$ times per year | 0.06 compounded monthly |
| $d$ | Effective annual discount rate | $d = i/(1+i)$ |
| $v$ | Discount factor | $v = 1/(1+i)$ |
| $a(t)$ | Accumulation function (value of $1 at time $t$) | $a(t) = (1+i)^t$ |
| $PV$ | Present value | $1,000 |
| $FV$ | Future value | $1,276.28 |
| $n$ | Number of periods | 5 years |
| $a_{\overline{n}|}$ | Present value of annuity-immediate for $n$ periods | $\frac{1-v^n}{i}$ |
| $\ddot{a}_{\overline{n}|}$ | Present value of annuity-due for $n$ periods | $\frac{1-v^n}{d}$ |
| $s_{\overline{n}|}$ | Future value of annuity-immediate for $n$ periods | $\frac{(1+i)^n - 1}{i}$ |

### 2.3 Core Equations & Derivations

#### Equation 1: Future Value (Compound Interest)
$$
FV = PV \times (1 + i)^n
$$

**Where:**
- $PV$ = Present value (initial amount)
- $i$ = Effective interest rate per period
- $n$ = Number of periods

**Derivation:**
- After 1 period: $PV(1+i)$
- After 2 periods: $PV(1+i)(1+i) = PV(1+i)^2$
- After $n$ periods: $PV(1+i)^n$

**Example:**
Invest $1,000 at 5% annual interest for 5 years:
$$
FV = 1000 \times (1.05)^5 = 1000 \times 1.27628 = \$1,276.28
$$

#### Equation 2: Present Value
$$
PV = \frac{FV}{(1+i)^n} = FV \times v^n
$$

**Where:** $v = \frac{1}{1+i}$ is the discount factor.

**Intuition:** Present value "discounts" future cash flows back to today.

**Example:**
What is the present value of $1,000 due in 5 years at 5% interest?
$$
PV = \frac{1000}{(1.05)^5} = \frac{1000}{1.27628} = \$783.53
$$

**Interpretation:** $783.53 today is equivalent to $1,000 in 5 years at 5% interest.

#### Equation 3: Effective vs. Nominal Interest Rates
If interest is compounded $m$ times per year at nominal rate $i^{(m)}$, the effective annual rate $i$ is:
$$
1 + i = \left(1 + \frac{i^{(m)}}{m}\right)^m
$$

**Solving for effective rate:**
$$
i = \left(1 + \frac{i^{(m)}}{m}\right)^m - 1
$$

**Example:**
Nominal rate of 6% compounded monthly:
$$
i = \left(1 + \frac{0.06}{12}\right)^{12} - 1 = (1.005)^{12} - 1 = 1.06168 - 1 = 0.06168 = 6.168\%
$$

**Interpretation:** 6% nominal compounded monthly is equivalent to 6.168% effective annual rate.

#### Equation 4: Present Value of Annuity-Immediate
An **annuity-immediate** pays $1 at the end of each period for $n$ periods. Its present value is:
$$
a_{\overline{n}|} = \frac{1 - v^n}{i} = \frac{1 - (1+i)^{-n}}{i}
$$

**Derivation:**
$$
a_{\overline{n}|} = v + v^2 + \cdots + v^n = v \frac{1 - v^n}{1 - v} = v \frac{1 - v^n}{1 - 1/(1+i)} = \frac{1 - v^n}{i}
$$

**Example:**
Present value of $1,000 per year for 10 years at 5%:
$$
PV = 1000 \times a_{\overline{10}|} = 1000 \times \frac{1 - (1.05)^{-10}}{0.05} = 1000 \times \frac{1 - 0.61391}{0.05} = 1000 \times 7.7217 = \$7,721.73
$$

#### Equation 5: Present Value of Annuity-Due
An **annuity-due** pays $1 at the beginning of each period for $n$ periods. Its present value is:
$$
\ddot{a}_{\overline{n}|} = \frac{1 - v^n}{d} = (1+i) \times a_{\overline{n}|}
$$

**Where:** $d = \frac{i}{1+i}$ is the effective discount rate.

**Relationship:**
$$
\ddot{a}_{\overline{n}|} = 1 + a_{\overline{n-1}|}
$$

**Example:**
Present value of $1,000 per year (paid at start of year) for 10 years at 5%:
$$
PV = 1000 \times \ddot{a}_{\overline{10}|} = 1000 \times (1.05) \times 7.7217 = 1000 \times 8.1078 = \$8,107.82
$$

**Interpretation:** Annuity-due is worth more because payments start immediately.

#### Equation 6: Future Value of Annuity-Immediate
$$
s_{\overline{n}|} = \frac{(1+i)^n - 1}{i}
$$

**Derivation:**
$$
s_{\overline{n}|} = (1+i)^{n-1} + (1+i)^{n-2} + \cdots + 1 = \frac{(1+i)^n - 1}{i}
$$

**Example:**
Future value of saving $1,000 per year for 10 years at 5%:
$$
FV = 1000 \times s_{\overline{10}|} = 1000 \times \frac{(1.05)^{10} - 1}{0.05} = 1000 \times \frac{1.62889 - 1}{0.05} = 1000 \times 12.5779 = \$12,577.89
$$

#### Equation 7: Perpetuity (Infinite Annuity)
A **perpetuity** pays $1 per period forever. Its present value is:
$$
a_{\overline{\infty}|} = \frac{1}{i}
$$

**Derivation:**
$$
a_{\overline{\infty}|} = \lim_{n \to \infty} \frac{1 - v^n}{i} = \frac{1}{i} \quad \text{(since } v^n \to 0 \text{ as } n \to \infty \text{)}
$$

**Example:**
Present value of $1,000 per year forever at 5%:
$$
PV = \frac{1000}{0.05} = \$20,000
$$

**Interpretation:** $20,000 invested at 5% generates $1,000 per year indefinitely.

#### Equation 8: Continuous Compounding
If interest is compounded continuously at force of interest $\delta$, the accumulation function is:
$$
a(t) = e^{\delta t}
$$

**Relationship to effective rate:**
$$
1 + i = e^\delta \quad \Rightarrow \quad \delta = \ln(1 + i)
$$

**Example:**
If $i = 5\%$, then $\delta = \ln(1.05) = 0.04879 = 4.879\%$.

### 2.4 Special Cases & Variants

**Case 1: Deferred Annuity**
An annuity that starts after $m$ periods and pays for $n$ periods:
$$
PV = v^m \times a_{\overline{n}|}
$$

**Case 2: Increasing Annuity**
Payments increase by $1 each period: $1, 2, 3, \ldots, n$.
$$
(Ia)_{\overline{n}|} = \frac{\ddot{a}_{\overline{n}|} - nv^n}{i}
$$

**Case 3: Geometric Annuity**
Payments grow at rate $g$ per period.
$$
PV = \frac{1 - \left(\frac{1+g}{1+i}\right)^n}{i - g} \quad \text{(if } i \neq g \text{)}
$$

**Case 4: Simple Interest**
Interest is not reinvested (linear growth):
$$
FV = PV \times (1 + n \times i)
$$

---

## 3. Theoretical Properties

### 3.1 Key Properties

1. **Property: Additive Property of Present Values**
   - **Statement:** $PV(CF_1 + CF_2) = PV(CF_1) + PV(CF_2)$
   - **Proof:** Linearity of discounting
   - **Practical Implication:** Can value complex cash flows by summing individual present values

2. **Property: Time Consistency**
   - **Statement:** $PV_0(CF_t) = PV_0(PV_s(CF_t))$ for $0 < s < t$
   - **Proof:** $(1+i)^{-t} = (1+i)^{-s} \times (1+i)^{-(t-s)}$
   - **Practical Implication:** Can discount in stages (e.g., to intermediate date, then to present)

3. **Property: Annuity Relationship**
   - **Statement:** $\ddot{a}_{\overline{n}|} = (1+i) \times a_{\overline{n}|}$
   - **Proof:** Annuity-due is annuity-immediate shifted one period earlier
   - **Practical Implication:** Easy conversion between annuity types

4. **Property: Perpetuity Limit**
   - **Statement:** $\lim_{n \to \infty} a_{\overline{n}|} = \frac{1}{i}$
   - **Proof:** $v^n \to 0$ as $n \to \infty$
   - **Practical Implication:** Long-term annuities (e.g., 100 years) approximate perpetuities

### 3.2 Strengths
✓ **Universal:** Applies to all financial calculations
✓ **Precise:** Exact formulas for standard cash flows
✓ **Composable:** Complex cash flows can be built from simple components
✓ **Testable:** SOA/CAS exams extensively cover these formulas
✓ **Practical:** Directly used in pricing, reserving, and investment analysis

### 3.3 Limitations
✗ **Constant Interest Rate:** Real rates fluctuate (yield curves, economic cycles)
✗ **Deterministic:** Ignores interest rate risk (stochastic models needed for risk management)
✗ **No Default Risk:** Assumes all payments are certain (credit risk not modeled)
✗ **Liquidity:** Assumes assets can be bought/sold at fair value (not always true)

### 3.4 Comparison of Interest Rate Types

| Type | Formula | Use Case | Example |
|------|---------|----------|---------|
| **Effective Annual (i)** | $FV = PV(1+i)^n$ | Standard for comparisons | 5% effective |
| **Nominal (i^(m))** | $i = (1 + i^{(m)}/m)^m - 1$ | Quoted rates (APR) | 6% compounded monthly |
| **Discount Rate (d)** | $d = i/(1+i)$ | Annuity-due calculations | 4.76% |
| **Force of Interest (δ)** | $\delta = \ln(1+i)$ | Continuous compounding | 4.88% |
| **Simple Interest** | $FV = PV(1 + ni)$ | Very short-term (< 1 year) | 5% simple |

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

**For Pricing/Reserving:**
- **Interest Rate Assumption:** Typically 3-5% for life insurance, 2-4% for annuities
- **Yield Curve:** Term structure of interest rates (1-year, 5-year, 10-year, 30-year)
- **Historical Data:** Past interest rates for calibration

**For Investment Analysis:**
- **Asset Returns:** Historical returns on bonds, stocks, real estate
- **Liability Cash Flows:** Timing and amount of future payments

**Data Quality Considerations:**
- **Accuracy:** Interest rates must be precise (even 0.1% difference matters over long periods)
- **Consistency:** Use same interest basis (effective vs. nominal) throughout
- **Timeliness:** Rates change; use current rates for pricing, historical for backtesting

### 4.2 Preprocessing Steps

**Step 1: Convert Nominal to Effective**
```
If given nominal rate i^(m) compounded m times per year:
  i_effective = (1 + i^(m)/m)^m - 1
```

**Step 2: Calculate Discount Factor**
```
v = 1 / (1 + i)
```

**Step 3: Set Up Cash Flow Timeline**
```
Time:     0    1    2    3    ...   n
Cash Flow: CF0  CF1  CF2  CF3  ...  CFn
```

**Step 4: Discount Each Cash Flow**
```
PV = CF0 + CF1*v + CF2*v^2 + CF3*v^3 + ... + CFn*v^n
```

### 4.3 Model Specification

**Present Value of General Cash Flow:**
$$
PV = \sum_{t=0}^n CF_t \times v^t
$$

**Software Implementation:**
```python
import numpy as np

def present_value(cash_flows, interest_rate):
    """
    Calculate present value of cash flows
    
    Parameters:
    cash_flows: List or array of cash flows [CF0, CF1, CF2, ...]
    interest_rate: Effective annual interest rate (e.g., 0.05 for 5%)
    
    Returns:
    Present value
    """
    v = 1 / (1 + interest_rate)
    times = np.arange(len(cash_flows))
    discount_factors = v ** times
    pv = np.sum(cash_flows * discount_factors)
    return pv

# Example: Cash flows of $100, $200, $300 at times 1, 2, 3
cash_flows = np.array([0, 100, 200, 300])
i = 0.05
pv = present_value(cash_flows, i)
print(f"Present Value: ${pv:.2f}")

# Annuity functions
def annuity_immediate_pv(payment, n, i):
    """Present value of annuity-immediate"""
    v = 1 / (1 + i)
    return payment * (1 - v**n) / i

def annuity_due_pv(payment, n, i):
    """Present value of annuity-due"""
    return (1 + i) * annuity_immediate_pv(payment, n, i)

def annuity_immediate_fv(payment, n, i):
    """Future value of annuity-immediate"""
    return payment * ((1 + i)**n - 1) / i

# Example: $1,000 per year for 10 years at 5%
payment = 1000
n = 10
i = 0.05

pv_immediate = annuity_immediate_pv(payment, n, i)
pv_due = annuity_due_pv(payment, n, i)
fv_immediate = annuity_immediate_fv(payment, n, i)

print(f"PV Annuity-Immediate: ${pv_immediate:.2f}")
print(f"PV Annuity-Due: ${pv_due:.2f}")
print(f"FV Annuity-Immediate: ${fv_immediate:.2f}")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1. **Present Value:** Current worth of future cash flows
2. **Future Value:** Amount to which current investment will grow
3. **Annuity Value:** PV or FV of series of payments

**Example Output:**
- Present Value of $1,000 in 5 years at 5%: $783.53
- Future Value of $1,000 today in 5 years at 5%: $1,276.28
- PV of $1,000/year for 10 years at 5%: $7,721.73

**Interpretation:**
- **PV:** Amount to invest today to receive future cash flows
- **FV:** Amount accumulated from current investment
- **Annuity PV:** Lump sum equivalent to stream of payments

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Consistency Checks:**
- **PV-FV Relationship:** $FV = PV \times (1+i)^n$ should hold
- **Annuity Relationship:** $\ddot{a}_{\overline{n}|} = (1+i) \times a_{\overline{n}|}$ should hold
- **Perpetuity Limit:** For large $n$, $a_{\overline{n}|} \approx 1/i$

**Sensitivity Analysis:**
- Vary interest rate by ±1%
- Measure impact on PV

### 5.2 Performance Metrics

**For Interest Rate Assumptions:**
- **Actual vs. Assumed:** Compare actual investment returns to assumed rate
- **Profit/Loss:** Difference between actual and assumed creates profit or loss

### 5.3 Validation Techniques

**Backtesting:**
- Use historical interest rates
- Calculate PV of past liabilities
- Compare to actual payments made

**Stress Testing:**
- Scenario: Interest rates drop by 2%
- Recalculate PV of liabilities
- Assess impact on reserves and capital

### 5.4 Sensitivity Analysis

| Interest Rate | PV of $1,000 in 10 years | Change from Base (5%) |
|---------------|--------------------------|----------------------|
| 3% | $744.09 | +19.4% |
| 4% | $675.56 | +8.4% |
| 5% (Base) | $613.91 | 0% |
| 6% | $558.39 | -9.0% |
| 7% | $508.35 | -17.2% |

**Interpretation:** PV is very sensitive to interest rate assumptions, especially for long durations.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1. **Trap: Confusing Nominal and Effective Rates**
   - **Why it's tricky:** 6% compounded monthly ≠ 6% effective annual
   - **How to avoid:** Always convert to effective annual rate for comparisons
   - **Example:** 6% monthly = 6.168% effective annual

2. **Trap: Forgetting to Discount Time 0 Cash Flow**
   - **Why it's tricky:** Time 0 cash flow has discount factor $v^0 = 1$ (no discounting)
   - **How to avoid:** Include $CF_0$ without discounting in PV formula
   - **Example:** If there's a $100 payment today, PV includes full $100

3. **Trap: Annuity-Immediate vs. Annuity-Due**
   - **Why it's tricky:** Timing of first payment matters significantly
   - **How to avoid:** Carefully read problem statement ("end of year" vs. "beginning of year")
   - **Example:** $\ddot{a}_{\overline{10}|} = 1.05 \times a_{\overline{10}|}$ (5% higher for annuity-due at 5% interest)

### 6.2 Implementation Challenges

1. **Challenge: Floating-Point Precision**
   - **Symptom:** $(1.05)^{10}$ calculated as 1.6288946... vs. exact value
   - **Diagnosis:** Floating-point arithmetic has limited precision
   - **Solution:** Use high-precision libraries (e.g., `decimal` in Python) for critical calculations

2. **Challenge: Very Long Durations**
   - **Symptom:** $(1.05)^{100}$ is very large; $v^{100}$ is very small
   - **Diagnosis:** Numerical overflow/underflow
   - **Solution:** Use logarithms: $\ln(FV) = \ln(PV) + n \ln(1+i)$

3. **Challenge: Negative Interest Rates**
   - **Symptom:** Some European bonds have negative yields
   - **Diagnosis:** Unusual economic conditions
   - **Solution:** Formulas still work, but $FV < PV$ (money loses value over time)

### 6.3 Interpretation Errors

1. **Error: Thinking Higher Interest Rate Always Means Higher Value**
   - **Wrong:** "10% interest is better than 5%, so PV is higher"
   - **Right:** Higher interest rate means **lower** PV of future cash flows (more discounting)

2. **Error: Adding PVs from Different Time Points**
   - **Wrong:** $PV_0 + PV_5 = Total PV$
   - **Right:** Must discount all cash flows to the same time point before adding

### 6.4 Edge Cases

**Edge Case 1: Zero Interest Rate**
- **Problem:** If $i = 0$, then $a_{\overline{n}|} = n$ (no discounting)
- **Workaround:** Formulas still work; PV = sum of undiscounted cash flows

**Edge Case 2: Interest Rate = Growth Rate (Geometric Annuity)**
- **Problem:** If $i = g$, the geometric annuity formula has division by zero
- **Workaround:** Use limit: $PV = n \times \frac{1}{1+i}$

**Edge Case 3: Infinite Duration with $i \leq 0$**
- **Problem:** Perpetuity formula $1/i$ is undefined or negative
- **Workaround:** Perpetuities only make sense for $i > 0$

---

## 7. Advanced Topics & Extensions

### 7.1 Modern Variants

**Extension 1: Stochastic Interest Rates**
- **Key Idea:** Interest rates are random variables (e.g., Vasicek, CIR models)
- **Benefit:** Captures interest rate risk
- **Reference:** Used in Economic Scenario Generators (ESGs) for risk management

**Extension 2: Yield Curves**
- **Key Idea:** Different interest rates for different maturities (term structure)
- **Benefit:** More realistic than flat interest rate
- **Reference:** Used in IFRS 17 discounting, bond pricing

**Extension 3: Inflation-Indexed Annuities**
- **Key Idea:** Payments increase with inflation (real vs. nominal rates)
- **Benefit:** Protects against purchasing power erosion
- **Reference:** TIPS (Treasury Inflation-Protected Securities), some pension plans

### 7.2 Integration with Other Methods

**Combination 1: TVM + Probability (Actuarial Present Value)**
- **Use Case:** Value life insurance benefits (uncertain timing)
- **Example:** $APV = \sum_{t=1}^{\omega} v^t \times {}_tp_x \times q_{x+t} \times B_t$

**Combination 2: TVM + Optimization**
- **Use Case:** Asset-liability matching (immunization)
- **Example:** Choose bond portfolio to match duration of liabilities

### 7.3 Cutting-Edge Research

**Topic 1: Negative Interest Rates**
- **Description:** Modeling when $i < 0$ (depositors pay to hold money)
- **Reference:** European Central Bank, Bank of Japan policies

**Topic 2: Cryptocurrency and DeFi**
- **Description:** Decentralized finance uses smart contracts for lending/borrowing
- **Reference:** Compound, Aave protocols (variable interest rates)

---

## 8. Regulatory & Governance Considerations

### 8.1 Regulatory Perspective

**Acceptability:**
- **Widely Accepted:** Yes, TVM is universal in finance and insurance
- **Jurisdictions:** All (SOA, CAS, IAA)
- **Documentation Required:** Actuaries must disclose interest rate assumptions in opinions

**Key Regulatory Concerns:**
1. **Concern: Interest Rate Assumptions**
   - **Issue:** Are assumed rates reasonable?
   - **Mitigation:** Use prescribed rates (statutory) or market-based rates (IFRS 17)

2. **Concern: Interest Rate Risk**
   - **Issue:** What if rates change?
   - **Mitigation:** Duration matching, hedging, stress testing

### 8.2 Model Governance

**Model Risk Rating:** Low (for deterministic TVM)
- **Justification:** Formulas are exact and well-established; main risk is in interest rate assumption

**Validation Frequency:** Annual (review interest rate assumptions)

**Key Validation Tests:**
1. **Formula Verification:** Ensure formulas are correctly implemented
2. **Sensitivity Analysis:** Test impact of ±1% interest rate change
3. **Benchmarking:** Compare assumed rates to market rates

### 8.3 Documentation Requirements

**Minimum Documentation:**
- ✓ Interest rate assumption and rationale
- ✓ Effective vs. nominal rate specification
- ✓ Formulas used (annuity, PV, FV)
- ✓ Sensitivity analysis results
- ✓ Comparison to regulatory prescribed rates (if applicable)

---

## 9. Practical Example

### 9.1 Worked Example: Life Insurance Pricing

**Scenario:** A life insurer is pricing a 10-year term life insurance policy for a 40-year-old. The policy pays $100,000 upon death. Annual premiums are paid at the beginning of each year. Assume:
- Mortality rate: $q_{40+t} = 0.001 \times 1.05^t$ (increasing with age)
- Interest rate: 4% effective annual

**Task:** Calculate the annual premium using equivalence principle (PV premiums = PV benefits).

**Step 1: Calculate PV of Benefits**

The benefit is paid at the end of the year of death. The probability of death in year $t$ is:
$$
{}_tp_{40} \times q_{40+t}
$$

where ${}_tp_{40}$ is the probability of surviving to age $40+t$.

For simplicity, assume deaths occur at end of year. The actuarial present value (APV) of benefits is:
$$
APV_{benefits} = \sum_{t=1}^{10} v^t \times {}_tp_{40} \times q_{40+t} \times 100,000
$$

**Step 2: Calculate Survival Probabilities**

$$
{}_tp_{40} = \prod_{k=0}^{t-1} (1 - q_{40+k})
$$

For $q_{40+k} = 0.001 \times 1.05^k$:
- ${}_{1}p_{40} = 1 - 0.001 = 0.999$
- ${}_{2}p_{40} = 0.999 \times (1 - 0.001 \times 1.05) = 0.999 \times 0.998950 = 0.997951$
- ... (continue for all years)

**Step 3: Calculate APV of Benefits (Numerical)**

| Year $t$ | $q_{40+t-1}$ | ${}_tp_{40}$ | $v^t$ | $APV_t$ |
|----------|--------------|--------------|-------|---------|
| 1 | 0.00100 | 1.000 | 0.96154 | $96.15 |
| 2 | 0.00105 | 0.999 | 0.92456 | $97.03 |
| 3 | 0.00110 | 0.998 | 0.88900 | $97.78 |
| ... | ... | ... | ... | ... |
| 10 | 0.00155 | 0.986 | 0.67556 | $103.46 |

**Total APV of Benefits:** $\approx \$1,050$ (sum of all years)

**Step 4: Calculate PV of Premiums**

Premiums are paid at the beginning of each year (annuity-due) for 10 years, but only if alive:
$$
APV_{premiums} = P \times \sum_{t=0}^{9} v^t \times {}_tp_{40}
$$

This is approximately:
$$
APV_{premiums} = P \times \ddot{a}_{\overline{10}|} \times \text{(survival adjustment)}
$$

For simplicity, assume all survive (conservative):
$$
\ddot{a}_{\overline{10}|} = \frac{1 - v^{10}}{d} = \frac{1 - (1.04)^{-10}}{0.04/1.04} = \frac{0.32473}{0.03846} = 8.4353
$$

**Step 5: Set APV Premiums = APV Benefits**
$$
P \times 8.4353 = 1,050
$$
$$
P = \frac{1,050}{8.4353} = \$124.48
$$

**Conclusion:** The annual premium is approximately $124.48.

**Interpretation:** The insurer collects $124.48 per year for 10 years (if the insured survives), which has a present value equal to the expected present value of the $100,000 death benefit.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1. **Time value of money: $1 today > $1 tomorrow** due to earning potential
2. **Present value discounts future cash flows:** $PV = FV \times v^n$
3. **Annuities have closed-form formulas:** $a_{\overline{n}|} = \frac{1-v^n}{i}$
4. **Effective vs. nominal rates:** Must convert for comparisons
5. **Applications in insurance:** Pricing, reserving, investment analysis

### 10.2 When to Use This Knowledge
**Ideal For:**
- ✓ SOA Exam FM / CAS Exam 2 preparation
- ✓ Life insurance pricing and reserving
- ✓ Annuity valuation
- ✓ Investment analysis (bond pricing, NPV)
- ✓ Pension obligation valuation

**Not Ideal For:**
- ✗ Short-term P&C insurance (minimal discounting)
- ✗ Stochastic modeling (use ESGs)
- ✗ Credit risk (use credit spreads)

### 10.3 Critical Success Factors
1. **Master the Formulas:** Memorize $a_{\overline{n}|}, \ddot{a}_{\overline{n}|}, s_{\overline{n}|}$
2. **Practice Conversions:** Nominal ↔ Effective, Immediate ↔ Due
3. **Check Units:** Ensure interest rate period matches payment period
4. **Use Financial Calculator:** TI BA II Plus or HP 12C for exams
5. **Understand Intuition:** Don't just memorize; understand why formulas work

### 10.4 Further Reading
- **Textbook:** "The Theory of Interest" by Stephen Kellison
- **Exam Prep:** Coaching Actuaries Adapt for Exam FM
- **SOA Exam FM Tables:** Memorize key annuity values
- **Online:** Khan Academy Finance & Capital Markets
- **Advanced:** "Interest Rate Models" by Damiano Brigo & Fabio Mercurio

---

## Appendix

### A. Glossary
- **Accumulation:** Growth of money over time
- **Discount:** Reduction of future value to present value
- **Annuity-Certain:** Fixed number of payments (no contingency)
- **Life Annuity:** Payments contingent on survival
- **Perpetuity:** Infinite annuity
- **Force of Interest:** Continuous compounding rate

### B. Key Formulas Summary

| Formula | Equation | Use |
|---------|----------|-----|
| **Future Value** | $FV = PV(1+i)^n$ | Accumulation |
| **Present Value** | $PV = FV \times v^n$ | Discounting |
| **Effective Rate** | $i = (1 + i^{(m)}/m)^m - 1$ | Convert nominal to effective |
| **Annuity-Immediate PV** | $a_{\overline{n}\|} = \frac{1-v^n}{i}$ | PV of payments at end of period |
| **Annuity-Due PV** | $\ddot{a}_{\overline{n}\|} = \frac{1-v^n}{d}$ | PV of payments at start of period |
| **Annuity-Immediate FV** | $s_{\overline{n}\|} = \frac{(1+i)^n-1}{i}$ | FV of payments at end of period |
| **Perpetuity** | $a_{\overline{\infty}\|} = \frac{1}{i}$ | PV of infinite payments |

### C. Financial Calculator Guide (TI BA II Plus)

**Present Value:**
```
N = 10 (number of periods)
I/Y = 5 (interest rate %)
PMT = 0
FV = 1000
CPT PV → -613.91
```

**Annuity:**
```
N = 10
I/Y = 5
PMT = 1000
FV = 0
CPT PV → -7721.73 (annuity-immediate)
```

**For annuity-due:** Set to BGN mode (2ND PMT, 2ND ENTER)

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,250+*
