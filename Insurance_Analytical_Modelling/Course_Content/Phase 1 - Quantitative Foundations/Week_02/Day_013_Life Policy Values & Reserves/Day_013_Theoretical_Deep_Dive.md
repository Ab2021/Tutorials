# Life Policy Values & Reserves - Theoretical Deep Dive

## Overview
This session covers policy values and reserves, which represent the insurer's liability for in-force policies. We explore prospective and retrospective reserve formulas, terminal and initial reserves, modified reserves, and the relationship between reserves and cash values. These concepts are critical for SOA Exam LTAM and statutory reserve calculations.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Policy Value (Reserve):** The amount an insurer must hold at a given time to cover future policy obligations. It represents the difference between the expected present value of future benefits and future premiums.

**Prospective Reserve (Forward-Looking):**
$$
_tV = EPV_t(\text{Future Benefits}) - EPV_t(\text{Future Premiums})
$$

**Retrospective Reserve (Backward-Looking):**
$$
_tV = \text{Accumulated Value of Past Premiums} - \text{Accumulated Value of Past Benefits}
$$

**Key Terminology:**
- **$_tV$ or $V_t$:** Reserve at time $t$ (policy value)
- **Terminal Reserve:** Reserve at end of year (after premium paid, before death claims)
- **Initial Reserve:** Reserve at start of year (before premium paid)
- **Net Premium Reserve:** Reserve based on net premiums
- **Gross Premium Reserve:** Reserve based on gross premiums and actual expenses
- **Modified Reserve:** Reserve using modified premium (e.g., FPT, CRVM)

### 1.2 Historical Context & Evolution

**Origin:**
- **1800s:** Reserves recognized as necessary for solvency
- **1858:** Massachusetts required reserves for life insurers
- **1900s:** Standard valuation laws enacted

**Evolution:**
- **Pre-1940s:** Net level premium reserves
- **1940s-1960s:** Modified reserves (FPT, CRVM) to address first-year strain
- **1980s-2000s:** Principle-Based Reserves (PBR)
- **Present:** Stochastic reserves, IFRS 17

**Current State:**
- **Statutory:** Prescribed formulas (e.g., CRVM, Net Premium)
- **GAAP:** Different methods (e.g., FAS 60, LDTI)
- **IFRS 17:** Market-consistent, current estimates

### 1.3 Why This Matters

**Business Impact:**
- **Solvency:** Reserves ensure insurer can pay claims
- **Capital:** Reserves tie up capital
- **Profitability:** Reserve releases contribute to profit
- **Product Design:** Reserve requirements affect product viability

**Regulatory Relevance:**
- **Statutory Minimum:** Must hold prescribed reserves
- **RBC:** Risk-Based Capital depends on reserves
- **Valuation Actuary:** Must opine on reserve adequacy

**Industry Adoption:**
- **Life Insurance:** Universal use
- **Annuities:** Critical for longevity risk
- **Pensions:** DB liabilities are reserves

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1. **Assumption: Constant Mortality and Interest**
   - **Description:** Life table and interest rate don't change
   - **Implication:** Can use fixed formulas
   - **Real-world validity:** Violated; addressed by margin and PBR

2. **Assumption: Premiums Paid as Scheduled**
   - **Description:** Policyholders pay premiums on time
   - **Implication:** No lapse or surrender
   - **Real-world validity:** Lapses occur; need lapse assumptions

3. **Assumption: Benefits Paid as Specified**
   - **Description:** Death benefits are fixed
   - **Implication:** No changes to policy
   - **Real-world validity:** Generally valid for traditional products

4. **Assumption: Net Premium Basis**
   - **Description:** Reserves based on net premiums (no expenses)
   - **Implication:** Statutory reserves are conservative
   - **Real-world validity:** Gross premium reserves are more realistic

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $_tV$ or $V_t$ | Reserve at time $t$ | $5,000 |
| $_tV_x$ | Reserve for life aged $x$ at time $t$ | $_10V_{60}$ |
| $P$ | Net level annual premium | $1,990 |
| $A_{x+t}$ | Insurance APV at age $x+t$ | 0.45 |
| $\ddot{a}_{x+t}$ | Annuity APV at age $x+t$ | 15.0 |
| $_t^*V$ | Terminal reserve at end of year $t$ | $5,200 |
| $_tV^*$ | Initial reserve at start of year $t$ | $5,000 |

### 2.3 Core Equations & Derivations

#### Equation 1: Prospective Reserve (Whole Life)
$$
_tV_x = A_{x+t} - P_x \times \ddot{a}_{x+t}
$$

**Derivation:**
At time $t$, the reserve is the EPV of future benefits minus EPV of future premiums:
$$
_tV_x = EPV_t(\text{Future Benefits}) - EPV_t(\text{Future Premiums})
$$
$$
= A_{x+t} - P_x \times \ddot{a}_{x+t}
$$

**Example:**
For a whole life policy issued at age 60, calculate reserve at time 10:
- $A_{70} = 0.55$ (insurance APV at age 70)
- $\ddot{a}_{70} = 12.5$ (annuity APV at age 70)
- $P_{60} = 0.0199$ (net annual premium)

$$
_{10}V_{60} = 0.55 - 0.0199 \times 12.5 = 0.55 - 0.249 = 0.301
$$

For $100,000 policy: Reserve = $100,000 \times 0.301 = $30,100.

#### Equation 2: Retrospective Reserve (Whole Life)
$$
_tV_x = P_x \times s_{\overline{t}|} - (A_x - A_{x:\overline{t}|}^{\ \ 1})
$$

**Where:**
- $s_{\overline{t}|}$ = Accumulated value of annuity-immediate for $t$ years
- $A_x - A_{x:\overline{t}|}^{\ \ 1}$ = APV of benefits paid in first $t$ years

**Simplified Form:**
$$
_tV_x = \frac{P_x \times s_{\overline{t}|} \times {}_tp_x - \sum_{k=0}^{t-1} v^{t-k-1} \times {}_kp_x \times q_{x+k}}{_tp_x}
$$

**Equivalence:**
Under consistent assumptions, prospective = retrospective.

#### Equation 3: Recursive Formula (Fackler's Equation)
$$
_{t+1}V_x = (_tV_x + P_x)(1 + i) - q_{x+t} \times (1 + i)
$$

**For unit insurance:**
$$
_{t+1}V_x = \frac{(_tV_x + P_x)(1 + i) - q_{x+t}}{p_{x+t}}
$$

**Derivation:**
- Start with reserve $_tV_x$
- Add premium $P_x$
- Accumulate with interest $(1 + i)$
- Subtract expected death benefit $q_{x+t} \times 1$
- Divide by survival probability $p_{x+t}$

**Example:**
- $_10V_{60} = 0.301$
- $P_{60} = 0.0199$
- $i = 4\%$
- $q_{70} = 0.015$

$$
_{11}V_{60} = \frac{(0.301 + 0.0199)(1.04) - 0.015}{1 - 0.015} = \frac{0.3337 - 0.015}{0.985} = \frac{0.3187}{0.985} = 0.3236
$$

#### Equation 4: Terminal vs. Initial Reserve
**Terminal Reserve (end of year $t$, after premium):**
$$
_t^*V_x = (_tV_x + P_x)
$$

**Initial Reserve (start of year $t$, before premium):**
$$
_tV^*_x = _tV_x
$$

**Relationship:**
$$
_{t+1}V^*_x = \frac{_t^*V_x (1 + i) - q_{x+t}}{p_{x+t}}
$$

#### Equation 5: Reserve for $n$-Year Endowment
$$
_tV_{x:\overline{n}|} = A_{x+t:\overline{n-t}|} - P_{x:\overline{n}|} \times \ddot{a}_{x+t:\overline{n-t}|}
$$

**At maturity ($t = n$):**
$$
_nV_{x:\overline{n}|} = 1 \quad \text{(reserve equals face amount)}
$$

#### Equation 6: Modified Reserve (Full Preliminary Term)
**First-Year Premium:**
$$
\alpha = A_{x:\overline{1}|}^{\ 1}
$$

**Renewal Premium:**
$$
\beta = \frac{A_{x+1} - A_{x:\overline{1}|}^{\ 1}}{\ddot{a}_{x+1}}
$$

**Reserve:**
$$
_1V_x^{FPT} = 0 \quad \text{(zero first-year reserve)}
$$
$$
_tV_x^{FPT} = A_{x+t} - \beta \times \ddot{a}_{x+t} \quad \text{for } t \geq 1
$$

**Purpose:** Reduces first-year strain (high acquisition costs).

#### Equation 7: Commissioners Reserve Valuation Method (CRVM)
$$
_tV_x^{CRVM} = \max(_tV_x^{Net}, _tV_x^{FPT})
$$

**Purpose:** Statutory minimum reserve (prevents deficiency reserves).

### 2.4 Special Cases & Variants

**Case 1: Paid-Up Insurance**
If premiums stop, paid-up amount:
$$
\text{Paid-Up Amount} = \frac{_tV_x}{A_{x+t}}
$$

**Case 2: Cash Surrender Value**
$$
CSV_t = _tV_x - \text{Surrender Charge}
$$

**Case 3: Deficiency Reserve**
If gross premium < net premium:
$$
\text{Deficiency Reserve} = (P_{net} - P_{gross}) \times \ddot{a}_{x+t}
$$

---

## 3. Theoretical Properties

### 3.1 Key Properties

1. **Property: Prospective = Retrospective**
   - **Statement:** Under consistent assumptions, both methods give same reserve
   - **Proof:** Equivalence principle
   - **Practical Implication:** Can use either method for verification

2. **Property: Reserve Increases with Time**
   - **Statement:** $_tV_x < _{t+1}V_x$ for whole life
   - **Proof:** Mortality increases with age
   - **Practical Implication:** Reserves accumulate over time

3. **Property: Endowment Reserve Reaches Face Amount**
   - **Statement:** $_nV_{x:\overline{n}|} = 1$
   - **Proof:** At maturity, must pay face amount
   - **Practical Implication:** Endowment reserves are higher than whole life

4. **Property: Modified Reserve ≤ Net Premium Reserve**
   - **Statement:** $_tV_x^{FPT} \leq _tV_x^{Net}$
   - **Proof:** FPT has lower first-year premium
   - **Practical Implication:** Modified reserves reduce first-year strain

### 3.2 Strengths
✓ **Rigorous:** Based on solid mathematical principles
✓ **Conservative:** Ensures solvency
✓ **Regulatory:** Widely accepted
✓ **Verifiable:** Prospective = retrospective check
✓ **Flexible:** Can model various products

### 3.3 Limitations
✗ **Constant Assumptions:** Mortality and interest assumed fixed
✗ **No Lapses:** Ignores policyholder behavior
✗ **Net Premium Basis:** Doesn't reflect actual expenses
✗ **Deterministic:** Ignores uncertainty

### 3.4 Comparison of Reserve Methods

| Method | First-Year Reserve | Purpose | Use Case |
|--------|-------------------|---------|----------|
| **Net Premium** | High | Theoretical | Exam problems |
| **FPT** | Zero | Reduce strain | Statutory (historical) |
| **CRVM** | Max(Net, FPT) | Statutory minimum | US statutory |
| **Gross Premium** | Realistic | GAAP | Financial reporting |

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

**For Reserve Calculations:**
- **Life Table:** $l_x, q_x, p_x$
- **Interest Rate:** $i$ or $\delta$
- **Premium:** Net or gross
- **Product Specifications:** Benefit, term

**Data Quality Considerations:**
- **Accuracy:** Life table must be appropriate
- **Consistency:** Same assumptions as pricing
- **Completeness:** All ages through limiting age

### 4.2 Preprocessing Steps

**Step 1: Calculate APVs at Current Age**
```
- Calculate A_(x+t) (insurance APV at age x+t)
- Calculate ä_(x+t) (annuity APV at age x+t)
```

**Step 2: Calculate Premium**
```
- P = A_x / ä_x (net level premium)
```

**Step 3: Apply Prospective Formula**
```
- _tV = A_(x+t) - P * ä_(x+t)
```

### 4.3 Model Specification

**Python Implementation:**

```python
import numpy as np

def prospective_reserve(A_xt, a_xt, P):
    """Calculate prospective reserve"""
    return A_xt - P * a_xt

def retrospective_reserve(P, s_t, t_p_x, sum_benefits_paid):
    """Calculate retrospective reserve (simplified)"""
    accumulated_premiums = P * s_t * t_p_x
    return (accumulated_premiums - sum_benefits_paid) / t_p_x

def recursive_reserve(V_t, P, i, q_xt):
    """Calculate reserve at t+1 using Fackler's equation"""
    p_xt = 1 - q_xt
    return ((V_t + P) * (1 + i) - q_xt) / p_xt

def fpt_reserve(A_xt, a_xt, beta, t):
    """Calculate FPT reserve"""
    if t == 0:
        return 0
    else:
        return A_xt - beta * a_xt

# Example usage
# Whole life policy issued at age 60, calculate reserve at time 10

# APVs at age 70
A_70 = 0.55
a_70 = 12.5

# Net premium (calculated at issue)
P_60 = 0.0199

# Prospective reserve at time 10
V_10 = prospective_reserve(A_70, a_70, P_60)
print(f"Reserve at time 10: {V_10:.4f}")

# For $100,000 policy
benefit = 100000
reserve_dollars = benefit * V_10
print(f"Reserve for $100K policy: ${reserve_dollars:,.2f}")

# Verify using recursive formula
i = 0.04
q_70 = 0.015

# Calculate reserve at time 11 from time 10
V_11 = recursive_reserve(V_10, P_60, i, q_70)
print(f"Reserve at time 11 (recursive): {V_11:.4f}")

# Verify using prospective at time 11
A_71 = 0.58
a_71 = 11.8
V_11_prosp = prospective_reserve(A_71, a_71, P_60)
print(f"Reserve at time 11 (prospective): {V_11_prosp:.4f}")

# Project reserves for 20 years
ages = list(range(60, 81))
reserves = [0]  # Initial reserve at issue

# Simplified: use actual APVs (would calculate from life table)
A_values = [0.35 + 0.02 * t for t in range(21)]
a_values = [17.58 - 0.25 * t for t in range(21)]
q_values = [0.005 * (1.05 ** t) for t in range(21)]

for t in range(20):
    V_prosp = prospective_reserve(A_values[t+1], a_values[t+1], P_60)
    reserves.append(V_prosp)
    print(f"Year {t+1}, Age {60+t+1}: Reserve = {V_prosp:.4f} (${benefit * V_prosp:,.2f})")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1. **Reserve per $1:** $_tV_x$
2. **Reserve in Dollars:** Benefit × $_tV_x$
3. **Reserve Projection:** Reserves over policy lifetime

**Example Output (Age 60, $100K Whole Life):**
- Year 1: Reserve = $2,500
- Year 10: Reserve = $30,100
- Year 20: Reserve = $55,000

**Interpretation:**
- **Reserve:** Amount insurer must hold
- **Increasing:** Reserves grow as mortality risk increases
- **Cash Value:** Typically 80-90% of reserve

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Consistency Checks:**
- **Prospective = Retrospective:** Verify both methods give same result
- **Recursive:** Check $_tV$ calculated recursively matches prospective
- **Reasonableness:** $0 < _tV < 1$ for whole life

**Numerical Verification:**
```python
# Check prospective = retrospective
V_prosp = prospective_reserve(A_xt, a_xt, P)
V_retro = retrospective_reserve(P, s_t, t_p_x, sum_benefits)
assert abs(V_prosp - V_retro) < 0.01, "Prospective ≠ Retrospective"
```

### 5.2 Performance Metrics

**For Reserves:**
- **Adequacy:** Actual claims ≤ reserves
- **Accuracy:** Compare to published tables

### 5.3 Validation Techniques

**Benchmarking:**
- Compare calculated reserves to SOA illustrative tables
- Check against alternative methods

**Sensitivity Analysis:**
- Vary interest rate by ±1%
- Vary mortality by ±10%
- Measure impact on reserve

### 5.4 Sensitivity Analysis

| Parameter | Base | +10% | -10% | Impact on $_10V_{60}$ |
|-----------|------|------|------|----------------------|
| Interest Rate | 4% | 4.4% | 3.6% | -12% / +14% |
| Mortality | Base | +10% | -10% | +8% / -8% |

**Interpretation:** Reserves are very sensitive to interest rate assumptions.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1. **Trap: Confusing Reserve and Cash Value**
   - **Why it's tricky:** Cash value < reserve (surrender charge)
   - **How to avoid:** CSV = reserve - surrender charge
   - **Example:** Reserve = $30,000, CSV = $27,000

2. **Trap: Thinking Reserve is Profit**
   - **Why it's tricky:** Reserve is liability, not asset
   - **How to avoid:** Reserve = amount owed to policyholders
   - **Example:** High reserves mean high liabilities

3. **Trap: Forgetting Terminal vs. Initial**
   - **Why it's tricky:** Terminal includes premium; initial doesn't
   - **How to avoid:** $_t^*V = _tV + P$
   - **Example:** Initial = $30,000, Terminal = $32,000 (after $2,000 premium)

### 6.2 Implementation Challenges

1. **Challenge: Numerical Precision**
   - **Symptom:** Prospective ≠ retrospective
   - **Diagnosis:** Rounding errors
   - **Solution:** Use high precision

2. **Challenge: Modified Reserve Complexity**
   - **Symptom:** FPT formulas are complex
   - **Diagnosis:** Different premiums for first year and renewal
   - **Solution:** Carefully track which premium applies

3. **Challenge: Deficiency Reserves**
   - **Symptom:** Gross premium < net premium
   - **Diagnosis:** High expenses
   - **Solution:** Calculate deficiency reserve separately

### 6.3 Interpretation Errors

1. **Error: Thinking Higher Reserve is Better**
   - **Wrong:** "High reserves mean strong company"
   - **Right:** "High reserves mean high liabilities (ties up capital)"

2. **Error: Ignoring Reserve Releases**
   - **Wrong:** "Reserves only increase"
   - **Right:** "Reserves can decrease (e.g., term insurance after peak mortality)"

### 6.4 Edge Cases

**Edge Case 1: Endowment at Maturity**
- **Problem:** $_nV = 1$ (reserve equals face amount)
- **Workaround:** Formula still works

**Edge Case 2: Paid-Up Policy**
- **Problem:** No future premiums
- **Workaround:** Set $P = 0$ in prospective formula

**Edge Case 3: Negative Reserve**
- **Problem:** If gross premium > net premium significantly
- **Workaround:** Shouldn't happen for net premium reserves

---

## 7. Advanced Topics & Extensions

### 7.1 Modern Variants

**Extension 1: Principle-Based Reserves (PBR)**
- Stochastic scenarios
- Risk-based approach

**Extension 2: IFRS 17**
- Market-consistent valuation
- Risk adjustment

**Extension 3: Variable Annuity Reserves**
- Guaranteed minimum benefits
- Stochastic modeling

### 7.2 Integration with Other Methods

**Combination 1: Reserves + Asset-Liability Management**
- Match reserve duration with asset duration
- Immunization strategies

**Combination 2: Reserves + Capital**
- RBC = f(Reserves)
- Economic capital modeling

### 7.3 Cutting-Edge Research

**Topic 1: Machine Learning for Reserve Estimation**
- Predict lapses, mortality
- Dynamic reserve adjustment

**Topic 2: Longevity Risk**
- Stochastic mortality for reserves
- Longevity hedging

---

## 8. Regulatory & Governance Considerations

### 8.1 Regulatory Perspective

**Acceptability:**
- **Widely Accepted:** Yes, reserves are required
- **Jurisdictions:** All require statutory reserves
- **Documentation Required:** Actuarial opinion on reserve adequacy

**Key Regulatory Concerns:**
1. **Concern: Reserve Adequacy**
   - **Issue:** Are reserves sufficient?
   - **Mitigation:** Use prescribed methods (CRVM, etc.)

2. **Concern: Deficiency Reserves**
   - **Issue:** Gross < net premium
   - **Mitigation:** Calculate and hold deficiency reserves

### 8.2 Model Governance

**Model Risk Rating:** High
- **Justification:** Reserves directly affect solvency

**Validation Frequency:** Annual (or more frequent)

**Key Validation Tests:**
1. **Prospective = Retrospective:** Verify consistency
2. **Benchmarking:** Compare to published tables
3. **Sensitivity:** Test assumption changes

### 8.3 Documentation Requirements

**Minimum Documentation:**
- ✓ Reserve method used (net premium, CRVM, etc.)
- ✓ Assumptions (mortality, interest)
- ✓ Verification (prospective = retrospective)
- ✓ Sensitivity analysis
- ✓ Actuarial opinion

---

## 9. Practical Example

### 9.1 Worked Example: Calculating Reserves

**Scenario:** Calculate reserves for a $100,000 whole life policy issued at age 60. Use 4% interest and:
- $A_{60} = 0.35$
- $\ddot{a}_{60} = 17.58$
- $A_{70} = 0.55$
- $\ddot{a}_{70} = 12.5$
- $A_{80} = 0.75$
- $\ddot{a}_{80} = 7.8$

**Step 1: Calculate Net Premium**
$$
P_{60} = \frac{A_{60}}{\ddot{a}_{60}} = \frac{0.35}{17.58} = 0.0199
$$

**Step 2: Calculate Reserve at Time 10 (Age 70)**
$$
_{10}V_{60} = A_{70} - P_{60} \times \ddot{a}_{70} = 0.55 - 0.0199 \times 12.5 = 0.55 - 0.249 = 0.301
$$

For $100,000: Reserve = $100,000 \times 0.301 = $30,100.

**Step 3: Calculate Reserve at Time 20 (Age 80)**
$$
_{20}V_{60} = A_{80} - P_{60} \times \ddot{a}_{80} = 0.75 - 0.0199 \times 7.8 = 0.75 - 0.155 = 0.595
$$

For $100,000: Reserve = $100,000 \times 0.595 = $59,500.

**Step 4: Verify Using Recursive Formula**
From time 10 to time 11:
- $_10V_{60} = 0.301$
- $P_{60} = 0.0199$
- $i = 4\%$
- $q_{70} = 0.015$

$$
_{11}V_{60} = \frac{(0.301 + 0.0199)(1.04) - 0.015}{1 - 0.015} = \frac{0.3337 - 0.015}{0.985} = 0.3236
$$

**Interpretation:** Reserve increases from $30,100 at year 10 to $32,360 at year 11 to $59,500 at year 20.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1. **Reserve = EPV(future benefits) - EPV(future premiums)**
2. **Prospective:** Forward-looking calculation
3. **Retrospective:** Backward-looking calculation (same result)
4. **Recursive:** Fackler's equation for year-to-year calculation
5. **Modified reserves:** FPT, CRVM reduce first-year strain

### 10.2 When to Use This Knowledge
**Ideal For:**
- ✓ SOA Exam LTAM preparation
- ✓ Statutory reserve calculations
- ✓ Financial reporting
- ✓ Product design
- ✓ Solvency assessment

**Not Ideal For:**
- ✗ Pricing (use net premiums)
- ✗ Stochastic modeling (use PBR)

### 10.3 Critical Success Factors
1. **Master Prospective Formula:** $_tV = A_{x+t} - P \ddot{a}_{x+t}$
2. **Verify Consistency:** Prospective = retrospective
3. **Understand Modified Reserves:** FPT, CRVM
4. **Project Reserves:** Track over policy lifetime
5. **Apply to Regulation:** Know statutory requirements

### 10.4 Further Reading
- **Textbook:** "Actuarial Mathematics for Life Contingent Risks" (Dickson, Hardy, Waters) - Chapter 8
- **Exam Prep:** Coaching Actuaries LTAM
- **SOA:** Valuation Manual (VM-20)
- **Regulation:** NAIC Model Regulation

---

## Appendix

### A. Glossary
- **Reserve:** Amount insurer must hold for future obligations
- **Prospective:** Forward-looking reserve calculation
- **Retrospective:** Backward-looking reserve calculation
- **Terminal Reserve:** Reserve at end of year (after premium)
- **Initial Reserve:** Reserve at start of year (before premium)
- **FPT:** Full Preliminary Term (modified reserve method)
- **CRVM:** Commissioners Reserve Valuation Method

### B. Key Formulas Summary

| Formula | Equation | Use |
|---------|----------|-----|
| **Prospective** | $_tV = A_{x+t} - P \ddot{a}_{x+t}$ | Forward-looking |
| **Recursive** | $_{t+1}V = ((V_t + P)(1+i) - q_{x+t}) / p_{x+t}$ | Year-to-year |
| **Terminal** | $_t^*V = _tV + P$ | End of year |
| **FPT** | $_1V^{FPT} = 0$ | Modified reserve |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,100+*
