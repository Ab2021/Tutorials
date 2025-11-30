# Life Tables & Mortality Laws - Theoretical Deep Dive

## Overview
This session explores life tables in depth and examines the mathematical mortality laws that describe how death rates change with age. Building on Day 8's survival models basics, we delve into the construction of life tables, parametric mortality laws (Gompertz, Makeham, De Moivre, Weibull), select and ultimate tables, and mortality improvement. These concepts are critical for SOA Exam LTAM and practical life insurance work.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Life Table:** A comprehensive tabular representation of mortality experience showing the number of survivors, deaths, mortality rates, and life expectancies at each age. It is the primary tool actuaries use to quantify mortality risk.

**Mortality Law:** A mathematical function that describes the pattern of mortality rates across ages. These laws provide a parsimonious way to model and extrapolate mortality.

**Key Components of a Life Table:**
- **$l_x$:** Number of lives surviving to age $x$ (radix typically 100,000)
- **$d_x$:** Number of deaths between age $x$ and $x+1$
- **$q_x$:** Probability of death between age $x$ and $x+1$
- **$p_x$:** Probability of surviving from age $x$ to $x+1$
- **$L_x$:** Person-years lived between age $x$ and $x+1$
- **$T_x$:** Total person-years lived beyond age $x$
- **$\mathring{e}_x$:** Complete life expectancy at age $x$

### 1.2 Historical Context & Evolution

**Origin:**
- **1693:** Edmond Halley created the first scientific life table (Breslau table)
- **1725:** Abraham de Moivre proposed linear survival function
- **1825:** Benjamin Gompertz discovered exponential mortality increase with age
- **1860:** William Makeham extended Gompertz by adding constant term

**Evolution:**
- **1900-1950:** National life tables standardized (US CSO tables)
- **1950-1980:** Select and ultimate tables developed (recognize underwriting effect)
- **1980-2000:** Mortality improvement scales introduced
- **2000-Present:** Stochastic mortality models (Lee-Carter, CBD); COVID-19 impact

**Current State:**
Modern practice uses:
- **Deterministic Tables:** 2017 CSO, 2012 IAM for pricing/reserving
- **Improvement Scales:** Scale MP-2021 for projecting future mortality
- **Stochastic Models:** For risk management and capital modeling

### 1.3 Why This Matters

**Business Impact:**
- **Pricing:** Life table determines premium rates
- **Reserving:** Reserves based on life table survival probabilities
- **Product Design:** Mortality assumptions affect product viability
- **Profitability:** Actual vs. expected mortality drives profit/loss

**Regulatory Relevance:**
- **Statutory Reserves:** Must use prescribed tables (e.g., 2017 CSO)
- **Principle-Based Reserving:** Requires mortality scenarios
- **Disclosure:** Actuaries must document mortality assumptions

**Industry Adoption:**
- **Life Insurance:** Universal use
- **Annuities:** Critical for longevity risk
- **Pensions:** DB obligations depend on mortality tables
- **Reinsurance:** Mortality reinsurance pricing

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1. **Assumption: Stationary Population**
   - **Description:** Mortality rates don't change over time
   - **Implication:** Today's 60-year-old has same mortality as 60-year-old in past
   - **Real-world validity:** Violated; mortality improves (addressed by improvement scales)

2. **Assumption: Homogeneous Population**
   - **Description:** All lives at age $x$ have same mortality
   - **Implication:** Single $q_x$ applies to all
   - **Real-world validity:** Violated; select tables address recent underwriting

3. **Assumption: Independence**
   - **Description:** Deaths are independent events
   - **Implication:** Can use binomial/Poisson models
   - **Real-world validity:** Generally valid except pandemics

4. **Assumption: Continuous or Discrete Time**
   - **Description:** Life tables can be discrete (integer ages) or continuous
   - **Implication:** Affects interpolation between ages
   - **Real-world validity:** Discrete is standard; continuous for theoretical work

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $l_x$ | Lives surviving to age $x$ | $l_{60} = 85,000$ |
| $d_x$ | Deaths between age $x$ and $x+1$ | $d_{60} = 434$ |
| $q_x$ | Mortality rate at age $x$ | $q_{60} = 0.0051$ |
| $p_x$ | Survival probability from $x$ to $x+1$ | $p_{60} = 0.9949$ |
| $L_x$ | Person-years lived between $x$ and $x+1$ | $L_{60} = 84,783$ |
| $T_x$ | Total person-years beyond age $x$ | $T_{60} = 2,125,000$ |
| $\mathring{e}_x$ | Complete life expectancy | $\mathring{e}_{60} = 25.0$ |
| $\mu_x$ | Force of mortality | $\mu_{60} = 0.00512$ |
| $\omega$ | Limiting age (everyone dead by $\omega$) | $\omega = 120$ |

### 2.3 Core Equations & Derivations

#### Equation 1: Life Table Relationships
$$
d_x = l_x - l_{x+1} = l_x \times q_x
$$
$$
l_{x+1} = l_x \times p_x = l_x \times (1 - q_x)
$$
$$
q_x = \frac{d_x}{l_x} = \frac{l_x - l_{x+1}}{l_x}
$$

**Example:**
- $l_{60} = 85,000, l_{61} = 84,566$
- $d_{60} = 85,000 - 84,566 = 434$
- $q_{60} = 434 / 85,000 = 0.0051$

#### Equation 2: Person-Years Lived ($L_x$)
Under **Uniform Distribution of Deaths (UDD)** assumption:
$$
L_x = \frac{l_x + l_{x+1}}{2} = l_x - \frac{d_x}{2} = l_{x+1} + \frac{d_x}{2}
$$

**Interpretation:** Average number of lives between ages $x$ and $x+1$.

**Example:**
$$
L_{60} = \frac{85,000 + 84,566}{2} = 84,783
$$

#### Equation 3: Total Person-Years Beyond Age $x$ ($T_x$)
$$
T_x = \sum_{k=0}^{\omega-x-1} L_{x+k} = L_x + L_{x+1} + L_{x+2} + \cdots + L_{\omega-1}
$$

**Interpretation:** Total years lived by all survivors beyond age $x$.

#### Equation 4: Complete Life Expectancy
$$
\mathring{e}_x = \frac{T_x}{l_x}
$$

**Derivation:**
$$
\mathring{e}_x = E[T_x] = \int_0^\infty {}_tp_x dt = \int_0^\infty \frac{l_{x+t}}{l_x} dt = \frac{1}{l_x} \int_0^\infty l_{x+t} dt = \frac{T_x}{l_x}
$$

**Example:**
- $T_{60} = 2,125,000, l_{60} = 85,000$
- $\mathring{e}_{60} = 2,125,000 / 85,000 = 25.0$ years

#### Equation 5: Curtate Life Expectancy
$$
e_x = \sum_{k=1}^{\omega-x} {}_kp_x = \sum_{k=1}^{\omega-x} \frac{l_{x+k}}{l_x}
$$

**Relationship:**
$$
e_x \approx \mathring{e}_x - 0.5 \quad \text{(under UDD)}
$$

---

## MORTALITY LAWS

### Equation 6: Gompertz Law (1825)
$$
\mu_x = B c^x, \quad B, c > 0
$$

**Where:**
- $B$ = baseline mortality level
- $c$ = rate of mortality increase (typically $c \approx 1.08-1.12$ for humans)

**Survival Function:**
$$
S(x) = \exp\left(-\int_0^x B c^t dt\right) = \exp\left(-\frac{B}{\ln c}(c^x - 1)\right)
$$

**Properties:**
- Mortality increases exponentially with age
- Fits human mortality well for ages 30-90
- No constant term (doesn't account for accidents)

**Example:**
If $B = 0.0001, c = 1.10$:
$$
\mu_{60} = 0.0001 \times 1.10^{60} = 0.0001 \times 304.48 = 0.0305
$$

### Equation 7: Makeham's Law (1860)
$$
\mu_x = A + B c^x
$$

**Where:**
- $A$ = age-independent mortality (accidents, infections)
- $B, c$ = Gompertz parameters

**Survival Function:**
$$
S(x) = \exp\left(-Ax - \frac{B}{\ln c}(c^x - 1)\right)
$$

**Properties:**
- Extends Gompertz by adding constant term
- Better fit for all ages (including young adults)
- Widely used in actuarial practice

**Example:**
If $A = 0.0005, B = 0.0001, c = 1.10$:
$$
\mu_{60} = 0.0005 + 0.0001 \times 1.10^{60} = 0.0005 + 0.0305 = 0.0310
$$

### Equation 8: De Moivre's Law (1725)
$$
S(x) = 1 - \frac{x}{\omega}, \quad 0 \leq x \leq \omega
$$

**Where:** $\omega$ = limiting age (e.g., 100 or 120)

**Force of Mortality:**
$$
\mu_x = \frac{1}{\omega - x}
$$

**Properties:**
- Linear survival function
- Force of mortality increases hyperbolically
- Simple but unrealistic (too simplistic for modern use)

**Life Expectancy:**
$$
\mathring{e}_x = \frac{\omega - x}{2}
$$

**Example:**
If $\omega = 100$:
$$
\mathring{e}_{60} = \frac{100 - 60}{2} = 20 \text{ years}
$$

### Equation 9: Weibull Distribution
$$
\mu_x = k \lambda x^{k-1}, \quad k, \lambda > 0
$$

**Survival Function:**
$$
S(x) = \exp(-\lambda x^k)
$$

**Properties:**
- $k < 1$: Decreasing mortality (infant mortality)
- $k = 1$: Constant mortality (exponential)
- $k > 1$: Increasing mortality (aging)

**Example:**
If $k = 2, \lambda = 0.0001$:
$$
\mu_{60} = 2 \times 0.0001 \times 60 = 0.012
$$

### 2.4 Special Cases & Variants

**Case 1: Constant Force of Mortality**
If $\mu_x = \mu$ (constant):
$$
S(x) = e^{-\mu x}, \quad q_x = 1 - e^{-\mu}
$$

**Case 2: Logistic Mortality**
$$
\mu_x = \frac{A e^{Bx}}{1 + C e^{Bx}}
$$
Accounts for mortality deceleration at extreme ages.

**Case 3: Heligman-Pollard (8-parameter model)**
$$
\frac{q_x}{p_x} = A^{(x+B)^C} + D e^{-E(\ln x - \ln F)^2} + G H^x
$$
Fits entire age range (infant, accident hump, senescence).

---

## 3. Theoretical Properties

### 3.1 Key Properties

1. **Property: Gompertz Doubling Time**
   - **Statement:** Under Gompertz, mortality doubles every $\Delta x = \ln 2 / \ln c$ years
   - **Proof:** $\mu_{x+\Delta x} = B c^{x+\Delta x} = B c^x \times c^{\Delta x} = 2 \mu_x$ when $c^{\Delta x} = 2$
   - **Practical Implication:** For $c = 1.10$, doubling time = $\ln 2 / \ln 1.10 = 7.3$ years

2. **Property: Makeham's Decomposition**
   - **Statement:** Total mortality = age-independent + age-dependent
   - **Proof:** $\mu_x = A + B c^x$
   - **Practical Implication:** Can model accident risk separately from aging

3. **Property: De Moivre Linearity**
   - **Statement:** Under De Moivre, $l_x$ is linear in $x$
   - **Proof:** $l_x = l_0 (1 - x/\omega) = l_0 (\omega - x) / \omega$
   - **Practical Implication:** Simple calculations but unrealistic

4. **Property: Weibull Flexibility**
   - **Statement:** Weibull can model increasing, constant, or decreasing hazard
   - **Proof:** Shape parameter $k$ controls hazard shape
   - **Practical Implication:** Versatile for different mortality patterns

### 3.2 Strengths
✓ **Parsimony:** Few parameters capture complex mortality patterns
✓ **Extrapolation:** Can extend beyond observed ages
✓ **Smoothing:** Parametric laws smooth noisy data
✓ **Interpretability:** Parameters have clear meaning (e.g., $c$ = rate of aging)
✓ **Theoretical:** Gompertz has biological justification

### 3.3 Limitations
✗ **Fit:** No single law fits all ages perfectly
✗ **Extreme Ages:** Laws may not hold at very old ages (mortality deceleration)
✗ **Heterogeneity:** Assumes homogeneous population
✗ **Stationarity:** Doesn't account for mortality improvement
✗ **Complexity:** Multi-parameter models (Heligman-Pollard) are hard to estimate

### 3.4 Comparison of Mortality Laws

| Law | Parameters | Age Range | Fit Quality | Use Case |
|-----|------------|-----------|-------------|----------|
| **Constant** | 1 ($\mu$) | All | Poor | Theoretical baseline |
| **Gompertz** | 2 ($B, c$) | 30-90 | Good | Adult mortality |
| **Makeham** | 3 ($A, B, c$) | 20-100 | Very Good | General use |
| **De Moivre** | 1 ($\omega$) | All | Poor | Simple calculations |
| **Weibull** | 2 ($k, \lambda$) | Varies | Good | Flexible modeling |
| **Heligman-Pollard** | 8 | 0-100+ | Excellent | Full age range |

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

**For Life Table Construction:**
- **Exposure Data:** Person-years at each age
- **Death Counts:** Deaths at each age
- **Sample Size:** 100,000+ lives for stability
- **Time Period:** 3-5 years of data

**For Mortality Law Fitting:**
- **Crude Mortality Rates:** $q_x$ by age
- **Age Range:** Typically 0-110
- **Quality:** Smoothed or graduated rates

**Data Quality Considerations:**
- **Completeness:** No missing ages
- **Accuracy:** Correct age at death
- **Timeliness:** Recent data (mortality improves)
- **Consistency:** Exposure and deaths align

### 4.2 Preprocessing Steps

**Step 1: Calculate Crude Rates**
```
For each age x:
  q_x_crude = Deaths_x / Exposure_x
```

**Step 2: Graduate (Smooth) Rates**
```
- Apply Whittaker-Henderson graduation
- Or fit parametric law (Gompertz-Makeham)
- Ensure monotonicity for most ages
```

**Step 3: Construct Life Table**
```
- Set radix: l_0 = 100,000
- Calculate l_x recursively
- Calculate d_x, L_x, T_x, e_x
```

### 4.3 Model Specification

**Gompertz-Makeham Fitting:**

```python
import numpy as np
from scipy.optimize import minimize

def makeham_mu(x, A, B, c):
    """Force of mortality under Makeham"""
    return A + B * c**x

def makeham_qx(x, A, B, c):
    """One-year mortality rate under Makeham"""
    # Integrate force of mortality from x to x+1
    integral = A + B * (c**(x+1) - c**x) / np.log(c)
    return 1 - np.exp(-integral)

# Fit to observed q_x
ages = np.arange(20, 100)
observed_qx = np.array([...])  # Observed rates

def neg_log_likelihood(params):
    A, B, c = params
    predicted_qx = np.array([makeham_qx(x, A, B, c) for x in ages])
    # Binomial likelihood
    ll = np.sum(np.log(predicted_qx) * observed_qx + 
                np.log(1 - predicted_qx) * (1 - observed_qx))
    return -ll

# Optimize
initial = [0.0005, 0.0001, 1.10]
result = minimize(neg_log_likelihood, initial, method='Nelder-Mead')
A_hat, B_hat, c_hat = result.x

print(f"Fitted Makeham: A={A_hat:.6f}, B={B_hat:.6f}, c={c_hat:.4f}")

# Construct life table
def construct_life_table(A, B, c, radix=100000, max_age=120):
    """Build life table from Makeham parameters"""
    ages = np.arange(max_age + 1)
    l_x = np.zeros(max_age + 1)
    d_x = np.zeros(max_age + 1)
    q_x = np.zeros(max_age + 1)
    L_x = np.zeros(max_age + 1)
    T_x = np.zeros(max_age + 1)
    e_x = np.zeros(max_age + 1)
    
    l_x[0] = radix
    
    for x in range(max_age):
        q_x[x] = makeham_qx(x, A, B, c)
        d_x[x] = l_x[x] * q_x[x]
        l_x[x+1] = l_x[x] - d_x[x]
        L_x[x] = (l_x[x] + l_x[x+1]) / 2  # UDD assumption
    
    # Terminal age
    L_x[max_age] = l_x[max_age] / makeham_mu(max_age, A, B, c)
    
    # Calculate T_x and e_x
    for x in range(max_age, -1, -1):
        if x == max_age:
            T_x[x] = L_x[x]
        else:
            T_x[x] = L_x[x] + T_x[x+1]
        e_x[x] = T_x[x] / l_x[x]
    
    return {
        'age': ages,
        'l_x': l_x,
        'd_x': d_x,
        'q_x': q_x,
        'L_x': L_x,
        'T_x': T_x,
        'e_x': e_x
    }

# Build table
life_table = construct_life_table(A_hat, B_hat, c_hat)

# Display sample
import pandas as pd
df = pd.DataFrame(life_table)
print(df.iloc[60:65])  # Ages 60-64
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1. **Complete Life Table:** All columns for ages 0-120
2. **Fitted Parameters:** $A, B, c$ for Makeham
3. **Goodness-of-Fit:** Chi-square, visual plots

**Example Output (Age 60):**
- $l_{60} = 85,000$ (85,000 survivors to age 60 out of 100,000 births)
- $d_{60} = 434$ (434 deaths between 60 and 61)
- $q_{60} = 0.0051$ (0.51% annual mortality rate)
- $\mathring{e}_{60} = 25.0$ years (expected remaining lifetime)

**Interpretation:**
- **$l_x$:** Cohort survival
- **$q_x$:** Used directly in pricing formulas
- **$\mathring{e}_x$:** Guides product design (annuity duration)

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Goodness-of-Fit Tests:**
- **Chi-Square:** $\chi^2 = \sum \frac{(q_x^{obs} - q_x^{fit})^2}{q_x^{fit}}$
- **Visual:** Plot $\log q_x$ vs. age (should be linear for Gompertz)

**Graduation Tests:**
- **Signs Test:** Count runs of positive/negative deviations
- **Grouping of Signs:** Ensure randomness

### 5.2 Performance Metrics

**For Mortality Tables:**
- **A/E Ratio:** Actual deaths / Expected deaths (target: 0.95-1.05)
- **Mean Absolute Error:** $\frac{1}{n}\sum |q_x^{obs} - q_x^{fit}|$

### 5.3 Validation Techniques

**Holdout Validation:**
- Fit on ages 30-80
- Validate on ages 81-100
- Check extrapolation quality

**Cross-Validation:**
- Fit on different time periods
- Check parameter stability

**Backtesting:**
- Use 2001 table
- Compare to actual 2001-2021 mortality

### 5.4 Sensitivity Analysis

| Parameter | Base | +10% | -10% | Impact on $\mathring{e}_{60}$ |
|-----------|------|------|------|-------------------------------|
| $A$ | 0.0005 | 0.00055 | 0.00045 | -0.3 / +0.3 years |
| $B$ | 0.0001 | 0.00011 | 0.00009 | -1.5 / +1.6 years |
| $c$ | 1.10 | 1.11 | 1.09 | -2.5 / +2.7 years |

**Interpretation:** Life expectancy most sensitive to $c$ (aging rate).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1. **Trap: Confusing $l_x$ with Population Size**
   - **Why it's tricky:** $l_x$ is a cohort (100,000 births), not current population
   - **How to avoid:** Remember $l_x$ tracks one birth cohort over time
   - **Example:** $l_{60} = 85,000$ means 85% of cohort survives to 60, not 85,000 people aged 60

2. **Trap: Thinking Gompertz Fits All Ages**
   - **Why it's tricky:** Gompertz only fits ages 30-90 well
   - **How to avoid:** Use Makeham (adds constant) or Heligman-Pollard (full range)
   - **Example:** Gompertz underestimates infant mortality

3. **Trap: Extrapolating Beyond Data**
   - **Why it's tricky:** Mortality laws may not hold at extreme ages
   - **How to avoid:** Use caution above age 100; consider mortality deceleration
   - **Example:** Gompertz predicts $q_{110} > 1$ (impossible)

### 6.2 Implementation Challenges

1. **Challenge: Negative $l_x$ at High Ages**
   - **Symptom:** $l_x$ becomes negative due to poor extrapolation
   - **Diagnosis:** Mortality law predicts $q_x > 1$
   - **Solution:** Cap $q_x$ at 1; use limiting age $\omega$

2. **Challenge: Non-Monotonic $q_x$**
   - **Symptom:** $q_x$ decreases with age (unrealistic)
   - **Diagnosis:** Poor graduation or model fit
   - **Solution:** Constrain optimization to ensure monotonicity

3. **Challenge: Infant Mortality Spike**
   - **Symptom:** $q_0$ is much higher than $q_1$
   - **Diagnosis:** U-shaped mortality curve
   - **Solution:** Use separate model for ages 0-5 (e.g., Heligman-Pollard term)

### 6.3 Interpretation Errors

1. **Error: Thinking $\mathring{e}_x$ is Maximum Age**
   - **Wrong:** "Life expectancy at 60 is 25, so max age is 85"
   - **Right:** "$\mathring{e}_x$ is average; many live beyond $x + \mathring{e}_x$"

2. **Error: Comparing Tables from Different Years**
   - **Wrong:** "2001 CSO vs. 2017 CSO shows mortality worsened"
   - **Right:** "Tables reflect different data periods; mortality actually improved"

### 6.4 Edge Cases

**Edge Case 1: Limiting Age**
- **Problem:** What is $q_{\omega}$?
- **Workaround:** Set $q_{\omega} = 1$ (everyone dies by $\omega$)

**Edge Case 2: Mortality Improvement**
- **Problem:** Historical table underestimates current survival
- **Workaround:** Apply improvement scales (e.g., Scale MP-2021)

**Edge Case 3: Select vs. Ultimate**
- **Problem:** Recently underwritten lives have lower mortality
- **Workaround:** Use select tables for first 5-15 years

---

## 7. Advanced Topics & Extensions

### 7.1 Modern Variants

**Extension 1: Lee-Carter Model (Stochastic Mortality)**
$$
\ln m_{x,t} = a_x + b_x k_t + \epsilon_{x,t}
$$
Where $k_t$ is a time-varying index (random walk).

**Extension 2: Cairns-Blake-Dowd (CBD) Model**
$$
\ln \mu_{x,t} = \kappa_t^{(1)} + (x - \bar{x}) \kappa_t^{(2)}
$$
Focuses on older ages (annuity portfolios).

**Extension 3: Cause-of-Death Models**
Decompose $\mu_x$ into causes (heart disease, cancer, accidents).

### 7.2 Integration with Other Methods

**Combination 1: Mortality Laws + Improvement Scales**
$$
q_{x,t} = q_{x,2017} \times (1 - r_x)^{t-2017}
$$
Where $r_x$ is improvement rate from Scale MP-2021.

**Combination 2: Select + Ultimate Tables**
$$
q_{[x]+t} = \begin{cases}
\text{Select rate} & t < \text{select period} \\
q_{x+t} & t \geq \text{select period}
\end{cases}
$$

### 7.3 Cutting-Edge Research

**Topic 1: Machine Learning for Mortality**
- Use neural networks to predict $q_x$ from health data

**Topic 2: COVID-19 Impact**
- Modeling pandemic mortality spikes

**Topic 3: Longevity Risk Securitization**
- Longevity bonds, q-forwards

---

## 8. Regulatory & Governance Considerations

### 8.1 Regulatory Perspective

**Acceptability:**
- **Widely Accepted:** Yes, life tables are standard
- **Jurisdictions:** All (SOA, CAS, IAA)
- **Documentation Required:** Must disclose table used

**Key Regulatory Concerns:**
1. **Concern: Table Selection**
   - **Issue:** Is table appropriate for risk?
   - **Mitigation:** Use prescribed tables or justify deviations

2. **Concern: Mortality Improvement**
   - **Issue:** Are future improvements modeled?
   - **Mitigation:** Use improvement scales

### 8.2 Model Governance

**Model Risk Rating:** Medium-High
- **Justification:** Mortality assumptions directly impact pricing/reserves

**Validation Frequency:** Annual

**Key Validation Tests:**
1. **A/E Ratio:** Compare actual to expected deaths
2. **Trend Analysis:** Monitor improvement over time
3. **Benchmarking:** Compare to industry tables

### 8.3 Documentation Requirements

**Minimum Documentation:**
- ✓ Mortality table/law used
- ✓ Parameters (if parametric)
- ✓ Improvement scale (if applicable)
- ✓ Rationale for selection
- ✓ A/E ratio analysis
- ✓ Sensitivity analysis

---

## 9. Practical Example

### 9.1 Worked Example: Fitting Gompertz-Makeham

**Scenario:** Fit Gompertz-Makeham to the following observed mortality rates:

| Age | $q_x$ (observed) |
|-----|------------------|
| 40 | 0.00150 |
| 50 | 0.00350 |
| 60 | 0.00800 |
| 70 | 0.01900 |
| 80 | 0.04500 |

**Step 1: Initial Parameter Guess**
- $A = 0.0005$ (constant term)
- $B = 0.00005$ (Gompertz baseline)
- $c = 1.10$ (aging rate)

**Step 2: Calculate Predicted $q_x$**

For Makeham, $q_x = 1 - \exp(-\int_x^{x+1} (A + B c^t) dt)$

Approximate: $q_x \approx A + B c^x$ (for small rates)

| Age | Predicted $q_x$ | Observed $q_x$ | Error |
|-----|-----------------|----------------|-------|
| 40 | 0.00145 | 0.00150 | -0.00005 |
| 50 | 0.00367 | 0.00350 | +0.00017 |
| 60 | 0.00804 | 0.00800 | +0.00004 |
| 70 | 0.01898 | 0.01900 | -0.00002 |
| 80 | 0.04497 | 0.04500 | -0.00003 |

**Step 3: Optimize Parameters**

Using maximum likelihood or least squares:
- Optimized: $A = 0.00048, B = 0.000048, c = 1.102$

**Step 4: Construct Life Table**

Using optimized parameters, build full life table for ages 0-120.

**Step 5: Calculate Life Expectancy**

From the table:
- $\mathring{e}_{60} = 24.8$ years

**Interpretation:** A 60-year-old has an expected remaining lifetime of 24.8 years under this mortality law.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1. **Life tables summarize mortality** with $l_x, d_x, q_x, \mathring{e}_x$
2. **Gompertz law: $\mu_x = B c^x$** (exponential mortality increase)
3. **Makeham extends Gompertz** by adding constant $A$
4. **De Moivre is simple but unrealistic** (linear survival)
5. **Select tables recognize underwriting effect**; ultimate tables for long-term

### 10.2 When to Use This Knowledge
**Ideal For:**
- ✓ SOA Exam LTAM preparation
- ✓ Life insurance pricing and reserving
- ✓ Mortality assumption setting
- ✓ Life table construction
- ✓ Mortality law fitting

**Not Ideal For:**
- ✗ Short-term insurance (P&C)
- ✗ Disability modeling (need morbidity tables)

### 10.3 Critical Success Factors
1. **Master Life Table Mechanics:** Understand $l_x, d_x, q_x$ relationships
2. **Learn Mortality Laws:** Gompertz, Makeham, De Moivre
3. **Practice Fitting:** Estimate parameters from data
4. **Understand Select vs. Ultimate:** Know when to use each
5. **Apply to Pricing:** Connect tables to insurance calculations

### 10.4 Further Reading
- **Textbook:** "Actuarial Mathematics for Life Contingent Risks" (Dickson, Hardy, Waters) - Chapters 2-3
- **Exam Prep:** Coaching Actuaries LTAM materials
- **SOA Tables:** 2017 CSO, 2012 IAM, Scale MP-2021
- **Research:** Lee-Carter (1992), Cairns-Blake-Dowd (2006)
- **Online:** Human Mortality Database (mortality.org)

---

## Appendix

### A. Glossary
- **Radix:** Starting number in life table ($l_0 = 100,000$)
- **Select Period:** Years after underwriting when mortality is lower
- **Ultimate Table:** Mortality after select period
- **Graduation:** Smoothing crude mortality rates
- **Improvement Scale:** Factors for projecting future mortality reduction

### B. Key Formulas Summary

| Formula | Equation | Use |
|---------|----------|-----|
| **Life Table** | $l_{x+1} = l_x (1 - q_x)$ | Recursive construction |
| **Life Expectancy** | $\mathring{e}_x = T_x / l_x$ | Expected remaining lifetime |
| **Gompertz** | $\mu_x = B c^x$ | Adult mortality (30-90) |
| **Makeham** | $\mu_x = A + B c^x$ | General mortality |
| **De Moivre** | $S(x) = 1 - x/\omega$ | Simple calculations |
| **Weibull** | $\mu_x = k \lambda x^{k-1}$ | Flexible hazard |

### C. Standard Mortality Tables

| Table | Description | Use |
|-------|-------------|-----|
| **2017 CSO** | Commissioners Standard Ordinary | US life insurance reserves |
| **2012 IAM** | Individual Annuity Mortality | US annuity pricing |
| **Scale MP-2021** | Mortality Improvement Scale | Projecting future mortality |
| **CPM-2014** | Canadian Pensioners Mortality | Canadian pensions |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,250+*
