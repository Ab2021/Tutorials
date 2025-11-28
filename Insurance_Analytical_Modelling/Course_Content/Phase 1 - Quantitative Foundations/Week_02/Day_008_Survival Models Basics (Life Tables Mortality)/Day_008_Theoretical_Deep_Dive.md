# Survival Models Basics - Theoretical Deep Dive

## Overview
This session introduces survival models, which form the mathematical foundation for life insurance, annuities, and pension valuation. Survival models quantify the probability of living or dying over time, enabling actuaries to price products, set reserves, and manage longevity risk. These concepts are essential for SOA Exam LTAM (Long-Term Actuarial Mathematics) and all life insurance work.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Survival Model:** A mathematical framework that describes the distribution of time until death (or failure) for a life (or system). It quantifies the probability that a person aged $x$ will survive to age $x+t$ or die before age $x+t$.

**Key Random Variable:**
- **$T_x$:** Future lifetime random variable for a life aged $x$ (time until death from age $x$)
- **$K_x$:** Curtate future lifetime (integer years until death)

**Key Terminology:**
- **Survival Function $S(x)$:** Probability of surviving from birth to age $x$
- **Mortality Rate $q_x$:** Probability that a life aged $x$ dies before age $x+1$
- **Survival Probability ${}_tp_x$:** Probability that a life aged $x$ survives $t$ years
- **Force of Mortality $\mu_x$:** Instantaneous death rate at age $x$ (hazard function)
- **Life Table:** Tabular representation of survival probabilities and deaths
- **Life Expectancy $\mathring{e}_x$:** Expected remaining lifetime at age $x$

### 1.2 Historical Context & Evolution

**Origin:**
- **1693:** Edmond Halley created the first scientific life table using Breslau mortality data
- **1700s:** Life tables used for pricing annuities and tontines
- **1800s:** Actuarial profession formalized; standardized mortality tables developed

**Evolution:**
- **1900-1950:** National life tables (e.g., US CSO - Commissioners Standard Ordinary)
- **1950-1980:** Select and ultimate tables (recognize recent underwriting)
- **1980-2000:** Mortality improvement incorporated (people living longer)
- **2000-Present:** Stochastic mortality models (Lee-Carter, CBD); pandemic impact (COVID-19)

**Current State:**
Modern actuarial practice uses:
- **Deterministic Tables:** Standard tables (e.g., 2017 CSO) for pricing and reserving
- **Mortality Improvement:** Projection scales (e.g., Scale MP-2021) for future mortality reduction
- **Stochastic Models:** For risk management and capital modeling (longevity risk)

### 1.3 Why This Matters

**Business Impact:**
- **Pricing:** Life insurance premiums depend critically on mortality assumptions
- **Reserving:** Reserves are present value of future death benefits (mortality-dependent)
- **Profitability:** Mortality experience vs. pricing assumptions drives profit/loss
- **Risk Management:** Longevity risk (people living longer than expected) affects annuities and pensions

**Regulatory Relevance:**
- **Statutory Reserves:** Use prescribed mortality tables (e.g., 2017 CSO)
- **Principle-Based Reserving (PBR):** Requires stochastic mortality scenarios
- **Solvency II:** Longevity risk is a component of SCR

**Industry Adoption:**
- **Life Insurance:** Core product pricing and reserving
- **Annuities:** Critical for longevity risk assessment
- **Pensions:** Defined benefit obligations depend on survival probabilities
- **Reinsurance:** Mortality and longevity reinsurance markets

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1. **Assumption: Survival Function is Continuous and Decreasing**
   - **Description:** $S(x)$ is a continuous, non-increasing function with $S(0) = 1$ and $\lim_{x \to \infty} S(x) = 0$
   - **Implication:** Everyone eventually dies; probability of survival decreases with age
   - **Real-world validity:** Generally valid; $S(x)$ may have discontinuities in practice (infant mortality spike)

2. **Assumption: Independent Lives**
   - **Description:** Survival of one life is independent of another (unless joint life)
   - **Implication:** Can multiply survival probabilities for independent lives
   - **Real-world validity:** Violated for spouses (common lifestyle, shared environment)

3. **Assumption: Stationary Mortality**
   - **Description:** Mortality rates don't change over time (no mortality improvement)
   - **Implication:** Today's 60-year-old has same mortality as 60-year-old 10 years ago
   - **Real-world validity:** Violated in practice; mortality improves over time (medical advances)

4. **Assumption: Homogeneous Population**
   - **Description:** All lives aged $x$ have the same mortality rate $q_x$
   - **Implication:** Can use a single life table for all lives
   - **Real-world validity:** Violated; mortality varies by gender, health, socioeconomic status (select tables address this)

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $T_x$ | Future lifetime random variable (continuous) | Time until death from age $x$ |
| $K_x$ | Curtate future lifetime (integer years) | $K_x = \lfloor T_x \rfloor$ |
| $S(x)$ | Survival function from birth | $P(\text{survive to age } x)$ |
| ${}_tp_x$ | Probability of surviving $t$ years from age $x$ | $P(T_x > t)$ |
| ${}_tq_x$ | Probability of dying within $t$ years from age $x$ | $P(T_x \leq t)$ |
| $q_x$ | One-year mortality rate | ${}_1q_x$ |
| $p_x$ | One-year survival probability | ${}_1p_x = 1 - q_x$ |
| $\mu_x$ or $\mu(x)$ | Force of mortality at age $x$ | Instantaneous death rate |
| $l_x$ | Number of lives surviving to age $x$ (life table) | $l_0 = 100,000$ (radix) |
| $d_x$ | Number of deaths between age $x$ and $x+1$ | $d_x = l_x - l_{x+1}$ |
| $\mathring{e}_x$ | Complete life expectancy at age $x$ | $E[T_x]$ |
| $e_x$ | Curtate life expectancy at age $x$ | $E[K_x]$ |

### 2.3 Core Equations & Derivations

#### Equation 1: Survival Function
$$
S(x) = P(T_0 > x) = P(\text{newborn survives to age } x)
$$

**Properties:**
- $S(0) = 1$ (everyone is alive at birth)
- $S(\infty) = 0$ (everyone eventually dies)
- $S(x)$ is non-increasing

**Example:**
If $S(60) = 0.85$, then 85% of newborns survive to age 60.

#### Equation 2: Survival Probability from Age $x$
$$
{}_tp_x = P(T_x > t) = \frac{S(x+t)}{S(x)}
$$

**Derivation:**
$$
{}_tp_x = P(T_0 > x+t | T_0 > x) = \frac{P(T_0 > x+t)}{P(T_0 > x)} = \frac{S(x+t)}{S(x)}
$$

**Example:**
- $S(60) = 0.85, S(70) = 0.70$
- ${}_{10}p_{60} = \frac{0.70}{0.85} = 0.8235$ (82.35% of 60-year-olds survive to 70)

#### Equation 3: Mortality Probability
$$
{}_tq_x = P(T_x \leq t) = 1 - {}_tp_x = 1 - \frac{S(x+t)}{S(x)}
$$

**One-Year Mortality Rate:**
$$
q_x = {}_1q_x = 1 - p_x = 1 - \frac{S(x+1)}{S(x)}
$$

**Example:**
- $S(60) = 0.85, S(61) = 0.8457$
- $q_{60} = 1 - \frac{0.8457}{0.85} = 1 - 0.9949 = 0.0051$ (0.51% mortality rate)

#### Equation 4: Force of Mortality
$$
\mu_x = \mu(x) = \lim_{h \to 0} \frac{{}_hq_x}{h} = \frac{-S'(x)}{S(x)} = -\frac{d}{dx} \ln S(x)
$$

**Interpretation:** Instantaneous death rate at age $x$ (hazard function).

**Relationship to Survival Function:**
$$
S(x) = \exp\left(-\int_0^x \mu_t dt\right)
$$

**Example:**
If $\mu_x = 0.001$ (constant), then:
$$
S(x) = e^{-0.001x}
$$
$$
S(60) = e^{-0.06} = 0.9418
$$

#### Equation 5: Deferred Mortality Probability
$$
{}_{t|u}q_x = P(t < T_x \leq t+u) = {}_tp_x \times {}_uq_{x+t}
$$

**Interpretation:** Probability of dying between ages $x+t$ and $x+t+u$.

**Example:**
Probability a 60-year-old dies between ages 70 and 75:
$$
{}_{10|5}q_{60} = {}_{10}p_{60} \times {}_5q_{70}
$$

If ${}_{10}p_{60} = 0.90$ and ${}_5q_{70} = 0.05$:
$$
{}_{10|5}q_{60} = 0.90 \times 0.05 = 0.045 \text{ (4.5%)}
$$

#### Equation 6: Complete Life Expectancy
$$
\mathring{e}_x = E[T_x] = \int_0^\infty t \cdot f_{T_x}(t) dt = \int_0^\infty {}_tp_x dt
$$

**Derivation (Integration by Parts):**
$$
\mathring{e}_x = \int_0^\infty {}_tp_x dt
$$

**Example:**
If ${}_tp_{60} = e^{-0.02t}$ (exponential survival):
$$
\mathring{e}_{60} = \int_0^\infty e^{-0.02t} dt = \frac{1}{0.02} = 50 \text{ years}
$$

#### Equation 7: Curtate Life Expectancy
$$
e_x = E[K_x] = \sum_{k=1}^\infty k \cdot P(K_x = k) = \sum_{k=1}^\infty {}_kp_x
$$

**Relationship:**
$$
e_x \approx \mathring{e}_x - 0.5 \quad \text{(under uniform distribution of deaths assumption)}
$$

**Example:**
If $\mathring{e}_{60} = 25$ years, then $e_{60} \approx 24.5$ years.

#### Equation 8: Life Table Relationships
$$
l_{x+1} = l_x - d_x = l_x \times p_x
$$
$$
d_x = l_x \times q_x
$$
$$
{}_tp_x = \frac{l_{x+t}}{l_x}
$$

**Example Life Table:**
| Age $x$ | $l_x$ | $d_x$ | $q_x$ | $p_x$ | $\mathring{e}_x$ |
|---------|-------|-------|-------|-------|------------------|
| 60 | 85,000 | 434 | 0.0051 | 0.9949 | 25.0 |
| 61 | 84,566 | 456 | 0.0054 | 0.9946 | 24.1 |
| 62 | 84,110 | 479 | 0.0057 | 0.9943 | 23.2 |

### 2.4 Special Cases & Variants

**Case 1: Constant Force of Mortality (Exponential Distribution)**
If $\mu_x = \mu$ (constant), then:
$$
S(x) = e^{-\mu x}, \quad {}_tp_x = e^{-\mu t}, \quad \mathring{e}_x = \frac{1}{\mu}
$$

**Case 2: Gompertz Law**
$$
\mu_x = B c^x, \quad B, c > 0
$$
Mortality increases exponentially with age (fits human mortality well for ages 30-90).

**Case 3: Makeham's Law**
$$
\mu_x = A + B c^x
$$
Adds a constant term $A$ (accident hazard) to Gompertz.

**Case 4: De Moivre's Law**
Assumes linear survival function:
$$
S(x) = 1 - \frac{x}{\omega}, \quad 0 \leq x \leq \omega
$$
where $\omega$ is the limiting age (everyone dies by age $\omega$).

---

## 3. Theoretical Properties

### 3.1 Key Properties

1. **Property: Multiplicative Property of Survival**
   - **Statement:** ${}_{t+s}p_x = {}_tp_x \times {}_sp_{x+t}$
   - **Proof:** ${}_{t+s}p_x = \frac{S(x+t+s)}{S(x)} = \frac{S(x+t)}{S(x)} \times \frac{S(x+t+s)}{S(x+t)}$
   - **Practical Implication:** Can compute long-term survival by multiplying short-term probabilities

2. **Property: Memoryless Property (Exponential Only)**
   - **Statement:** If $\mu_x = \mu$ (constant), then ${}_tp_x = e^{-\mu t}$ (independent of $x$)
   - **Proof:** Exponential distribution is memoryless
   - **Practical Implication:** Constant force of mortality means age doesn't matter (unrealistic for humans)

3. **Property: Relationship Between $\mathring{e}_x$ and $e_x$**
   - **Statement:** $\mathring{e}_x = e_x + \frac{1}{2}$ (approximately, under UDD)
   - **Proof:** Under uniform distribution of deaths, average death occurs at mid-year
   - **Practical Implication:** Curtate expectancy is about 0.5 years less than complete expectancy

4. **Property: Force of Mortality Determines Distribution**
   - **Statement:** $S(x) = \exp\left(-\int_0^x \mu_t dt\right)$
   - **Proof:** Solve differential equation $S'(x) = -\mu_x S(x)$
   - **Practical Implication:** Specifying $\mu_x$ fully determines the survival model

### 3.2 Strengths
✓ **Rigorous:** Mathematical framework is well-established
✓ **Flexible:** Can model various mortality patterns (Gompertz, Makeham, etc.)
✓ **Data-Driven:** Life tables are based on empirical mortality data
✓ **Regulatory:** Widely accepted for pricing and reserving
✓ **Interpretable:** Survival probabilities have clear business meaning

### 3.3 Limitations
✗ **Stationarity:** Standard tables assume mortality doesn't improve over time
✗ **Homogeneity:** Single table doesn't capture heterogeneity (gender, health, socioeconomic)
✗ **Independence:** Assumes lives are independent (violated for spouses, families)
✗ **Data Quality:** Life tables depend on quality of underlying mortality data
✗ **Extreme Ages:** Limited data at very old ages (100+); extrapolation is uncertain

### 3.4 Comparison of Mortality Laws

| Law | Force of Mortality $\mu_x$ | Characteristics | Use Case |
|-----|---------------------------|-----------------|----------|
| **Constant** | $\mu$ | Exponential survival; memoryless | Theoretical baseline |
| **Gompertz** | $B c^x$ | Exponential increase with age | Ages 30-90 (human mortality) |
| **Makeham** | $A + B c^x$ | Gompertz + constant (accidents) | All ages |
| **De Moivre** | $\frac{1}{\omega - x}$ | Linear survival; limiting age | Simple calculations |
| **Weibull** | $k x^{k-1}$ | Flexible shape | Reliability engineering |

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

**For Life Table Construction:**
- **Exposure Data:** Number of lives at each age (person-years)
- **Death Counts:** Number of deaths at each age
- **Sample Size:** 100,000+ lives for stable estimates
- **Time Period:** Typically 3-5 years of data

**For Mortality Modeling:**
- **Historical Mortality Rates:** $q_x$ by age and year
- **Covariates:** Gender, smoking status, health class (for select tables)

**Data Quality Considerations:**
- **Completeness:** No missing ages
- **Accuracy:** Deaths must be correctly attributed to age
- **Timeliness:** Recent data (mortality improves over time)
- **Consistency:** Exposure and deaths must align

### 4.2 Preprocessing Steps

**Step 1: Calculate Crude Mortality Rates**
```
For each age x:
  q_x = (Number of deaths at age x) / (Exposure at age x)
```

**Step 2: Smooth Mortality Rates**
```
- Use graduation techniques (e.g., Whittaker-Henderson)
- Fit parametric model (e.g., Gompertz-Makeham)
- Ensure monotonicity (q_x increases with age for most ages)
```

**Step 3: Construct Life Table**
```
- Set radix: l_0 = 100,000
- Calculate l_x recursively: l_{x+1} = l_x * (1 - q_x)
- Calculate d_x: d_x = l_x * q_x
- Calculate e_x: e_x = sum of (l_{x+k} / l_x) for k = 1, 2, ...
```

### 4.3 Model Specification

**Gompertz-Makeham Model:**
$$
\mu_x = A + B c^x
$$

**Parameter Estimation (Maximum Likelihood):**
Maximize:
$$
L(A, B, c) = \prod_{x} \left[{}_1p_x\right]^{E_x - D_x} \left[{}_1q_x\right]^{D_x}
$$

where $E_x$ = exposure at age $x$, $D_x$ = deaths at age $x$.

**Software Implementation:**
```python
import numpy as np
from scipy.optimize import minimize

def gompertz_makeham_mu(x, A, B, c):
    """Force of mortality under Gompertz-Makeham"""
    return A + B * c**x

def survival_prob(x, t, A, B, c):
    """Survival probability from age x to x+t"""
    # Integrate force of mortality
    integral = A * t + B * (c**(x+t) - c**x) / np.log(c)
    return np.exp(-integral)

def mortality_rate(x, A, B, c):
    """One-year mortality rate q_x"""
    return 1 - survival_prob(x, 1, A, B, c)

# Example: Fit Gompertz-Makeham to data
ages = np.arange(30, 100)
observed_qx = np.array([...])  # Observed mortality rates

def neg_log_likelihood(params):
    A, B, c = params
    predicted_qx = np.array([mortality_rate(x, A, B, c) for x in ages])
    # Assuming binomial deaths
    ll = np.sum(observed_qx * np.log(predicted_qx) + 
                (1 - observed_qx) * np.log(1 - predicted_qx))
    return -ll

# Optimize
initial_params = [0.0001, 0.00005, 1.10]
result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')
A_hat, B_hat, c_hat = result.x

print(f"Fitted Parameters: A={A_hat:.6f}, B={B_hat:.6f}, c={c_hat:.4f}")

# Construct life table
l_0 = 100000
l_x = [l_0]
for x in range(100):
    p_x = survival_prob(x, 1, A_hat, B_hat, c_hat)
    l_x.append(l_x[-1] * p_x)

# Calculate life expectancy
def life_expectancy(x, A, B, c, max_age=120):
    """Complete life expectancy at age x"""
    ages_future = np.arange(x, max_age)
    p_xt = np.array([survival_prob(x, t, A, B, c) for t in range(max_age - x)])
    return np.sum(p_xt)

e_60 = life_expectancy(60, A_hat, B_hat, c_hat)
print(f"Life expectancy at age 60: {e_60:.2f} years")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1. **Life Table:** $l_x, d_x, q_x, p_x, \mathring{e}_x$ for all ages
2. **Survival Probabilities:** ${}_tp_x$ for various $t$
3. **Life Expectancy:** $\mathring{e}_x$ at key ages (e.g., 0, 25, 60, 65)

**Example Output:**
- $q_{60} = 0.0051$ (0.51% annual mortality rate at age 60)
- ${}_{10}p_{60} = 0.9500$ (95% survive from 60 to 70)
- $\mathring{e}_{60} = 25.0$ years (expected remaining lifetime)

**Interpretation:**
- **$q_x$:** Used directly in pricing and reserving formulas
- **${}_tp_x$:** Probability of paying death benefit in year $t$
- **$\mathring{e}_x$:** Average payout duration for annuities

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Goodness-of-Fit:**
- **Chi-Square Test:** Compare observed vs. expected deaths
  - $\chi^2 = \sum \frac{(D_x - E_x q_x)^2}{E_x q_x}$
- **Visual Inspection:** Plot $\log q_x$ vs. age (should be roughly linear for Gompertz)

**Graduation Tests:**
- **Signs Test:** Check for runs of positive/negative deviations
- **Grouping of Signs Test:** Ensure deviations are random, not systematic

### 5.2 Performance Metrics

**For Mortality Tables:**
- **Actual-to-Expected (A/E) Ratio:** $\frac{\text{Actual Deaths}}{\text{Expected Deaths}}$
  - Target: 0.95-1.05 (within 5% of expected)
- **Mortality Improvement:** Year-over-year change in $q_x$
  - Typical: 1-2% annual improvement

### 5.3 Validation Techniques

**Holdout Validation:**
- Fit model on years 2015-2019
- Validate on year 2020
- Check if 2020 deaths fall within 95% confidence interval

**Cross-Validation:**
- Fit model on different subsets (e.g., by region)
- Check consistency of parameters

**Backtesting:**
- Use historical table (e.g., 2001 CSO)
- Compare predicted vs. actual mortality over 20 years

### 5.4 Sensitivity Analysis

| Parameter | Base | +10% | -10% | Impact on $\mathring{e}_{60}$ |
|-----------|------|------|------|-------------------------------|
| $A$ | 0.0001 | 0.00011 | 0.00009 | -0.5 years / +0.5 years |
| $B$ | 0.00005 | 0.000055 | 0.000045 | -1.2 years / +1.3 years |
| $c$ | 1.10 | 1.11 | 1.09 | -2.0 years / +2.2 years |

**Interpretation:** Life expectancy is most sensitive to $c$ (rate of mortality increase with age).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1. **Trap: Confusing $T_x$ and $K_x$**
   - **Why it's tricky:** $T_x$ is continuous (exact time), $K_x$ is discrete (integer years)
   - **How to avoid:** $K_x = \lfloor T_x \rfloor$ (floor function)
   - **Example:** If death occurs at age 60.7, then $T_{60} = 0.7$ but $K_{60} = 0$

2. **Trap: Confusing ${}_tp_x$ and $p_{x+t}$**
   - **Why it's tricky:** Different starting ages
   - **How to avoid:** ${}_tp_x$ starts at age $x$; $p_{x+t}$ is one-year survival from age $x+t$
   - **Example:** ${}_{10}p_{60}$ is 10-year survival from 60; $p_{70}$ is 1-year survival from 70

3. **Trap: Thinking Life Expectancy is Maximum Age**
   - **Why it's tricky:** $\mathring{e}_x$ is an average, not a maximum
   - **How to avoid:** Many people live longer than $\mathring{e}_x$ (it's the mean, not the mode or max)
   - **Example:** If $\mathring{e}_{60} = 25$, some will live to 100+ (age 85+)

### 6.2 Implementation Challenges

1. **Challenge: Extrapolating to Extreme Ages**
   - **Symptom:** Limited data above age 100; $q_x$ estimates are unstable
   - **Diagnosis:** Small sample size at old ages
   - **Solution:** Use parametric model (Gompertz) or cap at limiting age (e.g., $q_{120} = 1$)

2. **Challenge: Mortality Improvement**
   - **Symptom:** Historical table underestimates current survival
   - **Diagnosis:** Mortality has improved since table was created
   - **Solution:** Apply improvement scales (e.g., Scale MP-2021)

3. **Challenge: Select vs. Ultimate Mortality**
   - **Symptom:** Recently underwritten lives have lower mortality than general population
   - **Diagnosis:** Underwriting selects healthier lives
   - **Solution:** Use select tables for first 5-15 years, then ultimate table

### 6.3 Interpretation Errors

1. **Error: Thinking $q_x = 0.01$ Means "1% Die"**
   - **Wrong:** "1% of all people die at age $x$"
   - **Right:** "1% of people alive at age $x$ die before age $x+1$"

2. **Error: Adding Survival Probabilities**
   - **Wrong:** ${}_{10}p_{60} + {}_{10}p_{70} = {}_{20}p_{60}$
   - **Right:** ${}_{20}p_{60} = {}_{10}p_{60} \times {}_{10}p_{70}$ (multiply, not add)

### 6.4 Edge Cases

**Edge Case 1: Infant Mortality**
- **Problem:** $q_0$ is much higher than $q_1$ (U-shaped mortality curve)
- **Workaround:** Use separate model for ages 0-5

**Edge Case 2: Limiting Age**
- **Problem:** What is $q_{\omega}$ where $\omega$ is the limiting age?
- **Workaround:** Set $q_{\omega} = 1$ (everyone dies by limiting age)

**Edge Case 3: Negative Force of Mortality**
- **Problem:** If $S(x)$ increases, then $\mu_x < 0$ (impossible)
- **Workaround:** Ensure $S(x)$ is non-increasing (monotonicity constraint in graduation)

---

## 7. Advanced Topics & Extensions

### 7.1 Modern Variants

**Extension 1: Stochastic Mortality Models**
- **Key Idea:** Mortality rates are random variables (e.g., Lee-Carter model)
- **Benefit:** Captures uncertainty in future mortality
- **Reference:** Used for longevity risk management in annuities

**Extension 2: Cause-of-Death Models**
- **Key Idea:** Decompose $\mu_x$ into causes (heart disease, cancer, accidents)
- **Benefit:** Understand drivers of mortality; model medical advances
- **Reference:** Used in population forecasting

**Extension 3: Frailty Models**
- **Key Idea:** Heterogeneity in population (some are frailer than others)
- **Benefit:** Explains mortality plateaus at old ages
- **Reference:** Used in longevity research

### 7.2 Integration with Other Methods

**Combination 1: Survival Models + Interest Theory**
- **Use Case:** Actuarial present value of life insurance
- **Example:** $APV = \sum_{t=1}^{\infty} v^t \times {}_tp_x \times q_{x+t} \times B_t$

**Combination 2: Survival Models + Regression**
- **Use Case:** Mortality by covariates (gender, smoking, BMI)
- **Example:** $\log q_x = \beta_0 + \beta_1 \times \text{Age} + \beta_2 \times \text{Smoking}$

### 7.3 Cutting-Edge Research

**Topic 1: Machine Learning for Mortality Prediction**
- **Description:** Use neural networks to predict mortality from health data
- **Reference:** Emerging in research; not yet standard practice

**Topic 2: COVID-19 Impact on Mortality**
- **Description:** Modeling pandemic effects on life tables
- **Reference:** 2020-2021 mortality spikes; long-term effects uncertain

**Topic 3: Longevity Bonds**
- **Description:** Financial instruments that hedge longevity risk
- **Reference:** Proposed but limited market adoption

---

## 8. Regulatory & Governance Considerations

### 8.1 Regulatory Perspective

**Acceptability:**
- **Widely Accepted:** Yes, life tables are standard for pricing and reserving
- **Jurisdictions:** All (SOA, CAS, IAA)
- **Documentation Required:** Actuaries must disclose mortality assumptions in opinions

**Key Regulatory Concerns:**
1. **Concern: Mortality Assumptions**
   - **Issue:** Are assumed mortality rates reasonable?
   - **Mitigation:** Use prescribed tables (e.g., 2017 CSO) or justify deviations

2. **Concern: Mortality Improvement**
   - **Issue:** Are future improvements adequately modeled?
   - **Mitigation:** Use improvement scales (e.g., Scale MP-2021)

### 8.2 Model Governance

**Model Risk Rating:** Medium-High
- **Justification:** Mortality assumptions directly impact pricing and reserves; errors can lead to losses

**Validation Frequency:** Annual (or upon material change)

**Key Validation Tests:**
1. **A/E Ratio:** Compare actual deaths to expected
2. **Trend Analysis:** Monitor mortality improvement over time
3. **Benchmarking:** Compare to industry tables

### 8.3 Documentation Requirements

**Minimum Documentation:**
- ✓ Mortality table used (e.g., 2017 CSO)
- ✓ Improvement scale (if applicable)
- ✓ Rationale for table selection
- ✓ A/E ratio analysis
- ✓ Sensitivity analysis (impact of ±10% mortality)

---

## 9. Practical Example

### 9.1 Worked Example: Life Insurance Pricing

**Scenario:** Calculate the net single premium (NSP) for a $100,000 whole life insurance policy issued to a 60-year-old. Use the following simplified mortality table and 4% interest rate.

| Age $x$ | $l_x$ | $q_x$ |
|---------|-------|-------|
| 60 | 85,000 | 0.0051 |
| 61 | 84,566 | 0.0054 |
| 62 | 84,110 | 0.0057 |
| ... | ... | ... |
| 100 | 1,000 | 1.0000 |

**Step 1: Calculate Death Probabilities**

The probability a 60-year-old dies in year $t$ is:
$$
{}_{t-1}p_{60} \times q_{60+t-1}
$$

For year 1:
$$
p_{60}^{(0)} \times q_{60} = 1 \times 0.0051 = 0.0051
$$

For year 2:
$$
p_{60} \times q_{61} = (1 - 0.0051) \times 0.0054 = 0.9949 \times 0.0054 = 0.00537
$$

**Step 2: Calculate Present Value of Death Benefit**

The death benefit is paid at the end of the year of death. The actuarial present value (APV) is:
$$
APV = \sum_{t=1}^{40} v^t \times {}_{t-1}p_{60} \times q_{60+t-1} \times 100,000
$$

where $v = 1/1.04 = 0.96154$.

**Numerical Calculation:**

| Year $t$ | ${}_{t-1}p_{60}$ | $q_{60+t-1}$ | Death Prob | $v^t$ | $APV_t$ |
|----------|------------------|--------------|------------|-------|---------|
| 1 | 1.0000 | 0.0051 | 0.00510 | 0.96154 | $490.39 |
| 2 | 0.9949 | 0.0054 | 0.00537 | 0.92456 | $496.53 |
| 3 | 0.9895 | 0.0057 | 0.00564 | 0.88900 | $501.39 |
| ... | ... | ... | ... | ... | ... |
| 40 | 0.0118 | 1.0000 | 0.01176 | 0.20829 | $245.03 |

**Total APV (Net Single Premium):** $\approx \$35,000$

**Interpretation:** A 60-year-old should pay a one-time premium of $35,000 for a $100,000 whole life policy (assuming 4% interest and given mortality).

**Step 3: Sensitivity Analysis**

What if mortality is 10% higher (e.g., $q_x$ multiplied by 1.1)?
- New APV: $\approx \$38,500$ (+10% mortality → +10% premium)

What if interest rate is 3% instead of 4%?
- New APV: $\approx \$42,000$ (lower discount rate → higher PV)

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1. **Survival models quantify probability of living or dying** using $S(x), {}_tp_x, q_x, \mu_x$
2. **Life tables summarize mortality experience** with $l_x, d_x, q_x, \mathring{e}_x$
3. **Force of mortality $\mu_x$ is the instantaneous death rate**; determines survival function
4. **Gompertz-Makeham law fits human mortality well** for ages 30-90
5. **Life expectancy $\mathring{e}_x$ is the expected remaining lifetime** at age $x$

### 10.2 When to Use This Knowledge
**Ideal For:**
- ✓ SOA Exam LTAM preparation
- ✓ Life insurance pricing and reserving
- ✓ Annuity valuation
- ✓ Pension obligation calculation
- ✓ Longevity risk assessment

**Not Ideal For:**
- ✗ Short-term insurance (P&C) - mortality is not the primary risk
- ✗ Disability insurance - need morbidity models, not just mortality

### 10.3 Critical Success Factors
1. **Master the Notation:** Understand ${}_tp_x, q_x, \mu_x$ and their relationships
2. **Practice Calculations:** Compute survival probabilities, life expectancy by hand
3. **Understand Life Tables:** Know how to read and interpret $l_x, d_x, q_x$
4. **Learn Mortality Laws:** Gompertz, Makeham, De Moivre
5. **Apply to Insurance:** Connect survival models to pricing and reserving

### 10.4 Further Reading
- **Textbook:** "Actuarial Mathematics for Life Contingent Risks" by Dickson, Hardy & Waters (Chapter 2-3)
- **Exam Prep:** Coaching Actuaries Adapt for Exam LTAM
- **SOA Tables:** 2017 CSO Mortality Table, Scale MP-2021
- **Online:** SOA mortality improvement resources
- **Advanced:** "Modelling Longevity Dynamics for Pensions and Annuity Business" by Cairns, Blake & Dowd

---

## Appendix

### A. Glossary
- **Radix:** Starting number of lives in a life table (typically $l_0 = 100,000$)
- **Select Period:** Initial years after underwriting when mortality is lower
- **Ultimate Table:** Mortality rates after select period ends
- **Graduation:** Smoothing of crude mortality rates
- **Mortality Improvement:** Reduction in mortality rates over time

### B. Key Formulas Summary

| Formula | Equation | Use |
|---------|----------|-----|
| **Survival Probability** | ${}_tp_x = \frac{S(x+t)}{S(x)}$ | Prob. of surviving $t$ years from age $x$ |
| **Mortality Probability** | ${}_tq_x = 1 - {}_tp_x$ | Prob. of dying within $t$ years from age $x$ |
| **Force of Mortality** | $\mu_x = -\frac{d}{dx} \ln S(x)$ | Instantaneous death rate |
| **Life Expectancy** | $\mathring{e}_x = \int_0^\infty {}_tp_x dt$ | Expected remaining lifetime |
| **Life Table** | $l_{x+1} = l_x (1 - q_x)$ | Recursive construction |
| **Gompertz** | $\mu_x = B c^x$ | Exponential mortality increase |
| **Makeham** | $\mu_x = A + B c^x$ | Gompertz + constant |

### C. Standard Mortality Tables

| Table | Description | Use |
|-------|-------------|-----|
| **2017 CSO** | Commissioners Standard Ordinary | US life insurance statutory reserves |
| **2012 IAM** | Individual Annuity Mortality | US annuity pricing |
| **CPM-2014** | Canadian Pensioners Mortality | Canadian pensions |
| **CMI Tables** | Continuous Mortality Investigation | UK life insurance and pensions |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,300+*
