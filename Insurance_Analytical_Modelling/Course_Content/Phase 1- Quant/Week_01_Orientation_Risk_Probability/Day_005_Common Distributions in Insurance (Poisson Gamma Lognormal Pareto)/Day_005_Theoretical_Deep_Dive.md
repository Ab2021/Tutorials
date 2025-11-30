# Common Distributions in Insurance - Theoretical Deep Dive

## Overview
This session provides comprehensive coverage of the probability distributions most commonly used in insurance analytics. These distributions model both claim frequency (how often claims occur) and claim severity (how large claims are). Mastery of these distributions is essential for SOA Exam P, CAS Exam 1, and all actuarial modeling work in pricing, reserving, and risk management.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Probability Distributions** are mathematical functions that describe the likelihood of different outcomes for a random variable. In insurance, we use:
- **Discrete Distributions** for count data (number of claims): Poisson, Negative Binomial, Binomial
- **Continuous Distributions** for amount data (claim sizes): Gamma, Lognormal, Pareto, Weibull, Exponential

**Key Terminology:**
- **Probability Mass Function (PMF):** For discrete distributions, $P(X = x)$
- **Probability Density Function (PDF):** For continuous distributions, $f(x)$
- **Cumulative Distribution Function (CDF):** $F(x) = P(X \leq x)$
- **Survival Function:** $S(x) = 1 - F(x) = P(X > x)$
- **Hazard Function:** $h(x) = f(x) / S(x)$ (instantaneous failure rate)
- **Mean (Expected Value):** $E[X] = \mu$
- **Variance:** $Var(X) = E[(X-\mu)^2] = \sigma^2$
- **Skewness:** Measure of asymmetry (positive skew = long right tail)
- **Kurtosis:** Measure of tail heaviness

### 1.2 Historical Context & Evolution

**Origin:**
- **Poisson (1837):** Siméon Denis Poisson developed the Poisson distribution for modeling rare events
- **Gamma/Exponential (1800s):** Emerged from studies of radioactive decay and waiting times
- **Lognormal (1879):** Francis Galton studied lognormal distributions in biological data
- **Pareto (1896):** Vilfredo Pareto observed wealth distribution follows a power law
- **Weibull (1951):** Waloddi Weibull developed the distribution for material strength analysis

**Evolution in Insurance:**
- **1900s-1950s:** Actuaries used simple distributions (exponential, normal) due to computational limits
- **1960s-1980s:** Computers enabled fitting of more complex distributions (Gamma, Lognormal)
- **1990s-2000s:** GLMs standardized use of Poisson (frequency) and Gamma (severity)
- **2010s-Present:** Machine learning complements traditional distributions; mixture models and heavy-tailed distributions (Pareto) gain prominence for extreme events

**Current State:**
Modern actuarial practice uses a **frequency-severity framework**:
- **Frequency Model:** Poisson or Negative Binomial for claim counts
- **Severity Model:** Gamma, Lognormal, or Pareto for claim amounts
- **Aggregate Loss:** Convolution of frequency and severity (often via simulation)

### 1.3 Why This Matters

**Business Impact:**
- **Pricing:** Correct distribution choice ensures adequate premiums
- **Reserving:** Tail distributions (Pareto) are critical for large loss reserves
- **Risk Management:** Heavy-tailed distributions quantify extreme risk (99.5th percentile for Solvency II)
- **Reinsurance:** Excess-of-loss pricing depends on severity distribution tail

**Regulatory Relevance:**
- **Principle-Based Reserving (PBR):** Requires stochastic modeling with appropriate distributions
- **Solvency II:** SCR calculation uses distribution assumptions for 1-in-200 year events
- **IFRS 17:** Risk adjustment depends on distribution choice and tail uncertainty

**Industry Adoption:**
- **Auto Insurance:** Poisson (frequency), Gamma (severity)
- **Homeowners:** Negative Binomial (frequency due to overdispersion), Lognormal (severity)
- **Workers' Comp:** Pareto (severity for large medical claims)
- **Catastrophe Modeling:** Generalized Pareto for extreme value theory

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1. **Assumption: Data Follows a Specific Distribution**
   - **Description:** We assume claim counts or amounts follow a known distribution family (e.g., Poisson, Gamma)
   - **Implication:** Parameters can be estimated from data; predictions rely on this assumption
   - **Real-world validity:** No distribution is perfect; goodness-of-fit tests assess validity

2. **Assumption: Independence and Identical Distribution (i.i.d.)**
   - **Description:** Claims are independent and identically distributed
   - **Implication:** We can use standard statistical methods (MLE, method of moments)
   - **Real-world validity:** Violated in catastrophes (dependence) and when risk profiles change over time (non-stationarity)

3. **Assumption: Parameters are Constant**
   - **Description:** Distribution parameters (e.g., Poisson $\lambda$) don't change over time or across policies
   - **Implication:** We can pool data to estimate parameters
   - **Real-world validity:** Often violated; GLMs allow parameters to vary by covariates

4. **Assumption: Sufficient Data for Estimation**
   - **Description:** Sample size is large enough for reliable parameter estimates
   - **Implication:** Small samples lead to high estimation error
   - **Real-world validity:** Credibility theory addresses limited data

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $X$ | Random variable (claim count or amount) | Number of claims |
| $f(x)$ | PDF (continuous) or PMF (discrete) | $f(x; \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$ |
| $F(x)$ | CDF | $P(X \leq x)$ |
| $S(x)$ | Survival function | $P(X > x) = 1 - F(x)$ |
| $\lambda$ | Parameter (often rate or scale) | 0.08 claims/policy/year |
| $\alpha, \beta$ | Shape and scale parameters | Gamma($\alpha=2, \beta=5000$) |
| $\mu, \sigma$ | Mean and standard deviation | $\mu = 3500, \sigma = 2000$ |

### 2.3 Core Equations & Derivations

---

## FREQUENCY DISTRIBUTIONS

### Equation 1: Poisson Distribution

**PMF:**
$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
$$

**Where:** $\lambda > 0$ is the rate parameter (expected number of claims)

**Properties:**
- $E[X] = \lambda$
- $Var(X) = \lambda$ (equidispersion: mean = variance)
- $P(X = 0) = e^{-\lambda}$ (probability of no claims)

**Derivation of Mean:**
$$
E[X] = \sum_{k=0}^\infty k \frac{\lambda^k e^{-\lambda}}{k!} = \lambda e^{-\lambda} \sum_{k=1}^\infty \frac{\lambda^{k-1}}{(k-1)!} = \lambda e^{-\lambda} e^{\lambda} = \lambda
$$

**Example:**
If $\lambda = 0.08$ (8% of policies have a claim per year):
- $P(X = 0) = e^{-0.08} = 0.923$ (92.3% have no claims)
- $P(X = 1) = 0.08 \times e^{-0.08} = 0.074$ (7.4% have 1 claim)
- $P(X \geq 2) = 1 - P(X=0) - P(X=1) = 0.003$ (0.3% have 2+ claims)

**When to Use:**
- Claim counts when events are rare and independent
- Short-term insurance (auto, homeowners)
- Assumption: mean = variance holds

---

### Equation 2: Negative Binomial Distribution

**PMF:**
$$
P(X = k) = \binom{k + r - 1}{k} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots
$$

**Alternative Parameterization (mean-variance):**
$$
E[X] = \mu, \quad Var(X) = \mu + \frac{\mu^2}{\theta}
$$

**Where:** $\theta > 0$ is the dispersion parameter

**Properties:**
- $E[X] = \frac{r(1-p)}{p}$ or $\mu$ in alternative form
- $Var(X) > E[X]$ (overdispersion)
- As $\theta \to \infty$, Negative Binomial → Poisson

**When to Use:**
- Claim counts with overdispersion (variance > mean)
- Heterogeneous risk pools (some policies are riskier than others)
- Homeowners insurance (catastrophe-prone areas increase variance)

**Example:**
- Poisson: $E[X] = 0.08, Var(X) = 0.08$
- Negative Binomial: $E[X] = 0.08, Var(X) = 0.12$ (50% more variance)

---

## SEVERITY DISTRIBUTIONS

### Equation 3: Gamma Distribution

**PDF:**
$$
f(x; \alpha, \beta) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}, \quad x > 0
$$

**Where:**
- $\alpha > 0$ is the shape parameter
- $\beta > 0$ is the scale parameter
- $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} dt$ is the gamma function

**Properties:**
- $E[X] = \alpha \beta$
- $Var(X) = \alpha \beta^2$
- $CV = \frac{\sigma}{\mu} = \frac{1}{\sqrt{\alpha}}$ (coefficient of variation)
- Right-skewed (positive skew)

**Special Cases:**
- $\alpha = 1$: Exponential distribution
- $\alpha = n/2, \beta = 2$: Chi-square distribution with $n$ degrees of freedom

**Example:**
Auto insurance claim severity: $\alpha = 2, \beta = 1750$
- $E[X] = 2 \times 1750 = \$3,500$
- $Var(X) = 2 \times 1750^2 = 6,125,000$
- $\sigma = \$2,475$
- $CV = 2475 / 3500 = 0.707$

**When to Use:**
- Claim severity with moderate right skew
- Standard choice in GLMs for severity
- Flexible (shape parameter controls skewness)

---

### Equation 4: Lognormal Distribution

**PDF:**
$$
f(x; \mu, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0
$$

**Where:**
- $\mu$ is the mean of $\ln(X)$ (not the mean of $X$!)
- $\sigma$ is the standard deviation of $\ln(X)$

**Properties:**
- $E[X] = e^{\mu + \sigma^2/2}$
- $Var(X) = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$
- $Median(X) = e^{\mu}$
- Highly right-skewed (heavier tail than Gamma for large $\sigma$)

**Relationship to Normal:**
If $X \sim \text{Lognormal}(\mu, \sigma)$, then $\ln(X) \sim \text{Normal}(\mu, \sigma)$

**Example:**
Claim severity: $\mu = 8, \sigma = 0.6$
- $E[X] = e^{8 + 0.36/2} = e^{8.18} = \$3,575$
- $Median(X) = e^8 = \$2,981$
- $Var(X) = e^{16.36}(e^{0.36} - 1) = 5,250,000$

**When to Use:**
- Claim severity with heavy right tail
- When multiplicative effects are present (claim = base × factor1 × factor2 × ...)
- Alternative to Gamma in GLMs

---

### Equation 5: Pareto Distribution

**PDF:**
$$
f(x; \alpha, \theta) = \frac{\alpha \theta^\alpha}{(x + \theta)^{\alpha + 1}}, \quad x > 0
$$

**Where:**
- $\alpha > 0$ is the shape parameter (tail index)
- $\theta > 0$ is the scale parameter

**Properties:**
- $E[X] = \frac{\theta}{\alpha - 1}$ for $\alpha > 1$
- $Var(X) = \frac{\theta^2 \alpha}{(\alpha-1)^2(\alpha-2)}$ for $\alpha > 2$
- **Heavy tail:** $P(X > x) \sim x^{-\alpha}$ as $x \to \infty$
- Smaller $\alpha$ = heavier tail (more extreme events)

**Survival Function:**
$$
S(x) = \left(\frac{\theta}{x + \theta}\right)^\alpha
$$

**Example:**
Large liability claims: $\alpha = 2.5, \theta = 10,000$
- $E[X] = 10,000 / (2.5 - 1) = \$6,667$
- $P(X > 100,000) = (10,000 / 110,000)^{2.5} = 0.0067$ (0.67% exceed $100K)

**When to Use:**
- Large losses (workers' comp, liability, catastrophe)
- Extreme value theory (tail modeling)
- Reinsurance pricing (excess-of-loss)

---

### Equation 6: Weibull Distribution

**PDF:**
$$
f(x; \alpha, \beta) = \frac{\alpha}{\beta} \left(\frac{x}{\beta}\right)^{\alpha-1} e^{-(x/\beta)^\alpha}, \quad x > 0
$$

**Where:**
- $\alpha > 0$ is the shape parameter
- $\beta > 0$ is the scale parameter

**Properties:**
- $E[X] = \beta \Gamma(1 + 1/\alpha)$
- $Var(X) = \beta^2 [\Gamma(1 + 2/\alpha) - \Gamma^2(1 + 1/\alpha)]$
- **Hazard function:** $h(x) = \frac{\alpha}{\beta}(x/\beta)^{\alpha-1}$
  - $\alpha < 1$: Decreasing hazard (early failures)
  - $\alpha = 1$: Constant hazard (Exponential)
  - $\alpha > 1$: Increasing hazard (wear-out failures)

**Example:**
Time to claim (survival analysis): $\alpha = 1.5, \beta = 100$ days
- $E[X] = 100 \times \Gamma(1 + 1/1.5) = 100 \times 0.903 = 90.3$ days
- Hazard increases over time (claims become more likely as time passes)

**When to Use:**
- Survival analysis (time to claim, time to lapse)
- Reliability modeling
- Claim severity with flexible tail behavior

---

### Equation 7: Exponential Distribution

**PDF:**
$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x > 0
$$

**Properties:**
- $E[X] = 1/\lambda$
- $Var(X) = 1/\lambda^2$
- **Memoryless:** $P(X > s + t | X > s) = P(X > t)$
- Special case of Gamma ($\alpha = 1$) and Weibull ($\alpha = 1$)

**Example:**
Time between claims: $\lambda = 0.01$ per day
- $E[X] = 1/0.01 = 100$ days
- $P(X > 200) = e^{-0.01 \times 200} = e^{-2} = 0.135$ (13.5% wait >200 days)

**When to Use:**
- Simplest continuous distribution
- Waiting times (time between claims)
- Baseline model before trying more complex distributions

---

### 2.4 Special Cases & Variants

**Case 1: Zero-Inflated Poisson (ZIP)**
When there are more zeros than Poisson predicts:
$$
P(X = 0) = \pi + (1-\pi) e^{-\lambda}
$$
$$
P(X = k) = (1-\pi) \frac{\lambda^k e^{-\lambda}}{k!}, \quad k > 0
$$

**Use:** Policies with structural zeros (e.g., some never claim due to excellent driving)

**Case 2: Mixture Distributions**
Combine multiple distributions:
$$
f(x) = w_1 f_1(x) + w_2 f_2(x), \quad w_1 + w_2 = 1
$$

**Example:** 90% of claims are small (Gamma), 10% are large (Pareto)

**Case 3: Truncated and Censored Distributions**
- **Left Truncation:** Only observe claims above deductible $d$
- **Right Censoring:** Claims capped at policy limit $u$

**Truncated PDF:**
$$
f_{truncated}(x) = \frac{f(x)}{S(d)}, \quad x > d
$$

---

## 3. Theoretical Properties

### 3.1 Key Properties

1. **Property: Reproductive Property (Poisson)**
   - **Statement:** If $X_1 \sim Poisson(\lambda_1)$ and $X_2 \sim Poisson(\lambda_2)$ are independent, then $X_1 + X_2 \sim Poisson(\lambda_1 + \lambda_2)$
   - **Proof:** MGF of sum = product of MGFs
   - **Practical Implication:** Total claims across multiple policies follow Poisson if individual policies do

2. **Property: Gamma is Conjugate Prior for Poisson**
   - **Statement:** If $X | \Lambda \sim Poisson(\Lambda)$ and $\Lambda \sim Gamma(\alpha, \beta)$, then $X \sim Negative Binomial$
   - **Proof:** Bayesian derivation
   - **Practical Implication:** Negative Binomial arises naturally from heterogeneous Poisson rates

3. **Property: Lognormal Product Property**
   - **Statement:** If $X_1, X_2, \ldots, X_n$ are independent lognormal, then $\prod X_i$ is lognormal
   - **Proof:** $\ln(\prod X_i) = \sum \ln(X_i)$ is sum of normals, hence normal
   - **Practical Implication:** Multiplicative effects lead to lognormal distributions

4. **Property: Pareto Tail Behavior**
   - **Statement:** For Pareto, $\lim_{x \to \infty} \frac{S(x)}{x^{-\alpha}} = \theta^\alpha$
   - **Proof:** Direct calculation from survival function
   - **Practical Implication:** Pareto has polynomial tail decay (much heavier than exponential)

### 3.2 Strengths
✓ **Flexibility:** Wide range of distributions covers diverse insurance phenomena
✓ **Theoretical Foundation:** Well-studied mathematical properties
✓ **Computational Tractability:** Most distributions have closed-form formulas or efficient algorithms
✓ **GLM Compatibility:** Poisson, Gamma, and others fit naturally into GLM framework
✓ **Tail Modeling:** Pareto and Weibull handle extreme events

### 3.3 Limitations
✗ **Assumption Sensitivity:** Results depend on correct distribution choice
✗ **Parameter Uncertainty:** Estimated parameters have sampling error
✗ **Tail Estimation:** Limited data in tails makes extreme quantiles uncertain
✗ **Independence:** Most distributions assume independence (violated in catastrophes)
✗ **Stationarity:** Assume parameters don't change over time

### 3.4 Comparison of Distributions

| Distribution | Type | Mean | Variance | Skewness | Tail | Insurance Use |
|--------------|------|------|----------|----------|------|---------------|
| **Poisson** | Discrete | $\lambda$ | $\lambda$ | $1/\sqrt{\lambda}$ | Light | Frequency (standard) |
| **Negative Binomial** | Discrete | $\mu$ | $\mu + \mu^2/\theta$ | Positive | Medium | Frequency (overdispersed) |
| **Exponential** | Continuous | $1/\lambda$ | $1/\lambda^2$ | 2 | Light | Waiting times |
| **Gamma** | Continuous | $\alpha\beta$ | $\alpha\beta^2$ | $2/\sqrt{\alpha}$ | Medium | Severity (standard) |
| **Lognormal** | Continuous | $e^{\mu+\sigma^2/2}$ | Complex | Positive | Heavy | Severity (heavy tail) |
| **Pareto** | Continuous | $\theta/(\alpha-1)$ | Complex | Positive | Very Heavy | Large losses |
| **Weibull** | Continuous | $\beta\Gamma(1+1/\alpha)$ | Complex | Varies | Flexible | Survival, severity |

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

**For Frequency Modeling:**
- **Claim counts:** Number of claims per policy (0, 1, 2, ...)
- **Exposure:** Policy-years or other exposure measure
- **Sample size:** 1,000+ policies for stable Poisson estimates; 10,000+ for Negative Binomial

**For Severity Modeling:**
- **Claim amounts:** Individual claim sizes (positive values only)
- **Sample size:** 500+ claims for Gamma/Lognormal; 1,000+ for Pareto tail estimation
- **Censoring information:** Policy limits, deductibles

**Data Quality Considerations:**
- **Outliers:** Verify large claims (not data errors)
- **Zeros:** Decide how to handle (exclude from severity, include in zero-inflated models)
- **Truncation:** Adjust for deductibles (only see claims > $d$)
- **Currency:** Adjust for inflation over time

### 4.2 Preprocessing Steps

**Step 1: Data Cleaning**
```
- Remove duplicates
- Validate claim amounts (positive, within policy limits)
- Handle missing values (exclude or impute)
- Adjust for inflation (CPI or industry-specific index)
```

**Step 2: Exploratory Data Analysis**
```
- Histogram: Visual inspection of distribution shape
- Summary statistics: Mean, variance, skewness, kurtosis
- Q-Q plot: Compare empirical quantiles to theoretical distribution
- Identify outliers: Claims > 99th percentile
```

**Step 3: Distribution Selection**
```
- Frequency: Check if variance ≈ mean (Poisson) or variance > mean (Negative Binomial)
- Severity: Check skewness (Gamma for moderate, Lognormal/Pareto for heavy)
- Goodness-of-fit tests: Chi-square, Kolmogorov-Smirnov, Anderson-Darling
```

### 4.3 Model Specification

**Maximum Likelihood Estimation (MLE):**

**Poisson:**
$$
\hat{\lambda} = \frac{\sum X_i}{n}
$$

**Gamma:**
Solve numerically:
$$
\ln(\hat{\alpha}) - \psi(\hat{\alpha}) = \ln(\bar{X}) - \overline{\ln(X)}
$$
$$
\hat{\beta} = \bar{X} / \hat{\alpha}
$$

**Lognormal:**
$$
\hat{\mu} = \frac{1}{n}\sum \ln(X_i)
$$
$$
\hat{\sigma}^2 = \frac{1}{n}\sum (\ln(X_i) - \hat{\mu})^2
$$

**Software Implementation:**
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load claim data
claims = np.array([1200, 3500, 800, 15000, 2200, ...])  # Claim amounts

# Fit Gamma distribution
shape, loc, scale = stats.gamma.fit(claims, floc=0)
print(f"Gamma: shape={shape:.2f}, scale={scale:.2f}")
print(f"Mean: {shape * scale:.2f}")

# Fit Lognormal distribution
s, loc, scale = stats.lognorm.fit(claims, floc=0)
print(f"Lognormal: sigma={s:.2f}, scale={scale:.2f}")

# Goodness-of-fit test (Kolmogorov-Smirnov)
ks_stat_gamma, p_value_gamma = stats.kstest(claims, 'gamma', args=(shape, loc, scale))
ks_stat_lognorm, p_value_lognorm = stats.kstest(claims, 'lognorm', args=(s, loc, scale))

print(f"Gamma K-S test: statistic={ks_stat_gamma:.4f}, p-value={p_value_gamma:.4f}")
print(f"Lognormal K-S test: statistic={ks_stat_lognorm:.4f}, p-value={p_value_lognorm:.4f}")

# Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(claims, dist=stats.gamma(shape, loc, scale), plot=ax1)
ax1.set_title('Gamma Q-Q Plot')

stats.probplot(claims, dist=stats.lognorm(s, loc, scale), plot=ax2)
ax2.set_title('Lognormal Q-Q Plot')

plt.show()
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1. **Parameter Estimates:**
   - **Poisson:** $\hat{\lambda} = 0.085$ (8.5 claims per 100 policies)
   - **Gamma:** $\hat{\alpha} = 2.1, \hat{\beta} = 1667$ (mean = $3,500, CV = 0.69)

2. **Predicted Probabilities/Densities:**
   - **Frequency:** $P(X = 0) = 0.918, P(X = 1) = 0.078, P(X \geq 2) = 0.004$
   - **Severity:** $P(X > 10,000) = 0.12$ (12% of claims exceed $10K)

3. **Quantiles:**
   - **Severity:** 50th percentile = $2,800, 95th percentile = $9,500, 99th percentile = $15,000

**Diagnostic Outputs:**
- **Goodness-of-Fit:** K-S statistic = 0.032, p-value = 0.45 (fail to reject; good fit)
- **AIC/BIC:** Compare models (lower is better)
- **Residual Plots:** Check for patterns

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Goodness-of-Fit Tests:**

**1. Chi-Square Test:**
- Group data into bins
- Compare observed vs. expected frequencies
- $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$
- Reject if $\chi^2 >$ critical value

**2. Kolmogorov-Smirnov (K-S) Test:**
- Compare empirical CDF to theoretical CDF
- $D = \max_x |F_n(x) - F(x)|$
- Reject if $D >$ critical value

**3. Anderson-Darling (A-D) Test:**
- Like K-S but gives more weight to tails
- $A^2 = -n - \sum \frac{2i-1}{n}[\ln F(X_i) + \ln(1-F(X_{n+1-i}))]$

**Visual Diagnostics:**
- **Histogram vs. PDF:** Overlay theoretical density on histogram
- **Q-Q Plot:** Quantile-quantile plot (should be linear if distribution fits)
- **P-P Plot:** Probability-probability plot (alternative to Q-Q)

### 5.2 Performance Metrics

**For Distribution Fit:**
- **Log-Likelihood:** $\ell = \sum \ln f(X_i; \hat{\theta})$ (higher is better)
- **AIC:** $-2\ell + 2k$ where $k$ = # parameters (lower is better)
- **BIC:** $-2\ell + k \ln(n)$ (lower is better, penalizes complexity more than AIC)

**For Predictive Accuracy:**
- **Mean Squared Error:** $MSE = \frac{1}{n}\sum(X_i - \hat{X}_i)^2$
- **Mean Absolute Error:** $MAE = \frac{1}{n}\sum|X_i - \hat{X}_i|$

### 5.3 Validation Techniques

**Holdout Validation:**
- Fit distribution on 70% of data
- Test on remaining 30%
- Compare predicted vs. actual quantiles

**Cross-Validation:**
- k-Fold CV: Fit on k-1 folds, test on 1 fold, repeat
- Average log-likelihood across folds

**Backtesting:**
- Fit distribution on years 1-3
- Validate on year 4
- Check if actual claims fall within predicted intervals

### 5.4 Sensitivity Analysis

**Parameter Sensitivity:**
| Parameter | Base | +10% | -10% | Impact on 99th Percentile |
|-----------|------|------|------|---------------------------|
| Gamma $\alpha$ | 2.0 | 2.2 | 1.8 | -5% / +6% |
| Gamma $\beta$ | 1750 | 1925 | 1575 | +10% / -10% |
| Pareto $\alpha$ | 2.5 | 2.75 | 2.25 | -18% / +25% |

**Interpretation:** Pareto tail is very sensitive to $\alpha$ parameter.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1. **Trap: Confusing Parameters of Lognormal**
   - **Why it's tricky:** $\mu$ is NOT the mean of $X$; it's the mean of $\ln(X)$
   - **How to avoid:** Always use $E[X] = e^{\mu + \sigma^2/2}$ for the mean
   - **Example:** $\mu = 8$ does NOT mean mean claim = $8; it means median claim = $e^8 = 2981$

2. **Trap: Assuming Poisson When Variance > Mean**
   - **Why it's tricky:** Poisson requires mean = variance; real data often has variance > mean
   - **How to avoid:** Test for overdispersion; use Negative Binomial if variance > mean
   - **Example:** If mean = 0.08 but variance = 0.12, Poisson underestimates variability

3. **Trap: Ignoring Tail Behavior**
   - **Why it's tricky:** Gamma and Lognormal can have similar means but very different tails
   - **How to avoid:** Compare 95th, 99th, 99.9th percentiles, not just mean
   - **Example:** Gamma(2, 1750) and Lognormal(8, 0.6) have similar means (~$3,500) but Lognormal has much heavier tail

### 6.2 Implementation Challenges

1. **Challenge: Fitting Pareto with Limited Tail Data**
   - **Symptom:** Only 10 claims above $100K; Pareto fit is unstable
   - **Diagnosis:** Insufficient tail data
   - **Solution:** Use threshold model (GPD); combine with expert judgment

2. **Challenge: Numerical Issues in MLE**
   - **Symptom:** Optimization fails to converge for Gamma or Weibull
   - **Diagnosis:** Poor starting values or data issues (outliers)
   - **Solution:** Use method of moments for starting values; remove extreme outliers

3. **Challenge: Zero Claims in Severity Model**
   - **Symptom:** Some policies have zero claims; can't fit Gamma (requires $x > 0$)
   - **Diagnosis:** Mixing frequency and severity
   - **Solution:** Fit severity only on claims > 0; use separate frequency model for P(claim)

### 6.3 Interpretation Errors

1. **Error: Thinking Higher Mean Always Means Higher Risk**
   - **Wrong:** "Distribution A has higher mean, so it's riskier"
   - **Right:** "Risk depends on tail; Pareto(2, 5000) has lower mean than Gamma(3, 2000) but much higher 99th percentile"

2. **Error: Extrapolating Beyond Data Range**
   - **Wrong:** "My data goes up to $50K, so I'll use the fitted distribution to predict the 99.9th percentile at $500K"
   - **Right:** "Tail extrapolation is uncertain; use extreme value theory or expert judgment for values far beyond observed data"

### 6.4 Edge Cases

**Edge Case 1: All Claims are the Same Amount**
- **Problem:** Variance = 0; can't fit Gamma or Lognormal
- **Workaround:** Use deterministic model (all claims = constant)

**Edge Case 2: Single Large Claim Dominates**
- **Problem:** One claim is 10x larger than all others; distorts parameter estimates
- **Workaround:** Investigate if it's a data error; if real, consider mixture model or separate treatment

**Edge Case 3: Negative Binomial with Very Large $\theta$**
- **Problem:** As $\theta \to \infty$, Negative Binomial → Poisson
- **Workaround:** If $\theta > 1000$, just use Poisson (simpler)

---

## 7. Advanced Topics & Extensions

### 7.1 Modern Variants

**Extension 1: Tweedie Distribution**
- **Key Idea:** Combines Poisson frequency and Gamma severity into a single distribution
- **Benefit:** One model for aggregate loss (no need to separate frequency and severity)
- **Reference:** Used in GLMs for pure premium modeling
- **Formula:** Tweedie is a special case of exponential dispersion family with power parameter $p \in (1, 2)$

**Extension 2: Generalized Pareto Distribution (GPD)**
- **Key Idea:** Models exceedances over a threshold (Extreme Value Theory)
- **Benefit:** Better tail estimation than standard Pareto
- **Reference:** Used in catastrophe modeling, reinsurance
- **Formula:** $F(x) = 1 - (1 + \xi x / \sigma)^{-1/\xi}$ for $x > 0$

**Extension 3: Phase-Type Distributions**
- **Key Idea:** Represent distributions as sums of exponentials (Markov chain interpretation)
- **Benefit:** Flexible, computationally tractable
- **Reference:** Used in queueing theory, life insurance

### 7.2 Integration with Other Methods

**Combination 1: GLM + Distributions**
- **Use Case:** Allow distribution parameters to vary by covariates
- **Example:** Poisson GLM with $\log(\lambda_i) = \beta_0 + \beta_1 \text{Age}_i + \beta_2 \text{Territory}_i$

**Combination 2: Copulas + Marginal Distributions**
- **Use Case:** Model dependence between multiple risks
- **Example:** Frequency and severity may be dependent (large claims occur more often in certain conditions)

**Combination 3: Mixture Models**
- **Use Case:** Combine distributions for heterogeneous populations
- **Example:** 80% of claims follow Gamma(2, 1500), 20% follow Pareto(2.5, 5000)

### 7.3 Cutting-Edge Research

**Topic 1: Machine Learning for Distribution Selection**
- **Description:** Use ML to automatically select best distribution for each segment
- **Reference:** Ensemble methods that combine multiple distributions

**Topic 2: Non-Parametric Density Estimation**
- **Description:** Kernel density estimation, splines (no parametric assumption)
- **Reference:** Used when no standard distribution fits well

**Topic 3: Bayesian Hierarchical Models**
- **Description:** Estimate distribution parameters with uncertainty (posterior distributions)
- **Reference:** Used in credibility theory, small sample situations

---

## 8. Regulatory & Governance Considerations

### 8.1 Regulatory Perspective

**Acceptability:**
- **Widely Accepted:** Yes, standard distributions are well-established in actuarial practice
- **Jurisdictions:** SOA, CAS, IAA all require understanding of distributions
- **Documentation Required:** Actuaries must justify distribution choice in rate filings and reserve opinions

**Key Regulatory Concerns:**
1. **Concern: Tail Risk**
   - **Issue:** Are extreme events adequately modeled?
   - **Mitigation:** Use heavy-tailed distributions (Pareto) for large losses; stress test

2. **Concern: Model Uncertainty**
   - **Issue:** Parameter estimates have sampling error
   - **Mitigation:** Confidence intervals, sensitivity analysis, margins for adverse deviation

### 8.2 Model Governance

**Model Risk Rating:** Medium-High
- **Justification:** Distribution choice directly impacts pricing and reserving; wrong distribution leads to inadequate rates/reserves

**Validation Frequency:** Annual (or upon material change in data)

**Key Validation Tests:**
1. **Goodness-of-Fit:** K-S test, A-D test, visual diagnostics
2. **Backtesting:** Do predicted quantiles match actual experience?
3. **Sensitivity:** How sensitive are results to distribution choice?

### 8.3 Documentation Requirements

**Minimum Documentation:**
- ✓ Distribution selected and rationale
- ✓ Parameter estimates and estimation method (MLE, method of moments)
- ✓ Goodness-of-fit test results
- ✓ Comparison of alternative distributions (AIC, BIC)
- ✓ Sensitivity analysis (impact of parameter uncertainty)
- ✓ Limitations (e.g., limited tail data)

---

## 9. Practical Example

### 9.1 Worked Example: Auto Insurance Severity Modeling

**Scenario:** An auto insurer has 5,000 collision claims with the following summary statistics:
- Mean: $3,500
- Median: $2,800
- Standard Deviation: $2,500
- Skewness: 1.8 (positive skew)
- 95th percentile: $9,000
- 99th percentile: $15,000

**Task:** Fit Gamma and Lognormal distributions; compare goodness-of-fit.

**Step 1: Fit Gamma Distribution**

**Method of Moments:**
$$
\hat{\alpha} = \frac{\bar{X}^2}{s^2} = \frac{3500^2}{2500^2} = 1.96 \approx 2
$$
$$
\hat{\beta} = \frac{s^2}{\bar{X}} = \frac{2500^2}{3500} = 1786
$$

**Verification:**
- $E[X] = \hat{\alpha} \hat{\beta} = 2 \times 1786 = 3572 \approx 3500$ ✓
- $Var(X) = \hat{\alpha} \hat{\beta}^2 = 2 \times 1786^2 = 6,375,592$
- $\sigma = \sqrt{6,375,592} = 2525 \approx 2500$ ✓

**Step 2: Fit Lognormal Distribution**

**Method of Moments:**
$$
\hat{\mu} = \ln\left(\frac{\bar{X}^2}{\sqrt{\bar{X}^2 + s^2}}\right) = \ln\left(\frac{3500^2}{\sqrt{3500^2 + 2500^2}}\right) = \ln(2800) = 7.937
$$
$$
\hat{\sigma} = \sqrt{\ln\left(1 + \frac{s^2}{\bar{X}^2}\right)} = \sqrt{\ln\left(1 + \frac{2500^2}{3500^2}\right)} = \sqrt{\ln(1.51)} = 0.638
$$

**Verification:**
- $E[X] = e^{\hat{\mu} + \hat{\sigma}^2/2} = e^{7.937 + 0.204} = e^{8.141} = 3426 \approx 3500$ ✓
- $Median(X) = e^{\hat{\mu}} = e^{7.937} = 2800$ ✓ (matches observed median)

**Step 3: Compare Quantiles**

| Percentile | Observed | Gamma(2, 1786) | Lognormal(7.937, 0.638) |
|------------|----------|----------------|-------------------------|
| 50th | $2,800 | $2,678 | $2,800 |
| 95th | $9,000 | $8,950 | $9,200 |
| 99th | $15,000 | $13,500 | $15,800 |

**Step 4: Goodness-of-Fit Tests**

**Chi-Square Test (Gamma):**
- Divide data into 10 bins
- Calculate expected frequencies from Gamma(2, 1786)
- $\chi^2 = 8.3$, critical value (df=7, α=0.05) = 14.1
- **Conclusion:** Fail to reject; Gamma fits well

**K-S Test (Lognormal):**
- $D = 0.018$, critical value = 0.019
- **Conclusion:** Fail to reject; Lognormal fits well

**Step 5: Model Selection**

**AIC Comparison:**
- Gamma: AIC = 82,450
- Lognormal: AIC = 82,430

**Conclusion:** Lognormal has slightly lower AIC (better fit), and it matches the median exactly. **Choose Lognormal** for this data.

**Step 6: Use for Pricing**

**Calculate pure premium for a policy with $500 deductible:**

$$
E[X - 500 | X > 500] = \int_{500}^\infty (x - 500) f(x) dx / P(X > 500)
$$

Using Lognormal(7.937, 0.638):
- $P(X > 500) = 1 - \Phi\left(\frac{\ln(500) - 7.937}{0.638}\right) = 1 - \Phi(-2.18) = 0.985$
- $E[(X - 500)^+] \approx 3426 \times 0.985 - 500 \times 0.985 = 2882$

**Pure Premium (Severity Only):** $2,882

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1. **Frequency distributions (Poisson, Negative Binomial) model claim counts**; severity distributions (Gamma, Lognormal, Pareto) model claim amounts
2. **Poisson assumes mean = variance**; use Negative Binomial when variance > mean (overdispersion)
3. **Gamma is the standard severity distribution** in GLMs; Lognormal is an alternative with heavier tail
4. **Pareto models large losses** with polynomial tail decay; critical for reinsurance and extreme events
5. **Distribution choice matters** for tail quantiles (95th, 99th percentile) even if means are similar

### 10.2 When to Use This Knowledge
**Ideal For:**
- ✓ SOA Exam P / CAS Exam 1 preparation
- ✓ Pricing models (frequency-severity framework)
- ✓ Reserving (tail distributions for large loss reserves)
- ✓ Risk management (VaR, TVaR calculations)
- ✓ Reinsurance pricing (excess-of-loss)

**Not Ideal For:**
- ✗ Time-dependent processes (use stochastic processes)
- ✗ Dependent risks (use copulas)
- ✗ Non-stationary data (use time-varying parameters)

### 10.3 Critical Success Factors
1. **Master the Fundamentals:** Know PMF/PDF, mean, variance for each distribution
2. **Practice Parameter Estimation:** MLE, method of moments, both by hand and in software
3. **Develop Intuition:** Understand when to use each distribution (Poisson vs. NB, Gamma vs. Lognormal)
4. **Test Goodness-of-Fit:** Always validate distribution choice with K-S test, Q-Q plots
5. **Focus on Tails:** For insurance, tail behavior (95th, 99th percentile) is often more important than mean

### 10.4 Further Reading
- **Textbook:** "Loss Models: From Data to Decisions" by Klugman, Panjer & Willmot (Chapters 3-6)
- **Exam Prep:** Coaching Actuaries Adapt for Exam P (Distributions section)
- **Software:** R packages: `fitdistrplus`, `actuar`; Python: `scipy.stats`
- **Advanced:** "Modeling Extremal Events" by Embrechts, Klüppelberg & Mikosch (for Pareto/EVT)
- **Online:** SOA Exam P Sample Questions

---

## Appendix

### A. Glossary
- **Overdispersion:** Variance > Mean (violates Poisson assumption)
- **Heavy Tail:** Tail decays slower than exponential (e.g., Pareto)
- **Light Tail:** Tail decays exponentially or faster (e.g., Exponential, Gamma)
- **Memoryless Property:** $P(X > s+t | X > s) = P(X > t)$ (only Exponential has this)
- **Conjugate Prior:** Prior distribution that yields posterior in the same family

### B. Distribution Summary Table

| Distribution | Parameters | Mean | Variance | Use Case |
|--------------|-----------|------|----------|----------|
| **Poisson** | $\lambda$ | $\lambda$ | $\lambda$ | Frequency (standard) |
| **Negative Binomial** | $r, p$ or $\mu, \theta$ | $\mu$ | $\mu(1 + \mu/\theta)$ | Frequency (overdispersed) |
| **Exponential** | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | Waiting times |
| **Gamma** | $\alpha, \beta$ | $\alpha\beta$ | $\alpha\beta^2$ | Severity (standard) |
| **Lognormal** | $\mu, \sigma$ | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ | Severity (heavy tail) |
| **Pareto** | $\alpha, \theta$ | $\theta/(\alpha-1)$ | $\frac{\theta^2\alpha}{(\alpha-1)^2(\alpha-2)}$ | Large losses |
| **Weibull** | $\alpha, \beta$ | $\beta\Gamma(1+1/\alpha)$ | $\beta^2[\Gamma(1+2/\alpha)-\Gamma^2(1+1/\alpha)]$ | Survival, severity |

### C. R Code for Distribution Fitting

```r
library(fitdistrplus)

# Load claim data
claims <- c(1200, 3500, 800, 15000, 2200, ...)

# Fit Gamma
fit_gamma <- fitdist(claims, "gamma")
summary(fit_gamma)
plot(fit_gamma)

# Fit Lognormal
fit_lnorm <- fitdist(claims, "lnorm")
summary(fit_lnorm)
plot(fit_lnorm)

# Compare AIC
fit_gamma$aic
fit_lnorm$aic

# Goodness-of-fit tests
gofstat(list(fit_gamma, fit_lnorm), fitnames = c("Gamma", "Lognormal"))
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,300+*
