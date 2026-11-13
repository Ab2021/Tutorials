# Statistics & Probability — Senior Data Scientist Interview Prep
**Target Companies:** General Mills (India) · LSEG (London Stock Exchange Group, India)  
**Role Level:** Senior / Lead Data Scientist  
**Last Updated:** November 2026

---

## Table of Contents
1. [Probability Foundations](#1-probability-foundations)
2. [Probability Distributions](#2-probability-distributions)
3. [Central Limit Theorem & Sampling](#3-central-limit-theorem--sampling)
4. [Confidence Intervals](#4-confidence-intervals)
5. [Hypothesis Testing — t-test](#5-hypothesis-testing--t-test)
6. [Hypothesis Testing — Chi-Square](#6-hypothesis-testing--chi-square)
7. [ANOVA](#7-anova)
8. [A/B Testing Design & Analysis](#8-ab-testing-design--analysis)
9. [Multiple Testing Problem](#9-multiple-testing-problem)
10. [Bayesian Statistics](#10-bayesian-statistics)
11. [Bootstrapping](#11-bootstrapping)
12. [Type I & Type II Errors / Power Analysis](#12-type-i--type-ii-errors--power-analysis)
13. [Price Elasticity & Demand Modeling (General Mills)](#13-price-elasticity--demand-modeling-general-mills)
14. [Time Series Stationarity & ADF Test (LSEG)](#14-time-series-stationarity--adf-test-lseg)
15. [ARIMA Basics (LSEG)](#15-arima-basics-lseg)
16. [Value at Risk (VaR) (LSEG)](#16-value-at-risk-var-lseg)
17. [GARCH / Volatility Modeling (LSEG)](#17-garch--volatility-modeling-lseg)
18. [Maximum Likelihood Estimation](#18-maximum-likelihood-estimation)
19. [Sampling Methods](#19-sampling-methods)
20. [CUPED — Variance Reduction in A/B Tests](#20-cuped--variance-reduction-in-ab-tests)
21. [Sample Ratio Mismatch (SRM) Detection](#21-sample-ratio-mismatch-srm-detection)
22. [Cointegration & Pairs Trading (LSEG)](#22-cointegration--pairs-trading-lseg)

---

## Company Context Cheat Sheet

| Aspect | General Mills | LSEG |
|---|---|---|
| Domain | FMCG / CPG (food brands) | Financial infrastructure, indices, clearing |
| Key Stats Topics | A/B testing, demand modeling, price elasticity, marketing mix | VaR, GARCH, cointegration, stationarity |
| Data Size | Retail scanner / POS data, millions of SKU-weeks | Tick data, time-stamped order books, real-time feeds |
| Preferred Tools | `scipy`, `statsmodels`, `sklearn` | `statsmodels`, `arch`, `pandas`, `numpy` |
| Experiment Frame | Geo-matched market tests (brick-and-mortar) | Backtesting, risk model validation |

---

## 1. Probability Foundations

> **Difficulty: Easy**  
> **Companies: Both**

### Q1 — Bayes' Theorem (Classic Disease Test)

**Question:** A disease affects 1 in 1,000 people. A test is 99% accurate (sensitivity = specificity = 99%). If a random person tests positive, what is the probability they actually have the disease?

```python
# Bayes' Theorem: P(Disease | Positive) = P(Positive | Disease) * P(Disease) / P(Positive)

def bayes_theorem(p_disease, sensitivity, specificity):
    """
    p_disease   : prevalence, e.g. 0.001
    sensitivity : P(Positive | Disease)
    specificity : P(Negative | No Disease)
    """
    p_no_disease = 1 - p_disease
    p_false_positive = 1 - specificity  # P(Positive | No Disease)

    # Total probability of testing positive
    p_positive = (sensitivity * p_disease) + (p_false_positive * p_no_disease)

    # Posterior
    p_disease_given_positive = (sensitivity * p_disease) / p_positive
    return p_disease_given_positive

result = bayes_theorem(p_disease=0.001, sensitivity=0.99, specificity=0.99)
print(f"P(Disease | Positive Test) = {result:.4f}")  # ≈ 0.0902 (~9%)
```

**Key Insight:** Even with a highly accurate test, low prevalence means most positives are still false positives — the classic *base rate fallacy*. At LSEG, the same math applies to fraud/anomaly detection where positive events are rare.

---

### Q2 — Gambler's Ruin / Expected Value

**Question:** You roll a fair die repeatedly until you get a 6. What is the expected number of rolls?

```python
import numpy as np

# Geometric distribution: E[X] = 1/p where p = probability of success
p_six = 1 / 6
expected_rolls = 1 / p_six
print(f"Expected rolls: {expected_rolls:.1f}")  # 6.0

# Simulation verification
np.random.seed(42)
trials = 100_000
rolls = [0] * trials
for i in range(trials):
    count = 0
    while True:
        count += 1
        if np.random.randint(1, 7) == 6:
            break
    rolls[i] = count

print(f"Simulated mean: {np.mean(rolls):.3f}")  # Should be ~6.0

# P(takes more than 4 rolls) = P(no 6 in first 4) = (5/6)^4
p_more_than_4 = (5/6)**4
print(f"P(more than 4 rolls): {p_more_than_4:.4f}")  # ≈ 0.4823
```

---

## 2. Probability Distributions

> **Difficulty: Easy–Medium**  
> **Companies: Both**

### Q3 — When to Use Which Distribution

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ---- Binomial: discrete, fixed n trials, binary outcome ----
# Example: Out of 100 customers, each has 5% chance of buying.
# What's P(exactly 7 buy)?
n, p = 100, 0.05
print(f"P(X=7 | Binomial(100,0.05)) = {stats.binom.pmf(7, n, p):.4f}")
print(f"E[X] = {n*p}, Var = {n*p*(1-p)}")

# ---- Poisson: count events in interval, rare events ----
# Example: Average 3 orders per minute. P(exactly 5 orders in 1 min)?
lam = 3
print(f"P(X=5 | Poisson(3)) = {stats.poisson.pmf(5, lam):.4f}")

# ---- Normal: continuous, CLT-driven ----
mu, sigma = 0, 1
print(f"P(-1 < Z < 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")  # 68.27%

# ---- Log-normal: for asset prices (can't go negative) ----
# If X ~ Normal(mu, sigma), then e^X ~ Log-Normal
log_returns = np.random.normal(0.0005, 0.02, 252)  # daily log-returns
prices = 100 * np.exp(np.cumsum(log_returns))  # simulated price path
print(f"Simulated final price: {prices[-1]:.2f}")

# ---- Exponential: waiting time between Poisson events ----
# Rate 3 orders/min -> mean inter-arrival time = 1/3 min
rate = 3
print(f"P(wait < 0.5 min) = {stats.expon.cdf(0.5, scale=1/rate):.4f}")
```

**Interview Tip:** LSEG will ask why asset *prices* follow lognormal but *returns* follow normal. Answer: prices can't be negative, and % changes (log returns) aggregate via CLT.

---

## 3. Central Limit Theorem & Sampling

> **Difficulty: Easy**  
> **Companies: Both**

### Q4 — Demonstrate CLT with Simulation

**Question:** Explain the CLT and why it matters for A/B testing at scale.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(0)

# Underlying population: heavily right-skewed (exponential)
population = np.random.exponential(scale=5, size=100_000)

sample_sizes = [5, 30, 100, 500]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, n in zip(axes, sample_sizes):
    # Collect 2000 sample means
    sample_means = [np.mean(np.random.choice(population, n, replace=False))
                    for _ in range(2000)]
    ax.hist(sample_means, bins=40, density=True, alpha=0.7, color='steelblue')
    # Overlay normal PDF
    mu, sigma = np.mean(sample_means), np.std(sample_means)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    ax.set_title(f'n = {n}\nμ={mu:.2f}, σ={sigma:.2f}')

plt.suptitle('CLT: Sample Means Converge to Normal (Exponential Population)', y=1.02)
plt.tight_layout()
plt.savefig('clt_demo.png', dpi=100)
print("By n=30, distribution of sample means is approximately Normal — underpins z-tests & t-tests")
```

**Key CLT conditions:** Independence, finite variance, identically distributed (i.i.d.). At `n ≥ 30`, CLT usually kicks in.

---

## 4. Confidence Intervals

> **Difficulty: Easy–Medium**  
> **Companies: Both**

### Q5 — Compute CI from Scratch and with scipy

```python
import numpy as np
import scipy.stats as stats

np.random.seed(42)
# Scenario: General Mills measured average weekly units sold per store (n=50 stores)
sales = np.random.normal(loc=1500, scale=200, size=50)

n = len(sales)
x_bar = np.mean(sales)
s = np.std(sales, ddof=1)  # sample std, ddof=1 for unbiased

# 95% CI using t-distribution (small sample, unknown population sigma)
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
margin = t_crit * s / np.sqrt(n)

ci_lower = x_bar - margin
ci_upper = x_bar + margin
print(f"Sample mean: {x_bar:.1f}")
print(f"95% CI: ({ci_lower:.1f}, {ci_upper:.1f})")
print(f"t-critical value (df={n-1}): {t_crit:.4f}")

# Equivalent with scipy
ci = stats.t.interval(0.95, df=n-1, loc=x_bar, scale=s/np.sqrt(n))
print(f"scipy CI: ({ci[0]:.1f}, {ci[1]:.1f})")
```

**Common Mistake:** A 95% CI does NOT mean "95% probability the true mean is inside this interval." It means: if we repeated the experiment many times, 95% of computed intervals would contain the true mean.

---

## 5. Hypothesis Testing — t-test

> **Difficulty: Medium**  
> **Companies: Both (especially General Mills for A/B tests)**

### Q6 — Two-Sample t-test: Pricing A/B Test (General Mills)

**Scenario:** General Mills ran an A/B test on cereal pricing in matched stores. Control keeps $4.99, Treatment raises to $5.49. Did weekly sales volumes differ significantly?

```python
import numpy as np
import scipy.stats as stats

np.random.seed(0)

# Simulate store-level weekly sales
control_sales = np.random.normal(loc=1500, scale=220, size=80)   # 80 control stores
treatment_sales = np.random.normal(loc=1380, scale=230, size=80)  # 80 treatment stores

# Step 1: Descriptive stats
print("=== Descriptive Statistics ===")
print(f"Control:   mean={np.mean(control_sales):.1f}, std={np.std(control_sales, ddof=1):.1f}")
print(f"Treatment: mean={np.mean(treatment_sales):.1f}, std={np.std(treatment_sales, ddof=1):.1f}")

# Step 2: Check normality (Shapiro-Wilk)
_, p_norm_ctrl = stats.shapiro(control_sales[:50])  # Shapiro reliable for n<=50
_, p_norm_trt  = stats.shapiro(treatment_sales[:50])
print(f"\nShapiro-Wilk p (control): {p_norm_ctrl:.4f}")
print(f"Shapiro-Wilk p (treatment): {p_norm_trt:.4f}")

# Step 3: Levene's test for equal variances
stat_lev, p_lev = stats.levene(control_sales, treatment_sales)
print(f"Levene's test p-value: {p_lev:.4f}")
equal_var = p_lev > 0.05

# Step 4: Independent two-sample t-test
t_stat, p_value = stats.ttest_ind(control_sales, treatment_sales, equal_var=equal_var)
print(f"\n=== Two-Sample t-test ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Equal variances assumed: {equal_var}")

alpha = 0.05
if p_value < alpha:
    print(f"\nReject H₀: Sales differ significantly (p={p_value:.4f} < α={alpha})")
    print("→ Price increase had a statistically significant negative impact on sales volume.")
else:
    print(f"\nFail to reject H₀: No significant difference (p={p_value:.4f} ≥ α={alpha})")

# Effect size: Cohen's d
pooled_std = np.sqrt((np.std(control_sales, ddof=1)**2 + np.std(treatment_sales, ddof=1)**2) / 2)
cohens_d = (np.mean(control_sales) - np.mean(treatment_sales)) / pooled_std
print(f"\nCohen's d (effect size): {cohens_d:.3f}")
print("Interpretation: <0.2 small, 0.2-0.5 medium, >0.8 large")
```

---

### Q7 — Paired t-test: Before vs. After Promotion

**Scenario:** Same stores measured before and after a promotional campaign.

```python
import numpy as np
import scipy.stats as stats

np.random.seed(7)
n_stores = 40
before = np.random.normal(loc=1200, scale=150, size=n_stores)
after  = before + np.random.normal(loc=80, scale=100, size=n_stores)  # slight lift

# Paired t-test (each store is its own control)
t_stat, p_value = stats.ttest_rel(after, before)
diff = after - before

print(f"Mean lift per store: {np.mean(diff):.1f} units/week")
print(f"Paired t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
if p_value < 0.05:
    print("Promotion had a statistically significant positive effect.")
```

---

## 6. Hypothesis Testing — Chi-Square

> **Difficulty: Medium**  
> **Companies: Both**

### Q8 — Chi-Square Test of Independence: Purchase by Region

**Scenario:** Did product category preference differ across regions?

```python
import numpy as np
import scipy.stats as stats
import pandas as pd

# Contingency table: rows = region, cols = product category
observed = np.array([
    [120, 80,  50],   # North
    [ 90, 110, 60],   # South
    [150, 70,  30],   # West
])

regions = ['North', 'South', 'West']
categories = ['Cereal', 'Snacks', 'Beverages']
df_obs = pd.DataFrame(observed, index=regions, columns=categories)
print("Observed counts:")
print(df_obs)

chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-Square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("→ Product preference significantly differs across regions (reject H₀ independence).")
else:
    print("→ No significant difference in product preference across regions.")

print("\nExpected counts (under independence):")
print(pd.DataFrame(expected, index=regions, columns=categories).round(1))
```

---

## 7. ANOVA

> **Difficulty: Medium**  
> **Companies: General Mills**

### Q9 — One-Way ANOVA: Sales Across Marketing Channels

**Scenario:** General Mills used 4 marketing channels (TV, Digital, In-store, None). Do mean weekly sales differ?

```python
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

np.random.seed(42)
tv       = np.random.normal(1700, 200, 30)
digital  = np.random.normal(1600, 180, 30)
instore  = np.random.normal(1550, 220, 30)
none     = np.random.normal(1400, 170, 30)

# Step 1: One-way ANOVA (assumes normality, homoscedasticity, independence)
f_stat, p_value = stats.f_oneway(tv, digital, instore, none)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.6f}")

if p_value < 0.05:
    print("At least one channel mean is significantly different. Run post-hoc test.")
    
    # Step 2: Post-hoc — Tukey HSD to find WHICH pairs differ
    import numpy as np
    all_data = np.concatenate([tv, digital, instore, none])
    groups   = (['TV']*30 + ['Digital']*30 + ['InStore']*30 + ['None']*30)
    result = pairwise_tukeyhsd(all_data, groups, alpha=0.05)
    print(result)
```

**Key Assumption Checks:**
- **Normality:** Shapiro-Wilk per group
- **Homoscedasticity:** Bartlett's or Levene's test
- If assumptions violated → Kruskal-Wallis (non-parametric alternative)

---

## 8. A/B Testing Design & Analysis

> **Difficulty: Hard**  
> **Companies: Both**

### Q10 — Full A/B Test Pipeline: Sample Size → Analysis → Decision

```python
import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

# ============================================================
# STEP 1: Pre-experiment — Sample Size Calculation
# ============================================================
# Scenario: General Mills website conversion rate is 3% (control).
# We want to detect a lift to 3.5% (MDE = 0.5pp).
p_control  = 0.03
p_treatment = 0.035
alpha = 0.05
power = 0.80

# Effect size for proportions (Cohen's h)
effect_size = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
print(f"Cohen's h (effect size): {effect_size:.4f}")

# Sample size per group
analysis = NormalIndPower()
n_per_group = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1)
n_per_group = int(np.ceil(n_per_group))
print(f"Required n per group: {n_per_group}")

# ============================================================
# STEP 2: Run experiment — simulate outcome
# ============================================================
np.random.seed(1)
n = n_per_group
control_conversions   = np.random.binomial(1, p_control, n)
treatment_conversions = np.random.binomial(1, p_treatment, n)

obs_ctrl = control_conversions.sum()
obs_trt  = treatment_conversions.sum()
print(f"\nControl conversions:   {obs_ctrl}/{n} = {obs_ctrl/n:.4f}")
print(f"Treatment conversions: {obs_trt}/{n} = {obs_trt/n:.4f}")

# ============================================================
# STEP 3: Two-proportion z-test
# ============================================================
count = np.array([obs_trt, obs_ctrl])
nobs  = np.array([n, n])
z_stat, p_val = proportions_ztest(count, nobs, alternative='larger')  # one-sided: treatment > control
print(f"\nz-statistic: {z_stat:.4f}, p-value: {p_val:.6f}")

if p_val < alpha:
    print("→ Statistically significant lift detected. Ship the treatment.")
else:
    print("→ No significant lift. Do not ship.")

# ============================================================
# STEP 4: Business impact calculation
# ============================================================
monthly_visitors = 500_000
lift_rate = obs_trt/n - obs_ctrl/n
incremental_conversions = lift_rate * monthly_visitors
avg_order_value = 45  # USD
revenue_impact = incremental_conversions * avg_order_value
print(f"\nEstimated monthly incremental revenue: ${revenue_impact:,.0f}")
```

---

### Q11 — Peeking Problem / Sequential Testing

**Question:** A PM wants to check results daily and stop when p < 0.05. Why is this wrong?

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(99)

def simulate_peeking(n_total, p_ctrl, p_trt, alpha=0.05, n_simulations=1000):
    """What fraction of simulations hit p<alpha under the null (p_ctrl==p_trt)?"""
    from scipy.stats import chi2_contingency
    false_positives = 0
    for _ in range(n_simulations):
        ctrl_data = np.random.binomial(1, p_ctrl, n_total)
        trt_data  = np.random.binomial(1, p_trt,  n_total)
        # Peek at every 100 observations
        for t in range(100, n_total+1, 100):
            c0, c1 = ctrl_data[:t].sum(), trt_data[:t].sum()
            contingency = [[c0, t-c0], [c1, t-c1]]
            _, p, _, _ = chi2_contingency(contingency)
            if p < alpha:
                false_positives += 1
                break  # stopped early
    return false_positives / n_simulations

# Under the NULL (no real difference):
fpr_no_peek = 0.05  # nominal
fpr_peek    = simulate_peeking(5000, 0.05, 0.05, n_simulations=500)
print(f"Nominal false positive rate: {fpr_no_peek:.2f}")
print(f"False positive rate WITH peeking: {fpr_peek:.3f}")
print("→ Peeking inflates Type I error well beyond nominal alpha!")

# Fix: Group Sequential Testing or Always-Valid p-values (mSPRT)
print("\nFixes: O'Brien-Fleming boundaries, Pocock boundaries, or Bayesian testing.")
```

---

## 9. Multiple Testing Problem

> **Difficulty: Medium**  
> **Companies: Both**

### Q12 — Bonferroni & Benjamini-Hochberg Corrections

```python
import numpy as np
from statsmodels.stats.multitest import multipletests

np.random.seed(0)

# Simulate 20 hypothesis tests (most under H₀, a few real effects)
n_tests = 20
p_values = np.concatenate([
    np.random.uniform(0, 1, 15),     # null hypotheses
    np.random.uniform(0, 0.01, 5),   # true positives
])
np.random.shuffle(p_values)

alpha = 0.05

# No correction
print("=== No Correction ===")
print(f"Rejected: {(p_values < alpha).sum()} out of {n_tests}")
print(f"Expected false positives under H₀: {15 * alpha:.1f}")

# Bonferroni (controls Family-Wise Error Rate — very conservative)
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
print(f"\n=== Bonferroni Correction ===")
print(f"Adjusted alpha: {alpha/n_tests:.4f}")
print(f"Rejected: {reject_bonf.sum()}")

# Benjamini-Hochberg (controls False Discovery Rate — less conservative)
reject_bh, pvals_bh, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
print(f"\n=== Benjamini-Hochberg (FDR) ===")
print(f"Rejected: {reject_bh.sum()}")

print("\nRule of thumb:")
print("  Bonferroni → when ANY false positive is unacceptable (safety-critical)")
print("  B-H FDR   → when exploring multiple metrics in an A/B test")
```

---

## 10. Bayesian Statistics

> **Difficulty: Hard**  
> **Companies: Both**

### Q13 — Bayesian A/B Testing with Beta-Binomial

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Beta-Binomial conjugate model for conversion rates
# Prior: Beta(alpha_prior, beta_prior) — weakly informative
alpha_prior, beta_prior = 1, 1  # Uniform prior (no strong prior)

# Observed data
ctrl_conversions, ctrl_total = 300, 10000   # 3% control rate
trt_conversions,  trt_total  = 350, 10000   # 3.5% treatment rate

# Posterior: Beta(alpha_prior + successes, beta_prior + failures)
alpha_ctrl = alpha_prior + ctrl_conversions
beta_ctrl  = beta_prior  + (ctrl_total - ctrl_conversions)

alpha_trt  = alpha_prior + trt_conversions
beta_trt   = beta_prior  + (trt_total  - trt_conversions)

ctrl_posterior = stats.beta(alpha_ctrl, beta_ctrl)
trt_posterior  = stats.beta(alpha_trt,  beta_trt)

# P(treatment > control) via Monte Carlo
n_samples = 500_000
ctrl_samples = ctrl_posterior.rvs(n_samples)
trt_samples  = trt_posterior.rvs(n_samples)
prob_trt_better = (trt_samples > ctrl_samples).mean()
print(f"P(Treatment > Control) = {prob_trt_better:.4f}")

# Expected Relative Lift
relative_lift = ((trt_samples - ctrl_samples) / ctrl_samples)
print(f"Expected relative lift: {relative_lift.mean()*100:.2f}%")
print(f"95% Credible Interval for lift: ({np.percentile(relative_lift,2.5)*100:.2f}%, "
      f"{np.percentile(relative_lift,97.5)*100:.2f}%)")

# Posterior predictive plots
x = np.linspace(0.025, 0.045, 300)
plt.figure(figsize=(8, 4))
plt.plot(x, ctrl_posterior.pdf(x), label='Control posterior', color='steelblue')
plt.plot(x, trt_posterior.pdf(x), label='Treatment posterior', color='darkorange')
plt.axvline(0.03, ls='--', color='steelblue', alpha=0.5)
plt.axvline(0.035, ls='--', color='darkorange', alpha=0.5)
plt.legend()
plt.title(f'Bayesian A/B Test | P(Trt > Ctrl) = {prob_trt_better:.3f}')
plt.savefig('bayesian_ab.png', dpi=100)
print("\nAdvantage over frequentist: No fixed sample size needed; natural stopping criterion.")
```

---

## 11. Bootstrapping

> **Difficulty: Medium**  
> **Companies: Both**

### Q14 — Bootstrap CI for Median Sales Lift

```python
import numpy as np

np.random.seed(42)
# Observed lift per store after a promotion (not normally distributed)
lifts = np.array([50, 120, -30, 200, 80, 45, 300, -10, 90, 150,
                   60, 170, -20, 85, 110, 95, 230, 55, 75, 130])

observed_median = np.median(lifts)
print(f"Observed median lift: {observed_median:.1f} units")

# Bootstrap
n_boot = 10_000
boot_medians = np.array([
    np.median(np.random.choice(lifts, size=len(lifts), replace=True))
    for _ in range(n_boot)
])

ci_lower = np.percentile(boot_medians, 2.5)
ci_upper = np.percentile(boot_medians, 97.5)
print(f"95% Bootstrap CI for median: ({ci_lower:.1f}, {ci_upper:.1f})")
print(f"Bootstrap std error: {boot_medians.std():.2f}")

# If CI excludes 0 → significant positive lift
if ci_lower > 0:
    print("→ Median lift is significantly positive (CI excludes 0).")
```

**When to use bootstrapping:** Small samples, non-normal data, complex statistics (median, ratio, percentile) where analytical CIs don't exist.

---

## 12. Type I & Type II Errors / Power Analysis

> **Difficulty: Medium**  
> **Companies: Both**

### Q15 — Power Curve: Trade-off Between Alpha and Sample Size

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
effect_size = 0.3  # medium effect (Cohen's d)

# --- Effect of sample size on power (alpha=0.05) ---
n_range = np.arange(10, 500, 5)
power_vals = [analysis.power(effect_size=effect_size, nobs1=n, alpha=0.05, ratio=1)
              for n in n_range]

# --- Relationship between alpha and power (n=100) ---
alpha_range = np.linspace(0.001, 0.2, 100)
power_alpha  = [analysis.power(effect_size=effect_size, nobs1=100, alpha=a, ratio=1)
                for a in alpha_range]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(n_range, power_vals, 'b-')
axes[0].axhline(0.80, color='r', ls='--', label='Target power=0.80')
axes[0].set(xlabel='n per group', ylabel='Power', title='Power vs Sample Size')
axes[0].legend()

axes[1].plot(alpha_range, power_alpha, 'g-')
axes[1].axvline(0.05, color='r', ls='--', label='α=0.05')
axes[1].set(xlabel='α (Type I Error)', ylabel='Power (1 - β)', title='Power vs Alpha Trade-off')
axes[1].legend()
plt.tight_layout()
plt.savefig('power_analysis.png', dpi=100)

# Key numbers
n_needed = analysis.solve_power(effect_size=0.3, alpha=0.05, power=0.80, ratio=1)
print(f"n per group needed for 80% power (d=0.3, α=0.05): {int(np.ceil(n_needed))}")
print("\nType I Error (α): False Positive — reject H₀ when it's true  → risk of shipping useless feature")
print("Type II Error (β): False Negative — fail to reject false H₀  → risk of killing a good feature")
print("Power = 1 - β: Probability of detecting a real effect")
```

---

## 13. Price Elasticity & Demand Modeling (General Mills)

> **Difficulty: Hard**  
> **Companies: General Mills**

### Q16 — Log-Log Regression for Price Elasticity

**Scenario:** Estimate price elasticity of demand for a cereal brand using scanner data.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
n_weeks = 200

# Simulate scanner data (weekly, store-level)
price        = np.random.uniform(3.5, 6.5, n_weeks)      # shelf price
competitor_p = np.random.uniform(3.0, 6.0, n_weeks)      # competitor price
promo_flag   = np.random.binomial(1, 0.2, n_weeks)        # 20% weeks on promo
seasonality  = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

# Demand: log-log DGP with elasticity = -1.4
true_elasticity = -1.4
log_demand = (8.5
              + true_elasticity * np.log(price)
              + 0.5 * np.log(competitor_p)
              + 0.3 * promo_flag
              + np.log(seasonality)
              + np.random.normal(0, 0.1, n_weeks))
demand = np.exp(log_demand)

df = pd.DataFrame({'log_demand': np.log(demand),
                   'log_price':  np.log(price),
                   'log_comp_p': np.log(competitor_p),
                   'promo':      promo_flag})

# OLS regression
X = sm.add_constant(df[['log_price', 'log_comp_p', 'promo']])
model = sm.OLS(df['log_demand'], X).fit()
print(model.summary())

own_elasticity  = model.params['log_price']
cross_elasticity = model.params['log_comp_p']
print(f"\nOwn-price elasticity:  {own_elasticity:.3f}")
print(f"Cross-price elasticity: {cross_elasticity:.3f}")
print(f"True elasticity: {true_elasticity}")

# Break-even analysis: Price increase of 10%
price_increase_pct = 0.10
expected_volume_drop_pct = own_elasticity * price_increase_pct
breakeven_volume_drop = price_increase_pct / (1 + price_increase_pct)
print(f"\n=== Break-Even Analysis (10% price hike) ===")
print(f"Expected volume drop:    {expected_volume_drop_pct*100:.1f}%")
print(f"Break-even volume drop:  {breakeven_volume_drop*100:.1f}%")
if abs(expected_volume_drop_pct) > breakeven_volume_drop:
    print("→ Revenue will DECREASE. Do NOT raise price.")
else:
    print("→ Revenue will INCREASE. Price hike is justified.")
```

---

## 14. Time Series Stationarity & ADF Test (LSEG)

> **Difficulty: Hard**  
> **Companies: LSEG**

### Q17 — Augmented Dickey-Fuller Test on Stock Returns

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500

# Non-stationary: Random walk (prices)
prices = 100 + np.cumsum(np.random.normal(0, 1, n))

# Stationary: Log returns
log_returns = np.diff(np.log(prices))

def adf_test(series, name):
    result = adfuller(series, autolag='AIC')
    print(f"\n=== ADF Test: {name} ===")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values: {result[4]}")
    if result[1] < 0.05:
        print("→ STATIONARY (reject unit root H₀)")
    else:
        print("→ NON-STATIONARY (fail to reject H₀)")

adf_test(prices, "Raw Price (Random Walk)")
adf_test(log_returns, "Log Returns")

# KPSS test (complement to ADF — different null hypothesis)
# H₀ in KPSS: series IS stationary
def kpss_test(series, name):
    stat, p_val, n_lags, crit = kpss(series, regression='c', nlags='auto')
    print(f"\n=== KPSS Test: {name} ===")
    print(f"KPSS Statistic: {stat:.4f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("→ NON-STATIONARY (reject stationarity H₀)")
    else:
        print("→ STATIONARY (fail to reject H₀)")

kpss_test(prices, "Raw Price")
kpss_test(log_returns, "Log Returns")

print("\nRule: Use ADF + KPSS together. Agreement → confident conclusion.")
print("If ADF says stationary and KPSS says non-stationary → may be trend-stationary.")
```

---

## 15. ARIMA Basics (LSEG)

> **Difficulty: Hard**  
> **Companies: LSEG**

### Q18 — Fit ARIMA on Financial Returns

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)
# Simulate AR(1) process: returns_t = 0.3 * returns_{t-1} + noise
n = 500
returns = np.zeros(n)
for t in range(1, n):
    returns[t] = 0.3 * returns[t-1] + np.random.normal(0, 1)

# Step 1: Confirm stationarity
adf = adfuller(returns)
print(f"ADF p-value: {adf[1]:.4f} → {'Stationary' if adf[1] < 0.05 else 'Non-stationary'}")

# Step 2: ACF / PACF to pick p, q (d=0 since already stationary)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(returns, ax=axes[0], lags=20)
plot_pacf(returns, ax=axes[1], lags=20)
plt.savefig('acf_pacf.png', dpi=100)
# PACF cuts off at lag 1 → AR(1) ← PACF is for AR order
# ACF decays geometrically → confirms AR pattern

# Step 3: Fit ARIMA(1,0,0)
model = ARIMA(returns, order=(1, 0, 0))
result = model.fit()
print(result.summary())

# Step 4: Forecast 10 steps ahead
forecast = result.forecast(steps=10)
print(f"\nForecast next 10 returns: {forecast.round(4).values}")

# Step 5: Model selection via AIC/BIC
print("\n=== Model Comparison (AIC) ===")
for order in [(0,0,0),(1,0,0),(0,0,1),(1,0,1),(2,0,0)]:
    try:
        m = ARIMA(returns, order=order).fit()
        print(f"ARIMA{order}: AIC={m.aic:.2f}, BIC={m.bic:.2f}")
    except:
        pass
```

---

## 16. Value at Risk (VaR) (LSEG)

> **Difficulty: Hard**  
> **Companies: LSEG**

### Q19 — Three Methods for VaR Calculation

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

np.random.seed(42)

# Simulate 2 years of daily portfolio log-returns (fat-tailed)
n_days = 504
returns = np.random.standard_t(df=5, size=n_days) * 0.01  # t(5) for fat tails
portfolio_value = 10_000_000  # $10M

confidence = 0.99  # 99% VaR

# ============================================================
# Method 1: Historical Simulation (non-parametric)
# ============================================================
var_hist = -np.percentile(returns, (1 - confidence) * 100)
print(f"Historical VaR (99%): {var_hist:.4f} = ${var_hist * portfolio_value:,.0f}")

# ============================================================
# Method 2: Parametric (Variance-Covariance) — Assumes Normal
# ============================================================
mu, sigma = returns.mean(), returns.std()
z_score = stats.norm.ppf(1 - confidence)
var_param = -(mu + z_score * sigma)
print(f"Parametric VaR (99%, Normal): {var_param:.4f} = ${var_param * portfolio_value:,.0f}")

# ============================================================
# Method 3: Monte Carlo
# ============================================================
n_sims = 100_000
mc_returns = np.random.normal(mu, sigma, n_sims)
var_mc = -np.percentile(mc_returns, (1 - confidence) * 100)
print(f"Monte Carlo VaR (99%, Normal): {var_mc:.4f} = ${var_mc * portfolio_value:,.0f}")

# ============================================================
# Expected Shortfall (CVaR) — preferred by Basel III/IV
# ============================================================
threshold = np.percentile(returns, (1 - confidence) * 100)
es = -returns[returns <= threshold].mean()
print(f"\nExpected Shortfall / CVaR (99%): {es:.4f} = ${es * portfolio_value:,.0f}")
print("ES measures the average loss IN THE TAIL beyond VaR — a coherent risk measure.")

# Backtesting: Kupiec POF test
exceptions = (returns < -var_hist).sum()
expected_exc = n_days * (1 - confidence)
print(f"\nBacktest: {exceptions} exceptions vs expected {expected_exc:.1f}")
print(f"Exception rate: {exceptions/n_days:.4f} (expected: {1-confidence:.4f})")
```

---

## 17. GARCH / Volatility Modeling (LSEG)

> **Difficulty: Hard**  
> **Companies: LSEG**

### Q20 — GARCH(1,1) for Volatility Clustering

```python
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate GBM returns (use real data in practice)
n = 1000
returns_pct = np.random.normal(0, 1, n)  # scale to percentage returns

# Fit GARCH(1,1)
# r_t = mu + eps_t
# eps_t = sigma_t * z_t,  z_t ~ N(0,1)
# sigma²_t = omega + alpha * eps²_{t-1} + beta * sigma²_{t-1}
model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
result = model.fit(disp='off')
print(result.summary())

params = result.params
print(f"\nomega={params['omega']:.6f}, alpha={params['alpha[1]']:.4f}, beta={params['beta[1]']:.4f}")
print(f"Persistence (alpha+beta): {params['alpha[1]'] + params['beta[1]']:.4f}")
print("If alpha+beta < 1 → covariance-stationary (volatility mean-reverts)")

# Conditional volatility plot
cond_vol = result.conditional_volatility
print(f"\nAnnualized conditional volatility (last estimate): {cond_vol[-1] * np.sqrt(252):.2f}%")

# 10-day volatility forecast
forecasts = result.forecast(horizon=10)
vol_forecast = np.sqrt(forecasts.variance.iloc[-1].values)
print(f"Volatility forecast next 10 days: {vol_forecast.round(4)}")
print("\nGARCH captures volatility CLUSTERING: big moves follow big moves.")
print("Extension: GJR-GARCH / EGARCH for the leverage effect (bad news → more volatility).")
```

---

## 18. Maximum Likelihood Estimation

> **Difficulty: Medium**  
> **Companies: LSEG**

### Q21 — MLE for Exponential Distribution (Derive + Implement)

```python
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats

np.random.seed(42)
# Trade inter-arrival times follow Exponential(lambda=3 trades/min)
true_lambda = 3.0
data = np.random.exponential(scale=1/true_lambda, size=200)

# Closed-form MLE: lambda_hat = 1 / x_bar
lambda_mle = 1 / np.mean(data)
print(f"True lambda: {true_lambda:.4f}")
print(f"Analytical MLE lambda: {lambda_mle:.4f}")

# Numerical MLE (gradient-based) — works for any distribution
def neg_log_likelihood(params, data):
    lam = params[0]
    if lam <= 0:
        return np.inf
    # log-likelihood: n*log(lambda) - lambda*sum(x)
    ll = len(data) * np.log(lam) - lam * np.sum(data)
    return -ll  # minimize negative log-likelihood

result = minimize(neg_log_likelihood, x0=[1.0], args=(data,), method='Nelder-Mead')
lambda_numerical = result.x[0]
print(f"Numerical MLE lambda:  {lambda_numerical:.4f}")

# Confidence interval for lambda via Fisher Information
# Fisher info I(lambda) = n / lambda^2 -> SE = lambda / sqrt(n)
se = lambda_mle / np.sqrt(len(data))
ci = (lambda_mle - 1.96*se, lambda_mle + 1.96*se)
print(f"95% CI for lambda: ({ci[0]:.4f}, {ci[1]:.4f})")
```

---

## 19. Sampling Methods

> **Difficulty: Medium**  
> **Companies: Both**

### Q22 — Stratified Sampling for SKU Survey

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Population of 10,000 customers across 4 segments
segments = {'Premium': 1000, 'Regular': 5000, 'Budget': 3000, 'Trial': 1000}
total = sum(segments.values())

# Simple random sampling (may under-represent small segments)
sample_size = 500

# Proportional stratified sampling
strat_samples = {seg: int(np.round(count/total * sample_size))
                 for seg, count in segments.items()}
print("=== Proportional Stratified Sample ===")
for seg, n in strat_samples.items():
    print(f"{seg}: {n} samples (from {segments[seg]} population)")

# Why stratified over SRS?
print("\nBenefit: Guarantees proportional representation of each segment.")
print("Useful for General Mills when surveying across store tiers (premium, mass, discount).")

# Systematic sampling example
N = 10000  # population size
k = N // sample_size  # sampling interval = 20
start = np.random.randint(0, k)
systematic_indices = list(range(start, N, k))[:sample_size]
print(f"\nSystematic sampling: Every {k}th unit, starting at {start}")
print(f"Sample indices (first 5): {systematic_indices[:5]}")
```

---

## 20. CUPED — Variance Reduction in A/B Tests

> **Difficulty: Hard**  
> **Companies: General Mills**

### Q23 — CUPED Implementation

**CUPED** (Controlled-experiment Using Pre-Experiment Data) reduces variance by using pre-experiment covariate data.

```python
import numpy as np
import scipy.stats as stats

np.random.seed(42)
n = 2000

# Pre-experiment metric (covariate): e.g. prior week's sales per store
pre_sales = np.random.normal(1000, 150, n)

# Post-experiment metric (correlated with pre, plus treatment effect)
treatment = np.concatenate([np.ones(n//2), np.zeros(n//2)])
np.random.shuffle(treatment)

# True treatment effect = +50 units
post_sales = (0.7 * pre_sales                         # correlation with pre
              + 300                                    # intercept
              + 50 * treatment                         # true effect
              + np.random.normal(0, 100, n))           # noise

# Standard t-test (ignoring pre-data)
ctrl_post = post_sales[treatment == 0]
trt_post  = post_sales[treatment == 1]
t_std, p_std = stats.ttest_ind(trt_post, ctrl_post)
print(f"Standard t-test: t={t_std:.4f}, p={p_std:.6f}")
print(f"Variance of post-sales: {post_sales.var():.1f}")

# CUPED adjustment
# theta = Cov(post, pre) / Var(pre)
theta = np.cov(post_sales, pre_sales)[0,1] / np.var(pre_sales)
print(f"\nCUPED theta (regression coefficient): {theta:.4f}")

post_cuped = post_sales - theta * (pre_sales - np.mean(pre_sales))
print(f"CUPED adjusted variance: {post_cuped.var():.1f}")
print(f"Variance reduction: {(1 - post_cuped.var()/post_sales.var())*100:.1f}%")

ctrl_cuped = post_cuped[treatment == 0]
trt_cuped  = post_cuped[treatment == 1]
t_cuped, p_cuped = stats.ttest_ind(trt_cuped, ctrl_cuped)
print(f"\nCUPED t-test: t={t_cuped:.4f}, p={p_cuped:.6f}")
print(f"CUPED more powerful: {'YES' if p_cuped < p_std else 'NO'}")
print("→ Smaller p-value with same data = reduced required sample size (up to 30-50%)")
```

---

## 21. Sample Ratio Mismatch (SRM) Detection

> **Difficulty: Medium**  
> **Companies: General Mills**

### Q24 — Chi-Square Goodness-of-Fit for SRM

```python
import numpy as np
import scipy.stats as stats

# Expected 50-50 split, but observed imbalance (bug in assignment code)
expected_split = 0.50
n_total = 15000
observed_control   = 7900
observed_treatment = 7100

expected_control   = n_total * expected_split
expected_treatment = n_total * expected_split

# Chi-Square goodness-of-fit
chi2_stat, p_value = stats.chisquare(
    f_obs=[observed_control, observed_treatment],
    f_exp=[expected_control, expected_treatment]
)

print(f"Expected: {expected_control:.0f} / {expected_treatment:.0f}")
print(f"Observed: {observed_control} / {observed_treatment}")
print(f"Chi-square: {chi2_stat:.4f}, p-value: {p_value:.6f}")

if p_value < 0.01:
    print("\n⚠ SAMPLE RATIO MISMATCH DETECTED (p < 0.01)!")
    print("→ STOP the analysis. Debug the assignment mechanism.")
    print("Common causes: bot traffic, logging drops, page crashes in one variant.")
else:
    print("\n✓ No SRM detected. Split looks balanced.")
```

---

## 22. Cointegration & Pairs Trading (LSEG)

> **Difficulty: Hard**  
> **Companies: LSEG**

### Q25 — Engle-Granger Cointegration Test

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

np.random.seed(42)
n = 500

# Simulate two cointegrated price series (share common stochastic trend)
common_trend = np.cumsum(np.random.normal(0, 1, n))
price_A = common_trend + np.random.normal(0, 0.5, n)
price_B = 0.8 * common_trend + 2 + np.random.normal(0, 0.5, n)

# Step 1: Both series are individually non-stationary
adf_A = adfuller(price_A)
adf_B = adfuller(price_B)
print(f"ADF p-value: price_A={adf_A[1]:.4f}, price_B={adf_B[1]:.4f}")
print("Both > 0.05 → non-stationary individually")

# Step 2: Test cointegration
coint_stat, p_coint, crit_values = coint(price_A, price_B)
print(f"\nCointegration test statistic: {coint_stat:.4f}")
print(f"p-value: {p_coint:.6f}")
if p_coint < 0.05:
    print("→ Cointegrated! A linear combination is stationary.")
    print("   Useful for LSEG pairs trading / relative value strategies.")

# Step 3: Find the spread (hedge ratio via OLS)
X = sm.add_constant(price_B)
result = sm.OLS(price_A, X).fit()
beta = result.params['x1']
spread = price_A - beta * price_B
print(f"\nHedge ratio (beta): {beta:.4f}")
print(f"Spread mean: {spread.mean():.4f}")
print(f"Spread ADF p-value: {adfuller(spread)[1]:.6f}")
print("→ Spread is stationary (mean-reverting) → signal for pairs trade entry/exit")
```

---

## Quick Reference — Interview Answer Frameworks

### A/B Test End-to-End Checklist
```
1. Define H₀ and H₁ clearly
2. Choose metric (success, guardrail, secondary)
3. Power analysis → n per group
4. Run experiment (randomize, isolate groups)
5. Check SRM (chi-square on allocation)
6. Analyze with correct test (z-test for proportions, t-test for means)
7. Apply multiple testing correction if needed
8. Report effect size, CI, and business impact
9. Document guardrail metric outcomes
10. Ship / No-ship decision with business stakeholder
```

### Common Mistakes to Avoid
| Mistake | Correct Approach |
|---|---|
| Peeking at results before sample size reached | Fixed-horizon or sequential testing |
| Testing 20 metrics at α=0.05 without correction | Benjamini-Hochberg FDR correction |
| Using paired t-test for independent groups | Match test to data structure |
| Claiming p=0.049 is "significant" without effect size | Report Cohen's d + CI |
| Running ARIMA on non-stationary series | Difference first, verify ADF |
| Using VaR alone for risk reporting | Add Expected Shortfall (CVaR) |
| Assuming normality without checking | Shapiro-Wilk / Q-Q plots first |

### Key Formulas Cheat Sheet
```
Cohen's d:          d = (μ₁ - μ₂) / σ_pooled
Effect size (prop): h = 2 * (arcsin(√p₁) - arcsin(√p₂))
Bootstrap CI:       [percentile(2.5%), percentile(97.5%)] of boot distribution
Price elasticity:   ε = (ΔQ/Q) / (ΔP/P) = β in log-log model
VaR (parametric):   VaR = -(μ + z_α * σ) * Portfolio_Value
CUPED:              Y_adj = Y_post - θ(Y_pre - E[Y_pre])
                    θ = Cov(Y_post, Y_pre) / Var(Y_pre)
GARCH(1,1):         σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
Bayes:              P(H|D) = P(D|H)·P(H) / P(D)
```

---

## Difficulty Summary

| # | Topic | Difficulty | Company |
|---|---|---|---|
| Q1 | Bayes' Theorem | Easy | Both |
| Q2 | Geometric Distribution | Easy | Both |
| Q3 | Distribution Selection | Easy | Both |
| Q4 | CLT Simulation | Easy | Both |
| Q5 | Confidence Intervals | Easy | Both |
| Q6 | Two-Sample t-test (Pricing A/B) | Medium | General Mills |
| Q7 | Paired t-test | Medium | General Mills |
| Q8 | Chi-Square Independence | Medium | Both |
| Q9 | One-Way ANOVA | Medium | General Mills |
| Q10 | Full A/B Test Pipeline | Hard | Both |
| Q11 | Peeking Problem | Medium | Both |
| Q12 | Multiple Testing Correction | Medium | Both |
| Q13 | Bayesian A/B Testing | Hard | Both |
| Q14 | Bootstrap CI | Medium | Both |
| Q15 | Power Analysis | Medium | Both |
| Q16 | Price Elasticity (Log-Log) | Hard | General Mills |
| Q17 | ADF / KPSS Stationarity | Hard | LSEG |
| Q18 | ARIMA Fitting | Hard | LSEG |
| Q19 | Value at Risk (3 methods) | Hard | LSEG |
| Q20 | GARCH(1,1) | Hard | LSEG |
| Q21 | MLE (Exponential) | Medium | LSEG |
| Q22 | Stratified Sampling | Medium | Both |
| Q23 | CUPED Variance Reduction | Hard | General Mills |
| Q24 | Sample Ratio Mismatch | Medium | General Mills |
| Q25 | Cointegration / Pairs Trading | Hard | LSEG |

---

*25 questions total · Python implementations using `scipy.stats`, `statsmodels`, `arch` · Business contexts: FMCG (General Mills) + Financial infrastructure (LSEG)*
