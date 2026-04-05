# eBay DS/ML — Statistics & A/B Testing Deep Dive

## 📊 Part 1: Core Statistics Concepts

### 1.1 Probability Distributions You Must Know

| Distribution | When to Use | eBay Example |
|---|---|---|
| **Normal** | Continuous data, large samples (CLT) | Distribution of average order values |
| **Binomial** | Fixed trials, binary outcome | Did user convert? (yes/no) over N visits |
| **Poisson** | Count of rare events in fixed interval | Number of fraudulent transactions per hour |
| **Exponential** | Time between events | Time between purchases for a user |
| **Beta** | Probability of a probability (Bayesian) | Prior belief about CTR of a new listing |

### 1.2 Hypothesis Testing Framework

```
Step 1: Define H₀ (null) and H₁ (alternative)
Step 2: Choose significance level α (usually 0.05)
Step 3: Select test statistic
Step 4: Calculate p-value
Step 5: Compare p-value to α → reject/fail to reject H₀
```

**Key Tests & When to Use:**

| Test | Use When | Assumptions |
|---|---|---|
| **Z-test (proportions)** | Comparing conversion rates | Large sample, binary outcome |
| **Two-sample t-test** | Comparing means (revenue/AOV) | Continuous, normal-ish, equal variance |
| **Welch's t-test** | Comparing means, unequal variance | Continuous, doesn't need equal variance |
| **Chi-squared test** | Comparing categorical distributions | Expected frequency ≥ 5 |
| **Mann-Whitney U** | Non-parametric, comparing medians | No normality assumption needed |
| **Paired t-test** | Before/after on same units | Paired observations |
| **Fisher's Exact Test** | Small sample proportions | When chi-squared assumptions fail |

### 1.3 Key Formulas

**Confidence Interval for a proportion:**
```
p̂ ± z* × √(p̂(1-p̂)/n)

where z* = 1.96 for 95% CI
```

**Sample Size for A/B Test (proportions):**
```
n = (z_α/2 + z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²

where:
  z_α/2 = 1.96 (for α=0.05)
  z_β   = 0.84 (for power=0.80)
  p₁    = baseline conversion rate
  p₂    = expected conversion rate after treatment
```

**Z-test for Two Proportions:**
```
z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))

where p̂ = pooled proportion = (x₁+x₂)/(n₁+n₂)
```

---

## 🧪 Part 2: A/B Testing Deep Dive

### 2.1 The Complete A/B Test Design Checklist

```
1. HYPOTHESIS
   □ What feature/change are we testing?
   □ What is our hypothesis (directional or non-directional)?

2. METRICS
   □ Primary metric (the decision metric)
   □ Secondary metrics (supporting evidence)
   □ Guardrail metrics (must NOT degrade)

3. RANDOMIZATION
   □ Unit: user_id, session_id, device_id, or geo cluster?
   □ Consistent assignment across sessions/devices?
   □ Is there risk of interference/spillover?

4. SIZING
   □ Baseline metric value
   □ MDE (Minimum Detectable Effect)
   □ Significance level (α = 0.05 typical)
   □ Power (1-β = 0.80 typical)
   □ Calculated sample size per variant

5. DURATION
   □ Min 1-2 full weeks (capture day-of-week effects)
   □ Account for seasonality
   □ Avoid running during holidays/sales events

6. ANALYSIS
   □ SRM check (Sample Ratio Mismatch)
   □ Statistical significance of primary metric
   □ Practical significance (is the effect meaningful?)
   □ Segmented analysis (mobile/desktop, new/returning)
   □ Check guardrail metrics

7. DECISION
   □ Ship / Don't Ship / Iterate
   □ Document learnings
```

### 2.2 Marketplace-Specific Pitfalls

#### Problem: SUTVA Violations (Network Effects)
In a two-sided marketplace, treating one group affects the other.

**Example:** If we show better search results to treatment buyers, they buy more → sellers in the treatment also benefit → but *control buyers* now have fewer items available (supply consumed) → control metrics look artificially worse → we overestimate the treatment effect.

**Solutions:**
| Approach | How It Works | Trade-off |
|---|---|---|
| **Geo-based clustering** | Randomize by city/region, not user | Higher variance, needs more data |
| **Switchback testing** | Alternate entire platform between T/C over time | Temporal effects, complex analysis |
| **Two-sided randomization** | Randomize both buyer and seller sides | Complex, potential information loss |
| **Cluster-based testing** | Define isolated clusters of buyers+sellers | Hard to find truly isolated clusters |

#### Problem: Sample Ratio Mismatch (SRM)
When your treatment/control split isn't 50/50 as designed (e.g., 51.2% vs 48.8%).

**Causes:** Bot filtering, redirect bugs, browser caching, JS load failures
**Detection:** Chi-squared test on group sizes
**Action:** Do NOT trust results until SRM is resolved

#### Problem: Novelty / Primacy Effects
- **Novelty:** Users interact more because something is "new" (inflates metrics temporarily)
- **Primacy:** Users resist change initially (deflates metrics temporarily)
**Solution:** Run test longer (2-4 weeks), segment by "new" vs "returning" users

#### Problem: Multiple Testing
Running 10 variants × 5 metrics = 50 comparisons → high false positive risk

**Corrections:**
- **Bonferroni:** α_adjusted = α / m (conservative)
- **Benjamini-Hochberg (FDR):** Controls false discovery rate (less conservative)
- **Sequential testing:** Early stopping rules (always valid p-values)

### 2.3 Practice Questions with Detailed Answers

**Q1: Design an A/B test for a new "Price Suggestion" tool for sellers.**

```
HYPOTHESIS: Showing ML-based price suggestions to sellers will
increase listing conversion rate by ≥3%.

METRICS:
  Primary:   Listing-to-sale conversion rate
  Secondary: Time-to-first-sale, seller NPS, listing price accuracy
  Guardrail: Buyer satisfaction, return rate, seller GMV

RANDOMIZATION:
  Unit: seller_id (not listing — one seller may have many listings)
  Assignment: Consistent across all their listings

SIZING:
  Baseline conversion: 15%
  MDE: 3% relative (0.45% absolute)
  α=0.05, power=0.80
  n ≈ 34,000 sellers per group (calculate using formula above)

DURATION: 3 weeks (capture weekly patterns, allow learning curve)

ANALYSIS:
  1. Check SRM
  2. Compare conversion rates (z-test for proportions)
  3. Segment: by category, seller tenure, price range
  4. Check if sellers who USE the suggestion differ from those who ignore it
     (but DO NOT exclude non-users — intent-to-treat analysis)
```

**Q2: Your A/B test shows +2% conversion but -1% AOV. What do you recommend?**

```
ANALYSIS:
  1. Is +2% conversion statistically significant? Check p-value.
  2. Is -1% AOV statistically significant? Or within noise?
  3. Compute NET IMPACT on GMV:
     GMV = Traffic × Conversion × AOV
     If conversion ↑2% and AOV ↓1%, net GMV change ≈ +0.98% (positive)
  4. Is the AOV drop because users are buying cheaper items?
     Or because the feature is attracting more price-sensitive users?
  5. Check GUARDRAILS: return rate, customer satisfaction
  6. RECOMMENDATION: If GMV is net positive and guardrails are safe,
     ship with monitoring. Set a review checkpoint at 4 weeks.
```

**Q3: How many users do we need for this test?**
```
Given: baseline_cr = 5%, MDE = 10% relative (0.5% absolute), α=0.05, power=0.80

n_per_group = (1.96 + 0.84)² × (0.05×0.95 + 0.055×0.945) / (0.005)²
            = (2.8)² × (0.0475 + 0.051975) / 0.000025
            = 7.84 × 0.099475 / 0.000025
            ≈ 31,195 users per group
            ≈ 62,390 total

At 10,000 users/day → run for ~7 days minimum
```

**Q4: When should you NOT use an A/B test?**
```
1. Too few users (rare events like luxury purchases)
   → Use Bayesian methods or pre-post analysis

2. Network effects are too strong (pricing changes)
   → Use switchback or geo experiments

3. Ethical concerns (showing inferior medical info)
   → Use observational causal inference methods

4. Long-term effects needed (subscription churn)
   → Use cohort analysis + difference-in-differences

5. One-time events (site redesign, holiday campaign)
   → Use synthetic control or interrupted time series
```

---

## 📐 Part 3: 20 Statistics Interview Questions

### Probability & Distributions

1. *"You flip a fair coin 10 times. What's the probability of getting exactly 7 heads?"*
   → Binomial: C(10,7) × 0.5¹⁰ = 120/1024 ≈ 11.7%

2. *"On average, eBay sees 3 fraudulent transactions per hour. What's the probability of seeing 0 in the next hour?"*
   → Poisson: P(X=0) = e⁻³ × 3⁰/0! = e⁻³ ≈ 4.98%

3. *"If 2% of listings are counterfeit and our model has 95% sensitivity and 90% specificity, what's the probability a flagged listing is actually counterfeit?"*
   → Bayes' theorem: P(counterfeit|flagged) = (0.95×0.02)/((0.95×0.02)+(0.10×0.98)) ≈ 16.2%

4. *"Explain the Central Limit Theorem. Why does it matter for A/B testing?"*

5. *"What's the difference between a confidence interval and a credible interval?"*

### Hypothesis Testing

6. *"Type I vs Type II error — which is worse for fraud detection vs. search ranking?"*
   → Fraud: Type II (missing fraud) is worse. Search: Type I (false positive change) wastes eng effort.

7. *"P-value is 0.03. Does this mean there's a 3% chance the null hypothesis is true?"*
   → NO. P-value = probability of observing data this extreme IF H₀ is true.

8. *"When would you use a one-tailed vs two-tailed test?"*

9. *"Your test has p=0.06. Your manager asks you to 'collect more data until it's significant.' Is this valid?"*
   → NO — this is p-hacking. Must pre-specify sample size. Use sequential testing if needed.

10. *"Explain the relationship between confidence level, sample size, and margin of error."*

### A/B Testing (Applied)

11. *"We ran a test for 2 days and got p=0.01. Should we stop early?"*
    → No. Short duration + significance → likely false positive (peeking problem). Need pre-set duration.

12. *"How do you handle A/B tests where the metric is highly skewed (e.g., revenue)?"*
    → Log-transform revenue, use trimmed means, or use non-parametric bootstrap.

13. *"What is a 'guardrail metric'? Give 3 examples for eBay search."*
    → Metric that must NOT degrade. Examples: page load time, seller impression fairness, bounce rate.

14. *"Explain the difference between statistical significance and practical significance."*
    → Stat sig = unlikely due to chance. Practical sig = large enough effect to matter for business.

15. *"An experiment shows +5% GMV but -2% seller satisfaction. How do you decide?"*

### Advanced / Bayesian

16. *"When would you use Bayesian A/B testing instead of frequentist?"*
    → When you want to incorporate prior knowledge, need continuous monitoring, or want P(A>B) directly.

17. *"Explain multi-armed bandit. When is it better than A/B testing?"*
    → MAB allocates more traffic to winning variant over time. Better for optimization (not measurement).

18. *"What is Simpson's Paradox? Give a marketplace example."*
    → Overall mobile converts better, but within each category, desktop converts better. Caused by different category mix per device.

19. *"How do you estimate causal effects without an experiment?"*
    → Diff-in-diff, regression discontinuity, instrumental variables, propensity score matching.

20. *"Explain the bootstrap. When is it useful?"*
    → Resample with replacement to estimate distribution of any statistic. Useful for medians, percentiles, non-standard metrics where CLT doesn't apply cleanly.
