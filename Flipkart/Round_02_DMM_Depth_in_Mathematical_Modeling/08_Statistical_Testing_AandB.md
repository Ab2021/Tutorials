# 📊 Statistical Testing & A/B Design — DMM Round
### Deep Dive: Every Statistical Concept You Need to Know

---

## SECTION 1: HYPOTHESIS TESTING FUNDAMENTALS

### The Full Framework (What Interviewers Expect You to Know)

```
1. STATE HYPOTHESES
   H₀: Null hypothesis (no effect, status quo)
   H₁: Alternative hypothesis (what you're trying to show)

2. CHOOSE SIGNIFICANCE LEVEL
   α = 0.05 (5% chance of false positive / Type I error)
   α = 0.01 for high-stakes (fraud decisions, medical interventions)
   
3. SELECT TEST STATISTIC
   - Proportion comparison (fraud rates): z-test for proportions
   - Mean comparison (average spend): t-test (paired or independent)
   - Distribution comparison: KS test (non-parametric)
   - Count data (click counts): chi-square test

4. COMPUTE p-VALUE
   p-value = P(test statistic ≥ observed | H₀ is true)
   
5. MAKE DECISION
   If p < α: Reject H₀ → statistically significant
   If p ≥ α: Fail to reject H₀ → not enough evidence

6. REPORT EFFECT SIZE (not just p-value!)
   Cohen's d for means, Relative Risk Reduction for proportions
   "Statistically significant ≠ Practically significant"
```

---

## SECTION 2: A/B TEST DESIGN FOR FRAUD MODELS

### Full Design Template

**Step 1: Define Hypotheses**
- H₀: New fraud model catch rate = Current model catch rate
- H₁: New fraud model catch rate > Current model catch rate (one-sided)

**Step 2: Define Parameters**
- $\alpha = 0.05$ (5% acceptable false positive rate — acceptable to declare improvement when there isn't one)
- Power = $1 - \beta = 0.80$ (80% chance of detecting true improvement)
- Minimum Detectable Effect (MDE): $\delta = 0.03$ (3 percentage point absolute improvement in catch rate)
- Baseline catch rate: $p_0 = 0.70$ (70% current catch rate)

**Step 3: Compute Sample Size**

$$n = \frac{(z_\alpha + z_\beta)^2 \cdot 2p(1-p)}{\delta^2}$$

Where $p = (p_0 + p_1)/2$, $z_{0.05} = 1.645$, $z_{0.20} = 0.842$

$$n = \frac{(1.645 + 0.842)^2 \cdot 2 \times 0.715 \times 0.285}{(0.03)^2} \approx 7,200 \text{ fraud cases per group}$$

At 1% fraud rate → need 720,000 transactions per group → ~72 days at 10K txns/day

**Step 4: Randomization**
- Unit: User-level (not transaction-level to avoid contamination within user)
- Method: Hash(user_id) % 100 < 50 → Treatment (consistent assignment per user)

**Step 5: Run**
- Duration: Max(statistical requirement, 2 full weeks for seasonality)
- Intermediate looks: Only at pre-specified points (Pocock or O'Brien-Fleming boundaries)

**Step 6: Analyze**
- Primary: Two-proportion z-test on fraud catch rate
- Guardrail: Mann-Whitney U on false positive rate (non-parametric — more robust)
- Segmentation: Check if effect is consistent across user segments (don't just report aggregate)

---

## SECTION 3: MULTIPLE TESTING PROBLEM

**Why it matters:** If you run 20 A/B tests simultaneously at α=0.05, you'd expect 1 false positive by chance alone — even if none of the variants actually work.

**Solutions:**

**Bonferroni Correction (Conservative):**
$$\alpha_{adjusted} = \frac{\alpha}{m} = \frac{0.05}{20} = 0.0025$$
Pro: Simple, controls FWER (Family-Wise Error Rate)
Con: Very conservative when tests are correlated → loses power

**Benjamini-Hochberg (FDR Control — Preferred for many tests):**
1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest k: $p_{(k)} \leq \frac{k}{m} \times \alpha_{FDR}$
3. Reject all H₀ for $p \leq p_{(k)}$

Controls False Discovery Rate (FDR) = expected fraction of false positives among rejections.
Better: allows more discoveries while controlling error rate.

**In fraud A/B testing context:** When testing a new model on 10 customer segments simultaneously, use Bonferroni (few tests, high stakes). When doing exploratory feature selection on 200 features, use FDR.

---

## SECTION 4: BAYESIAN A/B TESTING

**Advantage over frequentist:** Can stop early, no p-value threshold, provides probability of being best.

**Setup:**
- Prior: $\theta_A \sim \text{Beta}(\alpha_0, \beta_0)$ (encode historical fraud rate belief)
- Data: $k_A$ frauds detected out of $n_A$ actual frauds
- Posterior: $\theta_A | data \sim \text{Beta}(\alpha_0 + k_A, \beta_0 + n_A - k_A)$

**Decision metric:**
$$P(\theta_B > \theta_A) = \int_0^1 P(\theta_A < \theta_B) d\theta_B$$

Computed via Monte Carlo simulation:
```python
import numpy as np

def bayesian_ab_test(alpha_A, beta_A, alpha_B, beta_B, n_samples=100000):
    """
    Returns P(B > A) via Monte Carlo simulation.
    Alpha, Beta are posterior parameters.
    """
    samples_A = np.random.beta(alpha_A, beta_A, n_samples)
    samples_B = np.random.beta(alpha_B, beta_B, n_samples)
    prob_B_better = (samples_B > samples_A).mean()
    
    # Expected lift
    expected_lift = (samples_B - samples_A).mean()
    
    return prob_B_better, expected_lift

# Example:
# Model A: Caught 700 out of 1000 fraud cases (with base prior 1,1)
# Model B: Caught 740 out of 1000 fraud cases
prob = bayesian_ab_test(1+700, 1+300, 1+740, 1+260)
```

**When to stop:** P(B > A) > 0.95 and Expected Lift > MDE (minimum detectable effect).

---

## SECTION 5: DISTRIBUTION TESTS

**KS Test (Kolmogorov-Smirnov):**
Tests if two samples come from the same distribution. Non-parametric.

$$D = \sup_x |F_1(x) - F_2(x)|$$

**In fraud context:**
- Compare fraud score distribution before vs. after model update (did the score distribution shift?)
- Compare feature distribution in training vs. serving (detect training-serving skew)
- PSI (Population Stability Index): similar to KS but gives an index

**PSI Formula:**
$$PSI = \sum_{i} (P_{new_i} - P_{base_i}) \times \ln\left(\frac{P_{new_i}}{P_{base_i}}\right)$$

| PSI Value | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 – 0.25 | Moderate shift | Monitor closely |
| > 0.25 | Significant shift | Retrain model |

---

## SECTION 6: CAUSAL INFERENCE (Senior-Level Topic)

**Why pure correlation fails in fraud:**
Correlation: Accounts that use mobile apps more tend to be less fraudulent.
Causal question: DOES using mobile apps make accounts less fraudulent? Or do legitimate users just prefer apps?

If we can't answer the causal question, we can't intervene effectively.

**Tools:**

**Randomized Controlled Trial (RCT):** Gold standard. Randomly assign treatment. Not always feasible (can't randomly make some users use mobile app).

**Difference-in-Differences (DiD):**
$$\hat{\tau} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$$

Assumption: Parallel trends — treatment and control would have moved together without intervention.

**Propensity Score Matching:**
Match treated users to similar untreated users on all observable confounders. Compare outcomes within matched pairs. Assumption: no unobserved confounders.

**Instrumental Variables (IV):**
Use an instrument Z (correlated with treatment X, affects outcome Y only through X).
Example: Weather affects in-store promotions (instrument) → promotions affect purchases → causal effect of promotion on purchase.

---

## SECTION 7: SURVIVAL ANALYSIS (From Your Resume)

**Kaplan-Meier Estimator:**
$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

Where $d_i$ = events (deaths/churns) at time $t_i$, $n_i$ = at-risk population.

**Key property:** Handles censoring — patients who leave the study without experiencing the event are included up to their last known alive time.

**Log-Rank Test:**
Compare survival curves between groups (e.g., high vs. low CLV customers).
$$\chi^2 = \frac{(\sum_i (O_i - E_i))^2}{\sum_i V_i}$$

**Cox Proportional Hazards:**
$$h(t|x) = h_0(t) \exp(\beta^T x)$$

**In insurance/CLV context:**
- Event: customer churn / policy lapse
- Censored: customers still active at study end
- Features: policy type, claim history, premium level
- Output: Hazard ratio per feature (exp(β)) — relative risk multiplier

**Unusual question:** Why is Cox PH model semi-parametric?

Answer: The baseline hazard $h_0(t)$ is non-parametric (estimated from data without assuming any distribution). Only the covariate effect $\exp(\beta^T x)$ is parametric. This flexibility is why Cox PH is so widely used — no need to specify the shape of the baseline hazard.

---

## SECTION 8: QUICK FORMULA REFERENCE

| Test | Formula | Use Case |
|---|---|---|
| Z-test (proportions) | $z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}}$ | A/B test on fraud rate |
| t-test (means) | $t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1 + 1/n_2}}$ | Compare average CLV |
| Chi-square | $\chi^2 = \sum\frac{(O-E)^2}{E}$ | Categorical association |
| Sample size | $n = \frac{(z_\alpha + z_\beta)^2 2p(1-p)}{\delta^2}$ | A/B test planning |
| Power | $1 - \Phi(z_\alpha - \frac{\delta}{\sqrt{p(1-p)/n}})$ | Post-hoc power analysis |
| Cohen's d | $d = \frac{\mu_1 - \mu_2}{\sigma_{pooled}}$ | Effect size for means |
| PSI | $\sum(P_{new}-P_{base})\ln(P_{new}/P_{base})$ | Input drift monitoring |
| KS Stat | $D = \sup_x |F_1(x) - F_2(x)|$ | Distribution comparison |
| Kaplan-Meier | $\hat{S}(t) = \prod_{t_i\leq t}(1 - d_i/n_i)$ | Survival curve |
| Cox PH | $h(t|x) = h_0(t)\exp(\beta^T x)$ | Hazard modeling |
