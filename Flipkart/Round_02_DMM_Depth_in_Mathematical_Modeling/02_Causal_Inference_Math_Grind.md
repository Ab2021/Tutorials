# 📈 DMM GRIND — Causal Inference & Experimentation Math
### Depth in Mathematical Modeling (Questions 51-75)
> Flipkart relies heavily on A/B testing and Causal Inference. Correlation is not causation; this document proves you know the math to tell the difference.

---

## ═══════════════════════════════════════
## SECTION F: ADVANCED CAUSAL INFERENCE
## ═══════════════════════════════════════

### Q51: Define the Fundamental Problem of Causal Inference. How does the Rubin Causal Model (Potential Outcomes framework) formalize it?
**Expected Answer:**
**Fundamental Problem:** For any individual $i$, we can only observe the outcome of ONE treatment state. We cannot simultaneously observe what happens if they get the treatment ($Y_i(1)$) AND if they don't ($Y_i(0)$).
**Rubin Causal Model:**
- $Y_i(1)$: Potential outcome if treated.
- $Y_i(0)$: Potential outcome if control.
- Individual Treatment Effect (ITE): $\tau_i = Y_i(1) - Y_i(0)$ (Unobservable).
- Average Treatment Effect (ATE): $E[Y_i(1) - Y_i(0)]$.
**Solution via Randomization:** If treatment $W_i \in \{0,1\}$ is randomly assigned, then $W_i \perp (Y_i(1), Y_i(0))$. 
This implies $E[Y_i(1)|W_i=1] = E[Y_i(1)]$.
Thus, ATE = $E[Y_i|W_i=1] - E[Y_i|W_i=0]$. Randomization makes the unobservable counterfactual average equal to the observed control average.

### Q52: What is Simpson's Paradox? Prove mathematically how it can happen.
**Expected Answer:**
**Concept:** A trend appears in several different groups of data but disappears or reverses when these groups are combined.
**Math/Example:**
Let $T$ be treatment (new model vs old), $S$ be success (fraud caught), and $C$ be confounder (transaction type: High-value vs Low-value).
Treatment $T=1$ is better in both groups:
$P(S=1 | T=1, C=High) > P(S=1 | T=0, C=High)$  (e.g., 90% > 85%)
$P(S=1 | T=1, C=Low) > P(S=1 | T=0, C=Low)$    (e.g., 20% > 10%)
However, overall:
$P(S=1 | T=1) < P(S=1 | T=0)$
**Why?** Because $T=1$ was given disproportionately to $C=Low$ (the harder group).
Let $P(C=Low | T=1) = 0.9$ and $P(C=Low | T=0) = 0.1$.
$P(S|T=1) = 0.9(0.2) + 0.1(0.9) = 0.18 + 0.09 = 0.27$
$P(S|T=0) = 0.1(0.1) + 0.9(0.85) = 0.01 + 0.765 = 0.775$.
0.27 < 0.775. The overall effect reversed due to the confounder distribution!
**Solution:** Control for covariates! Use Stratification or Inverse Probability Weighting (IPW).

### Q53: Derive Inverse Probability Weighting (IPW) and explain when to use it over matching.
**Expected Answer:**
**Goal:** Estimate ATE from observational data where treatment $W$ is not random but depends on covariates $X$. Assumes unconfoundedness: $(Y(0), Y(1)) \perp W | X$.
**Propensity Score:** $e(X) = P(W=1|X)$ (usually estimated via Logistic Regression).
**IPW Estimator:**
$$\hat{\tau}_{IPW} = \frac{1}{N}\sum_{i=1}^N \left( \frac{W_i Y_i}{e(X_i)} - \frac{(1-W_i) Y_i}{1-e(X_i)} \right)$$
**Proof of unbiasedness (for ATE):**
$E[\frac{W Y}{e(X)} | X] = \frac{1}{e(X)} E[W Y(1) | X] = \frac{1}{e(X)} P(W=1|X) E[Y(1)|X] = E[Y(1)|X]$
Taking expectation over X yields $E[Y(1)]$. Similarly for $Y(0)$.
**When to use IPW vs Matching:**
- IPW is better for high-dimensional $X$ (avoids curse of dimensionality by reducing to scalar $e(X)$).
- Matching is better when $e(X)$ is extremely close to 0 or 1 (overlap violation), because in IPW dividing by $e(X) \approx 0$ causes variance explosion.

### Q54: Explain Difference-in-Differences (DiD). What is the critical assumption and how do you test it?
**Expected Answer:**
**Concept:** Observational panel data. Treatment group gets treated at time $t^*$, control group never gets treated.
**Estimator:**
$$\hat{\tau}_{DiD} = (E[Y_{treatment, post}] - E[Y_{treatment, pre}]) - (E[Y_{control, post}] - E[Y_{control, pre}])$$
Removes baseline differences between groups (first diff) and general time trends (second diff).
**Math underlying it:**
Assume $Y_{igt} = \alpha_i + \gamma_t + \tau D_{it} + \epsilon_{igt}$ (fixed effects for group $\alpha_i$ and time $\gamma_t$).
DiD subtracts out $\alpha_i$ and $\gamma_t$, leaving $\tau$.
**Critical Assumption: Parallel Trends.** In the absence of treatment, the treatment and control groups would have followed the same trajectory.
**Testing it:** Event-study plot. Plot the difference between treatment and control means for multiple time periods *prior* to $t^*$. If the line is flat at 0, parallel trends holds pre-treatment, supporting the assumption.

### Q55: What is regression to the mean mathematically, and how does it ruin naive A/B testing?
**Expected Answer:**
**Math:** Let $X \sim \mathcal{N}(\mu, \sigma^2)$ be the true skill/quality. Let observed value $Y = X + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)$.
If we select a subpopulation with exceptionally low $Y$ (e.g., worst performing sellers) to get "treatment", their expected true value is:
$E[X | Y = y] = \mu + \frac{\sigma^2}{\sigma^2 + \sigma_\epsilon^2}(y - \mu)$
Since $\frac{\sigma^2}{\sigma^2 + \sigma_\epsilon^2} < 1$, the expected true value is closer to $\mu$ than the observed $y$.
In the next period, their expected observed value $Y_{t+1}$ will move toward $\mu$ naturally, *even with zero treatment effect*.
**Ruin:** If Flipkart introduces a "seller training program" only for the bottom 10% sellers, their metrics will improve next month strictly due to regression to the mean. A naive pre/post comparison will falsely conclude the training was highly effective. Randomization breaks this!

---

## ═══════════════════════════════════════
## SECTION G: ADVANCED A/B TESTING MATH
## ═══════════════════════════════════════

### Q56: Derive the required Sample Size formula for an A/B test.
**Expected Answer:**
Testing $H_0: \mu_1 - \mu_2 = 0$ vs $H_A: \mu_1 - \mu_2 = \delta$.
Variances assumed equal $\sigma^2$ and sample sizes equal $n$.
Test statistic $Z = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{2\sigma^2/n}}$.
1. For Type I error $\alpha$ (usually 0.05): We reject $H_0$ if $Z > Z_{1-\alpha/2}$.
2. For Type II error $\beta$ (Power = $1-\beta$, usually 0.80): Under $H_A$, mean of $Z$ is roughly $\frac{\delta}{\sqrt{2\sigma^2/n}}$.
We need $P(Reject | H_A) = 1-\beta$.
This implies: $Z_{1-\beta} = \frac{\delta}{\sqrt{2\sigma^2/n}} - Z_{1-\alpha/2}$.
Solving for $n$:
$$n = \frac{2\sigma^2 (Z_{1-\alpha/2} + Z_{1-\beta})^2}{\delta^2}$$
**Key Insights:**
- To detect half the effect size ($\delta/2$), you need 4x the sample size!
- Variance ($\sigma^2$) directly dictates sample size. Variance reduction techniques (CUPED) mathematically reduce required sample size.

### Q57: What is CUPED and how does it reduce variance in A/B tests mathematically?
**Expected Answer:**
**CUPED (Controlled Experiment Using Pre-Experiment Data):** Uses pre-experiment covariate $X$ (e.g., user's GMV last month) to reduce variance of the primary metric $Y$ (e.g., user's GMV during experiment).
Define adjusted metric: $\hat{Y}_{cuped} = Y - \theta(X - E[X])$.
Mean is preserved: $E[\hat{Y}_{cuped}] = E[Y]$.
Variance: 
$$Var(\hat{Y}_{cuped}) = Var(Y) + \theta^2 Var(X) - 2\theta Cov(X,Y)$$
Minimize variance by setting derivative w.r.t $\theta$ to zero:
$$\theta^* = \frac{Cov(X,Y)}{Var(X)}$$
Substitute optimal $\theta$:
$$Var(\hat{Y}_{cuped}) = Var(Y)(1 - \rho^2)$$
where $\rho$ is the correlation between pre-experiment covariates and experiment metric.
**Impact:** If past behavior correlates with future behavior at $\rho = 0.7$, variance is reduced by $1 - 0.7^2 = 51\%$. This halves the required sample size!

### Q58: Explain the Multiple Comparisons Problem and derive the Bonferroni Correction.
**Expected Answer:**
**Problem:** If you test 20 different features (hypotheses) at $\alpha = 0.05$, the probability of getting at least one false positive purely by chance is:
$P(\geq 1 \text{ false pos}) = 1 - P(\text{0 false pos}) = 1 - (1-0.05)^{20} \approx 64\%$.
**Bonferroni Correction:** Controls the Family-Wise Error Rate (FWER).
We want $P(\bigcup_{i=1}^m \{p_i \leq \alpha_{adjusted}\}) \leq \alpha$.
By Boole's Inequality (Union Bound):
$P(\bigcup E_i) \leq \sum P(E_i) = m \times \alpha_{adjusted}$.
Set $m \times \alpha_{adjusted} \leq \alpha \implies \alpha_{adjusted} = \alpha / m$.
**Drawback:** Very conservative. Increases Type II error (reduces power). Better modern alternative is controlling False Discovery Rate (FDR) using Benjamini-Hochberg procedure.

### Q59: How do you calculate significance for a ratio metric (e.g., Click-Through Rate = Clicks/Views) where denominator varies by user?
**Expected Answer:**
Naive approach: Just treat each view as a binomial trial. WRONG. Users have different numbers of views; users with many clicks dominate, violating the i.i.d. assumption.
**Math:** Let $R = \frac{\sum Y_i}{\sum X_i}$ (Total clicks / Total views). This is a ratio of random variables.
Use **Delta Method** to estimate the variance of a ratio.
Let $\mu_Y = E[Y], \mu_X = E[X]$. Taylor expansion yields:
$$Var(R) \approx \frac{1}{\mu_X^2} \left[ Var(Y) - 2R \cdot Cov(X,Y) + R^2 Var(X) \right]$$
In practice, use **Bootstrap** or the **Delta Method via summary statistics** per user: compute empirical variance of $Y_i - R X_i$. Calculate t-test using this corrected variance.

### Q60: Formulate the Multi-Armed Bandit using Thompson Sampling.
**Expected Answer:**
Instead of a fixed 50/50 A/B test, dynamically route traffic to the winning variant.
1. Assume reward for variant $k$ follows Bernoulli distribution with unknown parameter $\theta_k$.
2. Prior for $\theta_k$ is $Beta(\alpha_k, \beta_k)$ (Start with $\alpha=1, \beta=1$ uniform).
3. Observation: $t$ trials, $s$ successes, $f$ failures.
4. Posterior update: $\theta_k | \text{data} \sim Beta(\alpha_k + s, \beta_k + f)$.
**Algorithm:**
For each user:
- Sample a random $\hat{\theta}_k$ from the Posterior $Beta(\alpha_k, \beta_k)$ for all $K$ variants.
- Assign user to variant $k^* = \arg\max \hat{\theta}_k$.
- Observe reward. Update $\alpha, \beta$ for variant $k^*$.
**Intuition:** If variant A has high uncertainty (few pulls), its Beta distribution is wide, yielding occasional high samples → Exploration. If variant A has high proven success, its Beta mean is high → Exploitation.

---
*End of Causal & Stats Grind.*
