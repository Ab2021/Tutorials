# Severity GLMs (Part 2) - Theoretical Deep Dive

## Overview
While Gamma regression handles "typical" claims well, it fails when losses are extremely skewed (Heavy Tails). This session introduces the **Inverse Gaussian (IG)** distribution for heavier tails and explores **Extreme Value Theory (EVT)** using the **Pareto** distribution for large loss modeling. We also discuss the "Splicing" technique: using one model for the body and another for the tail.

---

## 1. Conceptual Foundation

### 1.1 The Heavy Tail Problem

**Gamma Limitations:**
*   Gamma has "exponentially bounded" tails. The probability of a massive loss drops off too quickly.
*   *Reality:* Liability claims can be \$10M+. A Gamma model fitted to the body (\$5k claims) will predict the probability of a \$10M claim is essentially zero.

**Inverse Gaussian (IG):**
*   Like Gamma, it is positive and right-skewed.
*   **Heavier Tail:** It decays slower than Gamma.
*   **Variance Function:** $V(\mu) \propto \mu^3$. (Gamma was $\mu^2$).
*   *Interpretation:* As claims get larger, their volatility explodes even faster than in the Gamma world.

### 1.2 Extreme Value Theory (EVT)

**The "Black Swan" Domain:**
*   Standard statistics (CLT) deals with averages.
*   EVT deals with *maxima*.
*   **Pickands-Balkema-de Haan Theorem:** For a wide class of distributions, the conditional excess over a high threshold $u$ follows a **Generalized Pareto Distribution (GPD)**.

### 1.3 Large Loss Loading

**Strategy:**
1.  **Attritional Model (Body):** Use Gamma/IG for losses $< \$100k$.
2.  **Large Loss Model (Tail):** Use Pareto for losses $> \$100k$.
3.  **Total Pure Premium:** $E[Loss_{Body}] + E[Loss_{Tail}]$.

---

## 2. Mathematical Framework

### 2.1 Inverse Gaussian GLM

**PDF:**
$$ f(y; \mu, \lambda) = \sqrt{\frac{\lambda}{2\pi y^3}} \exp \left( \frac{-\lambda(y-\mu)^2}{2\mu^2 y} \right) $$

**GLM Properties:**
*   **Variance Function:** $V(\mu) = \mu^3$.
*   **Canonical Link:** $1/\mu^2$.
*   **Preferred Link:** Log ($\ln \mu$).
*   **Deviance:**
    $$ D = \sum \frac{(y_i - \hat{\mu}_i)^2}{\hat{\mu}_i^2 y_i} $$

### 2.2 The Pareto Distribution (Type II / Lomax)

Used for the tail ($x > \theta$).
$$ F(x) = 1 - \left( \frac{\theta}{x + \theta} \right)^\alpha $$
*   **Survival Function:** $S(x) = P(X > x) = ( \frac{\theta}{x+\theta} )^\alpha$.
*   **Mean Excess Function:** $e(u) = E[X-u | X>u]$.
    *   For Pareto, $e(u)$ is linear in $u$. This is a key diagnostic plot.

### 2.3 Splicing (Composite Models)

We define a piecewise PDF:
$$ f(x) = \begin{cases} w_1 f_{Gamma}(x) & x \le \text{Threshold} \\ w_2 f_{Pareto}(x) & x > \text{Threshold} \end{cases} $$
*   Weights $w_1, w_2$ ensure the PDF integrates to 1 and is continuous at the threshold.

---

## 3. Theoretical Properties

### 3.1 Tail Index ($\alpha$)

The parameter $\alpha$ in Pareto determines how heavy the tail is.
*   $\alpha > 2$: Finite Mean and Variance.
*   $1 < \alpha \le 2$: Finite Mean, Infinite Variance. (Dangerous).
*   $\alpha \le 1$: Infinite Mean. (Uninsurable?).

### 3.2 The Mean Excess Plot

*   Plot Mean Excess $e(u)$ vs. Threshold $u$.
*   **Gamma:** Downward slope (light tail).
*   **Lognormal:** Concave up.
*   **Pareto:** Straight line with positive slope.
*   *Diagnostic:* Look for where the plot becomes linear. That is your EVT threshold $u$.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fitting Inverse Gaussian

*   Same process as Gamma, just change the `family` argument.
*   **Check:** If the Gamma residuals show a "U" shape in the Q-Q plot (tails are too fat), switch to IG.

### 4.2 Fitting Pareto (Hill Estimator)

*   Sort losses: $x_{(1)} \ge x_{(2)} \ge \dots \ge x_{(n)}$.
*   Select top $k$ losses.
*   **Hill Estimator:**
    $$ \hat{\gamma} = \frac{1}{k} \sum_{i=1}^k \ln(x_{(i)}) - \ln(x_{(k+1)}) $$
    $$ \hat{\alpha} = 1/\hat{\gamma} $$

### 4.3 Model Specification (Python Example)

Comparing Gamma, Inverse Gaussian, and Pareto Tail Estimation.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# Simulate Heavy-Tailed Data (Lognormal)
np.random.seed(42)
n = 2000
mu_log = 8  # exp(8) ~ 3000
sigma_log = 1.5 # High volatility

losses = np.random.lognormal(mu_log, sigma_log, n)
# Add some covariates
x = np.random.normal(0, 1, n)
losses = losses * np.exp(0.2 * x)

df = pd.DataFrame({'Loss': losses, 'x': x})

# 1. Gamma GLM
gamma_model = smf.glm("Loss ~ x", data=df, 
                      family=sm.families.Gamma(link=sm.families.links.log())).fit()

# 2. Inverse Gaussian GLM
ig_model = smf.glm("Loss ~ x", data=df, 
                   family=sm.families.InverseGaussian(link=sm.families.links.log())).fit()

print(f"Gamma AIC: {gamma_model.aic:.0f}")
print(f"IG AIC:    {ig_model.aic:.0f}")
# IG should be lower if tails are heavy.

# 3. Pareto Tail Estimation (EVT)
threshold = np.percentile(losses, 95)
excess_losses = losses[losses > threshold] - threshold

# Fit GPD (Generalized Pareto)
# shape (c), loc (0), scale
params = genpareto.fit(excess_losses)
shape_param = params[0] # xi = 1/alpha
scale_param = params[2]

print(f"\nEVT Threshold: {threshold:.0f}")
print(f"Pareto Shape (xi): {shape_param:.3f}")
if shape_param > 0:
    print("Tail is Heavy (Fréchet domain)")
else:
    print("Tail is Light/Bounded")

# Visualization: Mean Excess Plot
thresholds = np.linspace(1000, 50000, 50)
mean_excess = []
for u in thresholds:
    excess = losses[losses > u] - u
    if len(excess) > 5:
        mean_excess.append(np.mean(excess))
    else:
        mean_excess.append(np.nan)

plt.figure(figsize=(10, 5))
plt.plot(thresholds, mean_excess, 'o-')
plt.title("Mean Excess Plot")
plt.xlabel("Threshold u")
plt.ylabel("Mean Excess E[X-u | X>u]")
plt.grid(True)
plt.show()

# Interpretation:
# If the plot is linear upward, Pareto is appropriate for the tail.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **IG Coefficients:** Impact on the mean (similar to Gamma).
2.  **Pareto Alpha:** The "Tail Index".
    *   $\alpha \approx 1.5$: Very dangerous line of business (e.g., Product Liability).
    *   $\alpha \approx 3.0$: Manageable (e.g., Auto Physical Damage).

**Interpretation:**
*   **Pricing:** For the top 5% of risks, do not use the GLM. Use the Pareto expectation.
*   **Reinsurance:** The Pareto model drives the price of the "Excess of Loss" treaty.

---

## 5. Evaluation & Validation

### 5.1 Q-Q Plots (Tail Check)

*   Plot Empirical Quantiles vs. Model Quantiles.
*   **Gamma:** Often curves away at the top (underpredicts extreme losses).
*   **IG:** Fits the tail better but might overpredict the body.

### 5.2 The "Vanishing Variance" Test

*   Split data into groups by predicted mean.
*   Calculate variance in each group.
*   Plot Log(Variance) vs Log(Mean).
*   Slope 2 = Gamma. Slope 3 = IG.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Using IG for Everything**
    *   **Issue:** IG has a very light left tail (near zero).
    *   **Reality:** If you have many small claims, IG might fit poorly there.
    *   **Fix:** Use Gamma for small claims, IG/Pareto for large.

2.  **Trap: Infinite Variance**
    *   **Issue:** If estimated Pareto $\alpha < 2$, the variance is infinite.
    *   **Consequence:** Standard errors of your predictions are meaningless. Simulation is required.

### 6.2 Implementation Challenges

1.  **Threshold Selection:**
    *   Choosing the EVT threshold $u$ is an art. Too low = Bias (not Pareto yet). Too high = Variance (not enough data).
    *   **Tool:** Hill Plot.

---

## 7. Advanced Topics & Extensions

### 7.1 Lognormal-Pareto Composite

*   A popular model in actuarial science.
*   Body: Lognormal.
*   Tail: Pareto.
*   Parameters estimated jointly to ensure continuity.

### 7.2 Catastrophe Modeling

*   For things like Hurricanes, historical data is insufficient (even with EVT).
*   We rely on **Vendor Models** (RMS, AIR) which simulate physics (wind speed, pressure) rather than just fitting curves to claims.

---

## 8. Regulatory & Governance Considerations

### 8.1 Capping Losses in Filings

*   Regulators often require you to cap losses (e.g., at \$100k) for the base rate indication.
*   **Excess Factor:** You calculate a separate factor for losses > \$100k, spread evenly across all policyholders (or by territory).
*   *Reason:* It's unfair to penalize a single zip code for one random \$5M claim.

---

## 9. Practical Example

### 9.1 Worked Example: Large Loss Loading

**Data:**
*   Total Losses (Capped at \$100k): \$50M.
*   Total Losses (Excess of \$100k): \$5M.
*   Earned Premium: \$60M.

**Base Indication:**
*   Loss Ratio (Capped) = 50/60 = 83.3%.

**Excess Load:**
*   Load = 5M / 50M = 10%.
*   **Final Indication:** $83.3\% \times 1.10 = 91.6\%$.

**Interpretation:**
*   We model the 83.3% using GLMs (Gamma/IG).
*   We apply the 1.10 factor as a flat multiplier at the end.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Inverse Gaussian** is the "Heavy Gamma".
2.  **Pareto** is the "King of Tails".
3.  **Splicing** combines the best of both worlds.

### 10.2 When to Use This Knowledge
*   **Commercial Liability:** Where tails are long.
*   **Reinsurance:** Pricing layers (e.g., \$5M xs \$5M).

### 10.3 Critical Success Factors
1.  **Don't Extrapolate Blindly:** A Pareto fit on \$100k claims might not hold for \$100M claims.
2.  **Check the Slope:** Use the Mean Excess Plot.
3.  **Separate the Tail:** Don't let the tail distort the body model.

### 10.4 Further Reading
*   **Embrechts, Klüppelberg, Mikosch:** "Modelling Extremal Events for Insurance and Finance".
*   **CAS Exam 8 Syllabus:** Advanced Ratemaking.

---

## Appendix

### A. Glossary
*   **Kurtosis:** Measure of "tailedness".
*   **Threshold ($u$):** The value above which EVT applies.
*   **Mean Excess Loss:** Average amount by which a claim exceeds $u$.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **IG Variance** | $\mu^3 / \lambda$ | Assumption |
| **Pareto Survival** | $(\theta / (x+\theta))^\alpha$ | Tail Prob |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
