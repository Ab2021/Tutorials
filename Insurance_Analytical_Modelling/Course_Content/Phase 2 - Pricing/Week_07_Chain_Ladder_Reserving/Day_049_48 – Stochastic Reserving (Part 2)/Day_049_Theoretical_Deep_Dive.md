# Stochastic Reserving (Bayesian MCMC) - Theoretical Deep Dive

## Overview
Bootstrapping is powerful, but it relies entirely on the data. What if the data is thin, volatile, or missing? **Bayesian Reserving** allows us to incorporate **Expert Judgment** (Priors) into the stochastic framework. We explore **MCMC (Markov Chain Monte Carlo)**, the **Bayesian Chain Ladder**, and how to implement these models using probabilistic programming languages like **PyMC** or **Stan**.

---

## 1. Conceptual Foundation

### 1.1 The Bayesian Philosophy

**Frequentist (Bootstrap):**
*   "The LDF is a fixed, unknown number. The data is random."
*   We resample the data to see what the LDF *could* have been.

**Bayesian (MCMC):**
*   "The LDF is a random variable. The data is fixed."
*   We start with a belief (Prior) about the LDF, update it with data (Likelihood), and get a revised belief (Posterior).
*   $$ P(\theta | Data) \propto P(Data | \theta) \times P(\theta) $$

### 1.2 Why MCMC?

*   Calculating the Posterior analytically is impossible for complex models.
*   **MCMC (Markov Chain Monte Carlo):** A simulation technique that "wanders" around the parameter space.
*   It spends more time in areas of high probability.
*   After 10,000 steps, the history of the wanderer is a sample from the Posterior Distribution.

### 1.3 Hierarchical Models

*   **Pooling Strength:** If you have 50 small states, you can model them together.
*   State $i$ LDF $\sim$ Normal(Global Mean, Global Variance).
*   **Shrinkage:** Small states are pulled towards the Global Mean. Large states rely on their own data.

---

## 2. Mathematical Framework

### 2.1 Bayesian Chain Ladder (BCL)

**Likelihood (The Data):**
$$ C_{i, j+1} | C_{i, j}, f_j, \sigma_j \sim \text{LogNormal}(\ln(f_j \cdot C_{i, j}), \sigma_j^2) $$
*   This is the standard Chain Ladder assumption (Mack's Model).

**Priors (The Beliefs):**
*   $f_j \sim \text{LogNormal}(\mu_j, \tau_j^2)$.
    *   $\mu_j$: Industry Benchmark LDF.
    *   $\tau_j$: How confident are we in the benchmark?
*   $\sigma_j \sim \text{Gamma}(\alpha, \beta)$. (Prior on volatility).

### 2.2 The Posterior

$$ P(f, \sigma | C) \propto \prod_{i,j} \text{LogNormal}(C_{i, j+1} | \dots) \times \prod_j P(f_j) P(\sigma_j) $$
*   MCMC samples from this joint distribution.
*   **Result:** 10,000 sets of LDFs $\{f_0, f_1, \dots, f_{ult}\}$.

---

## 3. Theoretical Properties

### 3.1 Credibility "Built-In"

*   If Data Variance ($\sigma^2$) is high and Prior Variance ($\tau^2$) is low $\to$ Posterior stays close to Prior.
*   If Data Variance is low and Prior Variance is high $\to$ Posterior moves to Data.
*   **Bayesian MCMC automatically performs the Benktander/Credibility weighting.**

### 3.2 Tail Estimation

*   In Bootstrap, we can't bootstrap the tail (no data).
*   In Bayesian, we set a Prior for the Tail Factor (e.g., LogNormal(1.05, 0.02)).
*   The model integrates this uncertainty into the total reserve distribution naturally.

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyMC Implementation (Python)

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# Data: Cumulative Triangle (Flattened)
# x = C_{i, j}, y = C_{i, j+1}, dev_factor_index = j
# Assume we have vectors: x_data, y_data, dev_idx

with pm.Model() as bcl_model:
    # 1. Priors for LDFs (f)
    # Log-Normal priors centered at Industry Benchmark
    industry_ldfs = [1.5, 1.2, 1.1, 1.05] 
    f = pm.LogNormal("f", mu=np.log(industry_ldfs), sigma=0.1, shape=4)
    
    # 2. Prior for Sigma (Process Noise)
    sigma = pm.HalfCauchy("sigma", beta=1.0, shape=4)
    
    # 3. Likelihood
    # Expected value in next period
    mu_y = pm.math.log(x_data) + pm.math.log(f[dev_idx])
    
    # Observation
    y_obs = pm.LogNormal("y_obs", mu=mu_y, sigma=sigma[dev_idx], observed=y_data)
    
    # 4. Sampling
    trace = pm.sample(2000, tune=1000, chains=2)

# 5. Analysis
az.plot_trace(trace)
summary = az.summary(trace)
print(summary)

# 6. Prediction (Posterior Predictive Check)
with bcl_model:
    ppc = pm.sample_posterior_predictive(trace)
    
# The 'ppc' contains simulated future triangles.
# We sum the future diagonals to get the Reserve Distribution.
```

### 4.2 Stan / JAGS

*   **Stan:** Uses Hamiltonian Monte Carlo (HMC). Faster convergence for high-dimensional models.
*   **JAGS:** Uses Gibbs Sampling. Easier to write for simple conjugate models.
*   *Actuarial Standard:* Stan is becoming the gold standard due to its speed and the `brms` R package interface.

---

## 5. Evaluation & Validation

### 5.1 Trace Plots

*   **Convergence:** The trace should look like a "fuzzy caterpillar".
*   If it looks like a snake or gets stuck, the chain hasn't converged.
*   **R-hat Statistic:** Should be $< 1.05$. If $> 1.1$, run longer or fix the model.

### 5.2 Posterior Predictive Checks (PPC)

*   Simulate the *past* data using the fitted model.
*   Does the model reproduce the historical volatility?
*   If the actual data lies outside the 95% PPC interval, the model is misspecified (e.g., missed a calendar trend).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Uniform" Prior**
    *   **Issue:** Using `Uniform(0, 100)` for LDFs to be "unbiased".
    *   **Reality:** This is actually informative! It says 99.0 is just as likely as 1.5.
    *   **Fix:** Use "Weakly Informative" priors (e.g., LogNormal or HalfNormal) that constrain parameters to a reasonable range.

2.  **Trap: Over-parameterization**
    *   **Issue:** Fitting a separate sigma for every development period in a small triangle.
    *   **Result:** MCMC won't converge.
    *   **Fix:** Pool variances (e.g., $\sigma_j = \sigma \cdot e^{-\delta j}$).

### 6.2 Implementation Challenges

1.  **Computation Time:**
    *   Bootstrap: 10 seconds.
    *   MCMC: 10 minutes (or hours for complex hierarchical models).
    *   *Not suitable for real-time pricing, but fine for quarterly reserving.*

---

## 7. Advanced Topics & Extensions

### 7.1 Changing Settlement Rates (Berquist-Sherman in Bayes)

*   Add a parameter $\gamma_k$ for Calendar Year $k$ representing "Speed of Payment".
*   $E[C_{i, j}] = f_j \cdot C_{i, j-1} \cdot \gamma_{i+j}$.
*   MCMC can estimate $f$ and $\gamma$ simultaneously, separating the "true" development from the "speedup".

### 7.2 Correlated Lines (Copulas in Bayes)

*   Model Line A and Line B together.
*   Assume the residuals of Line A and Line B come from a Multivariate Normal distribution with correlation $\rho$.
*   MCMC estimates $\rho$ from the data.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Expert Judgment" Documentation

*   Regulators love Bayesian methods *if* you document the priors.
*   **Bad:** "We used a Prior."
*   **Good:** "We used a LogNormal Prior centered at 1.05 with a CV of 10%, based on NCCI industry data."

---

## 9. Practical Example

### 9.1 Worked Example: The "Sparse" Triangle

**Scenario:**
*   Reinsurer with only 3 years of data.
*   **Chain Ladder:** Fails (cannot calculate LDFs for later ages).
*   **Bootstrap:** Fails (not enough residuals).
*   **Bayesian:**
    *   Prior: Industry Curve.
    *   Likelihood: The 3 data points.
    *   **Result:** The model produces a reserve distribution that essentially reflects the Industry Curve variance, slightly updated by the 3 data points.
    *   *Value:* It gives a scientifically defensible range where other methods give `NaN`.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Priors** stabilize estimates.
2.  **MCMC** solves the math.
3.  **Posterior** is the answer.

### 10.2 When to Use This Knowledge
*   **Sparse Data:** New products, Reinsurance.
*   **Complex Structures:** Hierarchies, Correlations, Changing Trends.

### 10.3 Critical Success Factors
1.  **Check Convergence:** Always look at the trace plots.
2.  **Sensitivity Test Priors:** Does changing the prior change the answer? If yes, your data is weak (which is good to know).
3.  **Start Simple:** Don't build a hierarchical model until the simple BCL works.

### 10.4 Further Reading
*   **Verrall (2004):** "Bayesian models for claims reserving".
*   **Gelman et al.:** "Bayesian Data Analysis" (The Bible of Bayes).

---

## Appendix

### A. Glossary
*   **MCMC:** Markov Chain Monte Carlo.
*   **Trace Plot:** History of the MCMC sampler.
*   **Burn-in:** The first 1000 samples thrown away to let the chain converge.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Bayes Rule** | $P(\theta|D) \propto P(D|\theta)P(\theta)$ | Inference |
| **LogNormal** | $e^{\mu + \sigma Z}$ | LDF Prior |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
