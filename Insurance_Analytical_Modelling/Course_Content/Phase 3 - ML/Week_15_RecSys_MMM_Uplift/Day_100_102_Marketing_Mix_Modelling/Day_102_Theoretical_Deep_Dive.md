# Marketing Mix Modelling (MMM) (Part 2) - Advanced Bayesian Methods - Theoretical Deep Dive

## Overview
Yesterday, we built a simple Ridge Regression MMM. Today, we enter the modern era: **Bayesian MMM**. By combining "Priors" (Industry knowledge) with "Likelihood" (Data), we solve the biggest problem in MMM: **Multicollinearity**. We also explore **Geo-Lift Experiments** to calibrate our models.

---

## 1. Conceptual Foundation

### 1.1 Why Bayesian?

*   **The Problem with Frequentist (OLS/Ridge):**
    *   If TV and Search spend are highly correlated (0.9), OLS might say TV has a *negative* coefficient.
    *   *Actuarial View:* We *know* TV doesn't hurt sales.
*   **The Bayesian Solution:**
    *   We set a **Prior**: "TV coefficient must be positive (Normal distribution, mean=0.5, sd=0.1)."
    *   The model updates this belief based on data.

### 1.2 Calibration with Experiments

*   **Triangulation:**
    1.  **MMM:** Strategic, long-term view. (Correlation-based).
    2.  **Experiments (Geo-Lift):** Causal, short-term view. (Gold Standard).
*   **Calibration:** If a Geo-Lift test says TV ROAS is 2.0, we force the MMM to be close to 2.0.

### 1.3 Budget Optimization

*   **Goal:** Maximize Revenue subject to Budget $\le$ \$10M.
*   **Response Curves:** We need the *shape* of the saturation curve for every channel.
*   **Algorithm:** Sequential Least Squares Programming (SLSQP) or Genetic Algorithms.

---

## 2. Mathematical Framework

### 2.1 The Bayesian Formula

$$ P(\theta | Data) \propto P(Data | \theta) \times P(\theta) $$

*   **Posterior ($P(\theta | Data)$):** Our updated belief about Ad Effectiveness.
*   **Likelihood ($P(Data | \theta)$):** How well the model fits the sales data.
*   **Prior ($P(\theta)$):** Our initial guess (e.g., "TV ROAS is likely between 0.5 and 1.5").

### 2.2 Causal Impact (Synthetic Control)

$$ \tau_t = Y_t - \hat{Y}_t $$

*   **Scenario:** We stop ads in "Kansas" (Treatment) but keep them in "Missouri" (Control).
*   **Counterfactual ($\hat{Y}_t$):** What *would have happened* in Kansas if we kept ads running?
*   **Impact ($\tau_t$):** The difference.

---

## 3. Theoretical Properties

### 3.1 MCMC Sampling (NUTS)

*   **No-U-Turn Sampler (NUTS):** The algorithm used by PyMC3/NumPyro to explore the high-dimensional parameter space.
*   **Trace Plots:** We check if the "chains" have converged (mixed well).

### 3.2 Hierarchical Modeling

*   **Structure:**
    *   Global Mean (National Effectiveness).
    *   State Deviation (Local Effectiveness).
*   **Benefit:** "Borrowing Strength". Small states (Wyoming) get better estimates by borrowing data from large states (California).

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyMC3 Implementation (Simplified)

```python
import pymc3 as pm
import numpy as np

# Data
tv_spend = np.array([...])
sales = np.array([...])

with pm.Model() as mmm:
    # 1. Priors
    intercept = pm.Normal("intercept", mu=1000, sigma=100)
    beta_tv = pm.HalfNormal("beta_tv", sigma=1) # Positive only
    
    # 2. Adstock & Saturation (Deterministic)
    tv_adstock = geometric_adstock_pymc(tv_spend, decay=0.5)
    tv_effect = beta_tv * hill_function_pymc(tv_adstock)
    
    # 3. Likelihood
    mu = intercept + tv_effect
    sigma = pm.HalfNormal("sigma", sigma=100)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=sales)
    
    # 4. Sampling
    trace = pm.sample(1000, tune=1000)

# 5. Results
pm.plot_posterior(trace)
```

### 4.2 Budget Optimization (SciPy)

```python
from scipy.optimize import minimize

def objective(budget_allocation):
    # Calculate Total Sales based on Hill Curves
    return -1 * total_sales(budget_allocation)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 10_000_000})
bounds = [(0, 5_000_000), (0, 5_000_000)] # Min/Max per channel

result = minimize(objective, x0=[5_000_000, 5_000_000], bounds=bounds, constraints=constraints)
print(f"Optimal Allocation: {result.x}")
```

---

## 5. Evaluation & Validation

### 5.1 Posterior Predictive Checks (PPC)

*   **Concept:** Generate 1,000 fake datasets using the model.
*   **Check:** Does the *real* data look like the *fake* data?
*   **Bayesian P-value:** If the real data is an outlier in the posterior, the model is bad.

### 5.2 Out-of-Sample Lift

*   **Validation:** The model predicts "Stopping TV will drop sales by 10%".
*   **Test:** Stop TV.
*   **Result:** Sales dropped 9%. (Model is calibrated).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Flat Prior" Fallacy**
    *   "I don't want to bias the model, so I'll use a flat prior."
    *   *Reality:* A flat prior is still a choice. It says "ROAS of 1000 is just as likely as ROAS of 1". This is wrong. Use Weakly Informative Priors.

2.  **Trap: Overfitting with too many channels**
    *   Trying to model "Facebook Mobile Video" vs. "Facebook Desktop Feed".
    *   *Fix:* Group channels (e.g., "Social Video").

### 6.2 Implementation Challenges

1.  **Convergence Issues:**
    *   "Divergences" in NUTS sampling.
    *   *Fix:* Reparameterize the model or increase `target_accept`.

---

## 7. Advanced Topics & Extensions

### 7.1 Time-Varying Coefficients

*   **Concept:** TV effectiveness is not constant. It was higher during the Olympics.
*   **Gaussian Random Walk:** Allow $\beta_{tv}$ to drift slowly over time.

### 7.2 Carryover Effects (Adstock) Estimation

*   Instead of fixing Decay $\lambda = 0.5$, we let the model learn $\lambda$.
*   *Warning:* This makes the model highly non-linear and harder to sample.

---

## 8. Regulatory & Governance Considerations

### 8.1 Model Explainability

*   **Stakeholder:** The CFO asks "Why did you move \$2M to YouTube?"
*   **Answer:** "Because the Marginal ROAS of TV dropped below 1.0, and the Bayesian posterior for YouTube shows high potential upside."

---

## 9. Practical Example

### 9.1 Worked Example: The "Geo-Lift" Calibration

**Scenario:**
*   **MMM Uncalibrated:** Says Facebook ROAS is 4.0.
*   **Experiment:** We turn off FB in "Ohio" (Control) and keep it in "Michigan" (Test).
*   **Result:** CausalImpact shows Incremental ROAS is actually 1.5.
*   **Action:**
    *   Update the Bayesian Prior for FB ROAS to `Normal(1.5, 0.2)`.
    *   Re-run MMM.
    *   New Result: FB ROAS is 1.6. (Much closer to reality).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Bayesian MMM** fuses Data + Business Logic.
2.  **Calibration** anchors the model to reality.
3.  **Optimization** turns insights into money.

### 10.2 When to Use This Knowledge
*   **High Stakes:** When allocating \$50M+ budgets.
*   **Complex Funnels:** When attribution is broken.

### 10.3 Critical Success Factors
1.  **Patience:** MCMC sampling takes time.
2.  **Skepticism:** Always validate with experiments.

### 10.4 Further Reading
*   **Google:** "Challenges and Opportunities in Media Mix Modeling".
*   **PyMC3 Documentation**.

---

## Appendix

### A. Glossary
*   **Prior:** Initial belief.
*   **Posterior:** Updated belief.
*   **Likelihood:** Data fit.
*   **MCMC:** Markov Chain Monte Carlo.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Bayes Theorem** | $P(\theta|D) \propto P(D|\theta)P(\theta)$ | Inference |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
