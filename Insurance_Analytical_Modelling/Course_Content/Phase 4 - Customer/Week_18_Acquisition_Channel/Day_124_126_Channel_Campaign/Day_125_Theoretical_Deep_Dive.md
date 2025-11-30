# Channel & Campaign Optimization (Part 2) - A/B Testing & Experimentation - Theoretical Deep Dive

## Overview
"If you are not failing, you are not learning."
In Insurance, a 1% improvement in Conversion Rate can mean \$10M in revenue.
But random changes are dangerous. We need **Scientific Experimentation**.
Today, we move beyond "Frequentist A/B Testing" to **Bayesian Methods** and **Multi-Armed Bandits**.

---

## 1. Conceptual Foundation

### 1.1 The Experimentation Lifecycle

1.  **Hypothesis:** "Changing the CTA from 'Get Quote' to 'See Prices' will increase CTR."
2.  **Design:** Power Analysis (Sample Size).
3.  **Execution:** Randomly split traffic (50/50).
4.  **Analysis:** Calculate P-Value or Posterior Probability.
5.  **Decision:** Roll out or Roll back.

### 1.2 Frequentist vs. Bayesian

*   **Frequentist:** "If the Null Hypothesis (No Difference) were true, how likely is this data?" (P-Value).
    *   *Problem:* Hard to interpret. "P=0.04" does NOT mean "96% chance it's better."
*   **Bayesian:** "Given the data, what is the probability that Variant B is better than Variant A?"
    *   *Benefit:* Direct answer to the business question. "There is a 99% chance B is better."

---

## 2. Mathematical Framework

### 2.1 Sample Size Calculation (Frequentist)

$$ n = \frac{16 \sigma^2}{\delta^2} $$

*   **Scenario:** Baseline Conversion = 5%. MDE (Min Detectable Effect) = 0.5% (Relative 10%).
*   **Result:** You need 30,000 visitors per variant.
*   *Constraint:* If you only get 1,000 visitors/day, the test takes 2 months.

### 2.2 Bayesian Inference (Beta-Binomial)

*   **Prior:** $Beta(\alpha, \beta)$. (Belief before the test).
*   **Likelihood:** Binomial (Successes, Failures).
*   **Posterior:** $Beta(\alpha + \text{Successes}, \beta + \text{Failures})$.
*   **Expected Loss:** How much do we lose if we pick the wrong variant?

---

## 3. Theoretical Properties

### 3.1 Multi-Armed Bandits (MAB)

*   **Problem:** A/B testing wastes traffic on the "Loser" (50% of traffic sees the bad version).
*   **Solution:** **Thompson Sampling**.
    *   Dynamically shift traffic to the winner *during* the test.
    *   *Regret Minimization:* Maximize total conversions while learning.
*   **Use Case:** Short-term campaigns (e.g., "Black Friday Sale"). You don't have time for a 2-week A/B test.

### 3.2 SRM (Sample Ratio Mismatch)

*   **Symptom:** You targeted a 50/50 split, but got 48/52.
*   **Cause:** Bug in the randomization logic.
*   **Result:** The test is invalid. (The 2% missing users might be a specific segment, e.g., Mobile users).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Python Implementation (Bayesian Test)

```python
import numpy as np
from scipy.stats import beta

# 1. Data
visitors_A, conversions_A = 1000, 50
visitors_B, conversions_B = 1000, 60

# 2. Posterior Distributions (Flat Prior Beta(1,1))
posterior_A = beta(1 + conversions_A, 1 + (visitors_A - conversions_A))
posterior_B = beta(1 + conversions_B, 1 + (visitors_B - conversions_B))

# 3. Simulation (Monte Carlo)
samples = 100000
sample_A = posterior_A.rvs(samples)
sample_B = posterior_B.rvs(samples)

prob_B_better = np.mean(sample_B > sample_A)
print(f"Probability B is better: {prob_B_better:.2%}")
# Output: ~85%
```

### 4.2 Thompson Sampling Logic

```python
def choose_arm(arms):
    # Draw a sample from each arm's Beta distribution
    samples = [beta(a.alpha, a.beta).rvs() for a in arms]
    # Pick the arm with the highest sample
    return np.argmax(samples)
```

---

## 5. Evaluation & Validation

### 5.1 A/A Testing

*   **Method:** Run a test where Variant A and Variant B are *identical*.
*   **Goal:** P-Value should be uniform. If you find a "Significant Difference", your platform is broken.

### 5.2 Novelty Effect

*   **Phenomenon:** Users click the new button just because it's new.
*   **Impact:** Conversion spikes on Day 1, then drops.
*   **Fix:** Run the test for at least 2 business cycles (2 weeks).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Peeking**
    *   *Scenario:* You check the P-Value every day. On Day 3, it hits 0.04. You stop the test.
    *   *Problem:* False Positive Rate explodes.
    *   *Fix:* Decide the sample size in advance and stick to it (or use Sequential Testing).

2.  **Trap: Interference**
    *   *Scenario:* Test A changes the Price. Test B changes the Email.
    *   *Problem:* Users in Test A behave differently in Test B.
    *   *Fix:* Orthogonal Layering or Mutually Exclusive Experiments.

---

## 7. Advanced Topics & Extensions

### 7.1 Stratified Sampling (CUPED)

*   **CUPED (Controlled-Experiment Using Pre-Experiment Data):**
    *   Use historical data (Variance Reduction) to increase power.
    *   *Result:* You need 50% less traffic to reach significance.

### 7.2 Contextual Bandits

*   **Standard Bandit:** "Variant B is better for everyone."
*   **Contextual Bandit:** "Variant B is better for *Young People*. Variant A is better for *Old People*."
*   *Algorithm:* LinUCB (Linear Upper Confidence Bound).

---

## 8. Regulatory & Governance Considerations

### 8.1 Disparate Impact in Testing

*   **Risk:** Your "Optimized" pricing algorithm charges minorities more.
*   **Requirement:** Fairness constraints must be baked into the Bandit (Constrained Optimization).

---

## 9. Practical Example

### 9.1 Worked Example: The "Quote Flow" Optimization

**Scenario:**
*   **Baseline:** 5-page Quote Form. Conversion = 8%.
*   **Hypothesis:** "Removing the 'SSN' field will increase conversion."
*   **Experiment:**
    *   **Control:** Standard Form.
    *   **Variant:** No SSN (Price is estimated).
*   **Results:**
    *   Variant Conversion = 12% (+50% Lift).
    *   *But wait...*
    *   Variant *Bind Rate* (Final Purchase) = 2% (Lower quality leads).
*   **Lesson:** Optimize for the *End Metric* (Revenue), not the *Intermediate Metric* (Quote).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Power Analysis** prevents inconclusive tests.
2.  **Bayesian Methods** provide actionable probabilities.
3.  **Bandits** earn while they learn.

### 10.2 When to Use This Knowledge
*   **Product Management:** "Should we launch this feature?"
*   **Marketing:** "Which ad headline works best?"

### 10.3 Critical Success Factors
1.  **Culture:** It must be okay to fail. (Most experiments fail).
2.  **Platform:** You need a robust tool (Optimizely, VWO, or custom).

### 10.4 Further Reading
*   **Kohavi et al.:** "Trustworthy Online Controlled Experiments" (The Bible of A/B Testing).

---

## Appendix

### A. Glossary
*   **MDE:** Minimum Detectable Effect.
*   **SRM:** Sample Ratio Mismatch.
*   **CUPED:** Variance Reduction technique.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Beta PDF** | $x^{\alpha-1}(1-x)^{\beta-1}$ | Bayesian Prior/Posterior |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
