# AB Testing & Experimental Design (Part 1) - Fundamentals & Hypothesis Testing - Theoretical Deep Dive

## Overview
"Correlation is not causation. Experimentation is."
In Days 94-96, we built models. But do they work?
**A/B Testing** (Randomized Controlled Trials) is the gold standard for validating pricing changes, marketing campaigns, and UX improvements.
This day focuses on **Hypothesis Testing**, **Sample Size Calculation**, and the **Statistics of Experimentation**.

---

## 1. Conceptual Foundation

### 1.1 Why Experiment?

*   **Observational Data:** Biased. "Customers who bought the bundle are profitable." (Maybe profitable customers just happen to buy the bundle).
*   **Experimental Data:** Unbiased. We *force* a random group to see the bundle.
*   **Causality:** $X \to Y$.

### 1.2 The A/B Framework

*   **Control (A):** The current state (Champion).
*   **Treatment (B):** The new idea (Challenger).
*   **Randomization:** The unit of randomization (User, Policy, Agent) is critical.
*   **Metric:** The success criteria (Conversion Rate, Loss Ratio, Retention).

---

## 2. Mathematical Framework

### 2.1 Hypothesis Testing

*   **Null Hypothesis ($H_0$):** $Rate_A = Rate_B$. (The change did nothing).
*   **Alternative Hypothesis ($H_1$):** $Rate_A \neq Rate_B$. (There is a difference).
*   **p-value:** The probability of seeing this difference by random chance if $H_0$ were true.
*   **Significance Level ($\alpha$):** Usually 0.05. If $p < 0.05$, we reject $H_0$.

### 2.2 Power Analysis

*   **Power ($1-\beta$):** The probability of detecting a difference *if it exists*. Usually 0.80.
*   **Effect Size (MDE):** Minimum Detectable Effect. "We want to detect a 1% lift."
*   **Sample Size Formula:**
    $$ n \approx \frac{16 \sigma^2}{\delta^2} $$
    Where $\delta$ is the effect size. Smaller effect = Much larger sample needed.

---

## 3. Theoretical Properties

### 3.1 Randomization Unit

*   **User-Level:** Best for UX.
*   **Quote-Level:** Best for Pricing.
*   **Agent-Level:** Best for Commission changes.
*   **SUTVA (Stable Unit Treatment Value Assumption):** Treating User A should not affect User B. (Violated in Agent-level tests if agents talk).

### 3.2 Types of Errors

*   **Type I ($\alpha$):** False Positive. We say B is better, but it's not. (Cost: Deploying a bad feature).
*   **Type II ($\beta$):** False Negative. B was better, but we missed it. (Cost: Missed opportunity).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Power Calculation (Statsmodels)

```python
import statsmodels.stats.api as sms

# Calculate required sample size
# Effect size: 0.02 (2% lift from 10% to 10.2%)
effect_size = sms.proportion_effectsize(0.10, 0.102)

required_n = sms.NormalIndPower().solve_power(
    effect_size=effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
)

print(f"Required Sample Size per Group: {int(required_n)}")
```

### 4.2 Analyzing Results (T-Test)

```python
from scipy import stats

# Data: Conversion (0/1) for Group A and B
group_a = [0, 1, 0, ...]
group_b = [1, 1, 0, ...]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print(f"P-Value: {p_val:.4f}")
if p_val < 0.05:
    print("Significant Difference!")
else:
    print("No significant difference.")
```

---

## 5. Evaluation & Validation

### 5.1 A/A Testing

*   **Method:** Run a test where A and B are identical.
*   **Goal:** Check if the p-value distribution is uniform.
*   **Failure:** If A/A shows a "Significant Difference", your randomization engine is broken.

### 5.2 SRM (Sample Ratio Mismatch)

*   **Check:** If you split 50/50, did you actually get 50/50?
*   **Test:** Chi-Square test on the counts.
*   **Cause:** Latency (B takes longer to load, so users bounce before assignment).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Peeking

*   **Sin:** Checking the p-value every day and stopping when it's significant.
*   **Result:** Inflates Type I error. (You will find "significance" eventually by chance).
*   **Fix:** Fixed Horizon (Decide sample size in advance) or Sequential Testing (SPRT).

### 6.2 Novelty Effect

*   **Phenomenon:** Users click the new button just because it's new.
*   **Result:** Short-term lift, long-term regression.
*   **Fix:** Run the test long enough (e.g., 2 weeks) to let novelty wear off.

---

## 7. Advanced Topics & Extensions

### 7.1 Multi-Armed Bandits (MAB)

*   **Concept:** Explore and Exploit simultaneously.
*   **Algorithm:** Thompson Sampling.
*   **Benefit:** Minimizes "Regret" (wasted traffic on the losing variation) compared to A/B testing.
*   **Use:** Dynamic Pricing, Headline Optimization.

### 7.2 Switchback Testing

*   **Context:** Marketplaces (Uber/Lyft).
*   **Method:** Switch the *whole market* between A and B every hour.
*   **Why?** Handles network effects (SUTVA violations).

---

## 8. Regulatory & Governance Considerations

### 8.1 Pricing Experiments

*   **Risk:** Charging different prices to random people can be "Unfair Discrimination".
*   **Compliance:** Usually requires filing the "Test Plan" with the DOI.
*   **Limit:** Often limited to a small % of the book (e.g., 5%).

---

## 9. Practical Example

### 9.1 The "Deductible Nudge"

**Hypothesis:** Showing the \$1000 deductible as the default (instead of \$500) will increase conversion (lower price) without hurting profitability.
**Design:**
*   **Control:** Default \$500.
*   **Treatment:** Default \$1000.
*   **Metric:** Conversion Rate, Average Premium.
**Result:**
*   Conversion +5%.
*   Premium -2%.
*   **Net Profit:** +3%. (The volume lift outweighed the premium drop).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Randomization** eliminates bias.
2.  **Power** determines sample size.
3.  **P-value** measures surprise, not truth.

### 10.2 When to Use This Knowledge
*   **Product:** Testing new features.
*   **Marketing:** Testing email subject lines.

### 10.3 Critical Success Factors
1.  **Culture:** "We don't guess, we test."
2.  **Infrastructure:** You need a platform (Optimizely, LaunchDarkly) to manage assignments.

### 10.4 Further Reading
*   **Kohavi:** "Trustworthy Online Controlled Experiments".
*   **Box, Hunter, Hunter:** "Statistics for Experimenters".

---

## Appendix

### A. Glossary
*   **Lift:** $(Rate_B - Rate_A) / Rate_A$.
*   **Confidence Interval:** The range where the true lift likely lies.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Z-Score** | $\frac{p_B - p_A}{\sqrt{p(1-p)(2/n)}}$ | Proportion Test |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
