# AB Testing & Experimental Design (Part 2) - Advanced Designs & Bandits - Theoretical Deep Dive

## Overview
"Why wait for the p-value if you already know the winner?"
Traditional A/B testing is slow. It wastes traffic on losing variations until the test ends.
**Multi-Armed Bandits (MAB)** allow us to "Earn while we Learn", dynamically shifting traffic to the winner.
This day focuses on **Bandit Algorithms**, **Factorial Designs**, and **Sequential Testing**.

---

## 1. Conceptual Foundation

### 1.1 The Explore-Exploit Dilemma

*   **Explore:** Gather information about Option B (might be better).
*   **Exploit:** Send traffic to Option A (currently looks best).
*   **A/B Testing:** 100% Explore (during test) $\to$ 100% Exploit (after test).
*   **Bandit:** Smooth transition from Explore to Exploit.

### 1.2 Factorial Design (MVT)

*   **Scenario:** We want to test 3 Headlines and 2 Images.
*   **A/B/n:** Test all 6 combinations as separate groups. (Inefficient).
*   **Factorial:** Test Factors (Headline, Image) and their Interactions.
*   **Benefit:** Allows measuring "Does Image A work better with Headline B?"

---

## 2. Mathematical Framework

### 2.1 Thompson Sampling (Bayesian Bandit)

*   **Belief:** We model the conversion rate of each arm as a Beta distribution $Beta(\alpha, \beta)$.
    *   $\alpha$: Successes + 1.
    *   $\beta$: Failures + 1.
*   **Action:**
    1.  Sample a random value from each arm's Beta distribution.
    2.  Choose the arm with the highest sample.
    3.  Update the distribution based on the result.
*   **Result:** Probability Matching. If Arm A has 90% probability of being best, it gets 90% of traffic.

### 2.2 Sequential Probability Ratio Test (SPRT)

*   **Goal:** Stop the A/B test early if the result is obvious.
*   **Method:** Calculate the Likelihood Ratio $\Lambda_n$ after each observation.
*   **Boundaries:**
    *   If $\Lambda_n > A$: Reject $H_0$ (Winner found).
    *   If $\Lambda_n < B$: Accept $H_0$ (No difference).
    *   Else: Continue.

---

## 3. Theoretical Properties

### 3.1 Regret Minimization

*   **Regret:** The difference between "Total Reward of Optimal Strategy" and "Total Reward of Actual Strategy".
*   **A/B Regret:** Linear $O(N)$. (Regret grows constantly during the test).
*   **Bandit Regret:** Logarithmic $O(\log N)$. (Regret slows down as we converge).

### 3.2 Interaction Effects

*   **Assumption:** Factors are independent. (Headline doesn't affect Image).
*   **Reality:** Synergy. A "Funny" headline needs a "Funny" image.
*   **Factorial ANOVA:** Statistical test to detect these interactions.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Thompson Sampling (Python)

```python
import numpy as np
from scipy.stats import beta

class Bandit:
    def __init__(self):
        self.alpha = 1 # Prior Successes
        self.beta = 1  # Prior Failures

    def sample(self):
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward):
        if reward == 1:
            self.alpha += 1
        else:
            self.beta += 1

# Simulation
arm_A = Bandit() # True rate 0.10
arm_B = Bandit() # True rate 0.12

# Decision
if arm_A.sample() > arm_B.sample():
    # Show A, get reward...
    pass
```

### 4.2 Factorial Analysis (Statsmodels)

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Data: Conversion ~ Headline + Image + Headline:Image
model = ols('conversion ~ C(headline) + C(image) + C(headline):C(image)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
# Check p-value of interaction term
```

---

## 5. Evaluation & Validation

### 5.1 Offline Replay (Counterfactual Evaluation)

*   **Problem:** How to test a Bandit algorithm on historical data?
*   **Method:** Replay Method (Li et al.).
    *   Take a log of random traffic.
    *   If the Bandit chooses the *same* action as the log, keep the data point.
    *   If different, discard.
*   **Result:** Unbiased estimate of Bandit performance.

### 5.2 Stability

*   **Risk:** Bandits can be unstable in non-stationary environments (e.g., Weekend vs. Weekday).
*   **Fix:** Discounted Thompson Sampling (Forget old data).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Simpson's Paradox

*   **Scenario:** A is better than B on Weekdays. A is better than B on Weekends.
*   **Result:** But B is better than A overall?
*   **Cause:** Unequal sampling. If B was shown mostly on Weekends (high conversion time), it looks artificially good.
*   **Fix:** Stratification.

### 6.2 Delayed Feedback

*   **Issue:** In insurance, "Conversion" (Sale) happens days after the "Quote".
*   **Bandit Impact:** The Bandit thinks the arm produced 0 rewards (because of delay) and stops showing it.
*   **Fix:** Importance Sampling or waiting for maturity.

---

## 7. Advanced Topics & Extensions

### 7.1 Contextual Bandits

*   **Idea:** The best arm depends on the *User*.
*   **Model:** $Reward = f(Context, Action)$.
*   **Algorithm:** LinUCB (Linear Upper Confidence Bound).
*   **Use:** Personalization. "Show the 'Family' banner to users with kids."

### 7.2 Bayesian Optimization

*   **Context:** Continuous parameters (e.g., Price).
*   **Method:** Gaussian Process Regression to model the Reward function.
*   **Acquisition Function:** Expected Improvement (EI).

---

## 8. Regulatory & Governance Considerations

### 8.1 Explainability of Bandits

*   **Challenge:** "Why did the Bandit show this price?"
*   **Answer:** "Because it sampled a high value from the Beta distribution." (Not very satisfying to a regulator).
*   **Constraint:** Bandits are rarely used for *Pricing* in regulated lines. Mostly for Marketing/UX.

---

## 9. Practical Example

### 9.1 The "Headline" Bandit

**Scenario:** Testing 5 email subject lines.
**Method:** Thompson Sampling.
**Day 1:** Random traffic.
**Day 2:** Subject B gets 50% clicks. Bandit shifts 60% traffic to B.
**Day 3:** Subject B holds up. Bandit shifts 90% traffic to B.
**Result:** The campaign converged to the winner in 3 days, maximizing total clicks. A/B test would have wasted 80% of traffic on losers for 2 weeks.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Bandits** minimize regret.
2.  **Factorial Design** finds interactions.
3.  **Contextual Bandits** enable personalization.

### 10.2 When to Use This Knowledge
*   **High Volume:** Bandits need data fast.
*   **Short Lifespan:** Marketing campaigns (Black Friday).

### 10.3 Critical Success Factors
1.  **Engineering:** Bandits require real-time feedback loops.
2.  **Safety:** Put "Guardrails" on the bandit (e.g., min/max price).

### 10.4 Further Reading
*   **Sutton & Barto:** "Reinforcement Learning: An Introduction".
*   **Scott:** "A Modern Bayesian Look at the Multi-armed Bandit".

---

## Appendix

### A. Glossary
*   **Arm:** A variation (Treatment).
*   **Horizon:** The duration of the problem.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Beta Mean** | $\frac{\alpha}{\alpha + \beta}$ | Expected Rate |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
