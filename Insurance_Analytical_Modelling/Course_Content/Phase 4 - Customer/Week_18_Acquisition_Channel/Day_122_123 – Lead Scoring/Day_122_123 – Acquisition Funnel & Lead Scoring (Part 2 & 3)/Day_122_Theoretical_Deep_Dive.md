# Acquisition Funnel & Lead Scoring (Part 2) - Advanced Techniques - Theoretical Deep Dive

## Overview
"We know 50% of our marketing works. We just don't know which 50%."
Yesterday, we built a basic Lead Score. Today, we answer the harder questions:
1.  **Attribution:** Did the Facebook Ad or the Google Search cause the sale?
2.  **Zero-Inflation:** How do we model conversion when 98% of leads don't convert?
3.  **Uplift:** Who should we *not* call?

---

## 1. Conceptual Foundation

### 1.1 The Multi-Touch Attribution Problem

*   **The Journey:**
    1.  Day 1: User sees Instagram Ad (Awareness).
    2.  Day 3: User Googles "Best Insurance" (Consideration).
    3.  Day 7: User clicks "Retargeting Email" and buys (Conversion).
*   **Last Touch:** Email gets 100% credit. (Unfair to Instagram).
*   **First Touch:** Instagram gets 100% credit. (Unfair to Email).
*   **Markov Chain:** Calculates the *probability contribution* of each step.

### 1.2 The "Zero-Inflated" Reality

*   **Scenario:** You buy 10,000 leads. Only 200 buy (2%).
*   **Problem:** Standard Regression assumes the data is normally distributed (or Poisson). With 98% zeros, the model is biased towards predicting 0.
*   **Solution:** Zero-Inflated Models (ZIP / ZINB).

---

## 2. Mathematical Framework

### 2.1 Markov Chain Attribution

*   **States:** Start, Instagram, Google, Email, Conversion, Null (Churn).
*   **Transition Matrix ($T$):** Probability of moving from State A to State B.
*   **Removal Effect:**
    $$ \text{Effect}(C) = 1 - \frac{P(\text{Conversion} | \text{Channel } C \text{ removed})}{P(\text{Conversion})} $$
    *   If removing Instagram drops conversion probability from 2% to 1%, Instagram gets 50% credit.

### 2.2 Zero-Inflated Poisson (ZIP)

*   **Two Processes:**
    1.  **Bernoulli ($p$):** Is this a "Always Zero" lead? (e.g., Fake Phone Number).
    2.  **Poisson ($\lambda$):** If not, how many policies will they buy?
*   **Formula:**
    $$ P(Y=0) = \pi + (1-\pi)e^{-\lambda} $$
    $$ P(Y=k) = (1-\pi) \frac{\lambda^k e^{-\lambda}}{k!} $$

---

## 3. Theoretical Properties

### 3.1 The "Sleeping Dog" Hypothesis (Uplift)

*   **Quadrant:**
    1.  **Persuadables:** Buy only if treated. (Target these).
    2.  **Sure Things:** Buy anyway. (Don't waste money).
    3.  **Lost Causes:** Never buy. (Don't waste money).
    4.  **Sleeping Dogs:** Leave if treated. (Avoid!).
*   *Example:* Calling a customer might remind them to cancel their policy.

### 3.2 Shapley Value Attribution

*   **Game Theory:** Treat channels as players in a cooperative game.
*   **Value:** The average marginal contribution of a channel across all possible permutations of the journey.
*   *Pros:* Fairer than Markov.
*   *Cons:* Computationally expensive ($2^N$).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Python Implementation (Markov)

```python
import pandas as pd
from pychattr.channel_attribution import MarkovModel

# 1. Data: Path of Channels
df = pd.DataFrame({
    "path": ["Instagram > Google > Email", "Google > Email"],
    "conversion": [1, 0],
    "null": [0, 1]
})

# 2. Model
mm = MarkovModel(path_feature="path", conversion_feature="conversion", null_feature="null")
mm.fit(df)

# 3. Removal Effect
print(mm.removal_effects_)
# Output: {"Instagram": 0.5, "Google": 0.2, "Email": 0.3}
```

### 4.2 ZIP Regression (Statsmodels)

```python
import statsmodels.api as sm

# 1. Define Endog (Target) and Exog (Features)
model = sm.ZeroInflatedPoisson(y, X, exog_infl=X_infl)
result = model.fit()

print(result.summary())
```

---

## 5. Evaluation & Validation

### 5.1 Qini Curve (Uplift)

*   Similar to ROC Curve, but for Uplift.
*   **X-Axis:** % of population targeted.
*   **Y-Axis:** Incremental gains (Uplift).
*   *Goal:* Area Under Qini Curve (AUQC).

### 5.2 ROAS (Return on Ad Spend)

*   **Formula:** Revenue Attributed to Channel / Cost of Channel.
*   *Action:* If ROAS < 1, kill the channel.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Correlation is not Causation**
    *   *Scenario:* "People who click 'About Us' convert at 50%."
    *   *Action:* Force everyone to visit 'About Us'.
    *   *Result:* Conversion drops. They visited 'About Us' *because* they were interested, not the other way around.

2.  **Trap: The "Cookie" Problem**
    *   *Scenario:* User clicks Ad on Mobile, buys on Desktop.
    *   *Result:* Attribution breaks. Mobile gets 0 credit.
    *   *Fix:* Deterministic Identity Graph (Login required).

---

## 7. Advanced Topics & Extensions

### 7.1 Time-Decay Attribution

*   **Logic:** A click today is worth more than a click 30 days ago.
*   **Weight:** $W(t) = e^{-\lambda t}$.

### 7.2 Budget Optimization (Linear Programming)

*   **Objective:** Maximize Conversions.
*   **Constraints:**
    *   Total Budget $\le$ \$1M.
    *   Facebook Spend $\ge$ \$100k.
    *   CPA (Cost Per Acquisition) $\le$ \$500.
*   **Solver:** Scipy `linprog`.

---

## 8. Regulatory & Governance Considerations

### 8.1 GDPR & Tracking

*   **Rule:** You cannot track users across devices without consent (Cookie Banner).
*   **Impact:** Attribution data is becoming sparse (The "Cookiepocalypse").
*   **Solution:** Media Mix Modeling (MMM) - Top-down econometrics instead of bottom-up tracking.

---

## 9. Practical Example

### 9.1 Worked Example: Optimizing the Funnel

**Scenario:**
*   **Channel A (Facebook):** Last Touch CPA = \$800. (Looks bad).
*   **Channel B (Search):** Last Touch CPA = \$300. (Looks good).
*   **Action:** Fire the Facebook Agency?
*   **Markov Analysis:**
    *   Facebook is the "Opener". 80% of Search conversions *started* with Facebook.
    *   Removal Effect of Facebook = 60%.
    *   **True CPA:** Facebook = \$400. Search = \$400.
*   **Decision:** Keep Facebook. If you cut it, Search volume will dry up.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Attribution** assigns credit fairly.
2.  **Zero-Inflation** handles rare events.
3.  **Uplift** targets the persuadables.

### 10.2 When to Use This Knowledge
*   **CMO Dashboard:** "Where should I spend my next dollar?"
*   **Data Science:** Building the "Brain" of the marketing department.

### 10.3 Critical Success Factors
1.  **Data Linking:** Connecting the Click to the Policy Bind.
2.  **Experimentation:** Always run holdout groups.

### 10.4 Further Reading
*   **Google:** "Attribution Modeling in Google Analytics".

---

## Appendix

### A. Glossary
*   **ROAS:** Return on Ad Spend.
*   **CPA:** Cost Per Acquisition.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Removal Effect** | $1 - P(C|\neg X)/P(C)$ | Attribution |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
