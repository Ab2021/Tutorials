# Marketing Mix Modelling (MMM) (Part 1) - Theoretical Deep Dive

## Overview
"Half the money I spend on advertising is wasted; the trouble is I don't know which half." - John Wanamaker.
In Insurance, Customer Acquisition Cost (CAC) is a critical metric. **Marketing Mix Modelling (MMM)** is the actuarial science of marketing—using statistical models to quantify the impact of TV, Google Ads, and Billboards on Policy Sales.

---

## 1. Conceptual Foundation

### 1.1 The Attribution Problem

*   **Scenario:** A customer sees a TV ad, then a Facebook ad, then searches "Auto Insurance" on Google, clicks an ad, and buys.
*   **Last Click Attribution:** Google gets 100% credit. (Wrong, because TV started the journey).
*   **Multi-Touch Attribution (MTA):** Tracks individual user cookies. (Accurate but dying due to Privacy/Cookie death).
*   **MMM:** Top-down statistical regression. (Privacy-safe, robust).

### 1.2 Core Components of MMM

1.  **Base Sales:** Sales that would happen with *zero* advertising (Brand equity, Word of Mouth).
2.  **Incremental Sales:** Sales driven by marketing.
3.  **Control Variables:** Seasonality, Competitor Price, Economic conditions.

### 1.3 Adstock (The Memory Effect)

*   **Concept:** If you see a TV ad today, you might buy insurance next week. The effect "decays" over time.
*   **Geometric Adstock:** The most common decay function.
    *   $A_t = X_t + \lambda A_{t-1}$
    *   $\lambda$: Decay rate (0 to 1). High $\lambda$ means long memory (Brand building). Low $\lambda$ means immediate response (Call to Action).

### 1.4 Saturation (Diminishing Returns)

*   **Concept:** Spending \$1M on TV is good. Spending \$100M is not 100x better. You eventually reach everyone.
*   **Hill Function:** S-shaped curve.
    *   Initial spend has low impact (Threshold).
    *   Middle spend has high impact (Linear).
    *   High spend has low impact (Saturation).

---

## 2. Mathematical Framework

### 2.1 The Additive Model

$$ Y_t = \alpha + \sum_{i=1}^{N} \beta_i \text{Hill}( \text{Adstock}(X_{i,t}) ) + \gamma Z_t + \epsilon_t $$

*   $Y_t$: Sales at time $t$.
*   $\alpha$: Base Sales (Intercept).
*   $X_{i,t}$: Spend on Channel $i$ (TV, FB, Search).
*   $Z_t$: Control variables (Seasonality, Price).
*   $\beta_i$: Effectiveness of Channel $i$.

### 2.2 The Hill Function Equation

$$ \text{Hill}(x) = \frac{1}{1 + (\frac{K}{x})^S} $$

*   $K$: Half-saturation point (Spend level where you get 50% of max impact).
*   $S$: Shape parameter (Slope of the S-curve).

### 2.3 ROI and ROAS

*   **ROAS (Return on Ad Spend):** Revenue / Spend.
*   **mROAS (Marginal ROAS):** The derivative of the Response Curve.
    *   *Actuarial Insight:* We care about mROAS. If mROAS < 1.0, stop spending, even if average ROAS is high.

---

## 3. Theoretical Properties

### 3.1 Identification Strategy

*   **Collinearity:** TV spend and Facebook spend often move together (Campaigns).
*   **Ridge Regression:** Used to handle multicollinearity.
*   **Bayesian Priors:** We can inject prior knowledge (e.g., "TV cannot have a negative effect").

### 3.2 Privacy-First

*   MMM uses **Aggregated Data** (Weekly Spend vs. Weekly Sales).
*   It does *not* need User IDs or Cookies.
*   This makes MMM future-proof against GDPR and Apple's ATT (App Tracking Transparency).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Robyn (Meta's Library)

*   **Language:** R (mostly) and Python.
*   **Features:**
    *   Automated Hyperparameter Tuning (Evolutionary Algorithm).
    *   Multi-objective optimization (Minimize Error + Minimize Decomposition Distance).
*   **Output:** Pareto Front of models.

### 4.2 LightweightMMM (Google's Library)

*   **Language:** Python (JAX/NumPyro).
*   **Approach:** Bayesian.
*   **Key Feature:** Geo-level modeling. (Using data from 50 states increases sample size vs. national data).

### 4.3 Python Implementation (Simple Ridge)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

# 1. Adstock Function
def geometric_adstock(x, decay):
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t-1]
    return adstocked

# 2. Prepare Data
df = pd.read_csv('marketing_data.csv')
df['TV_Adstock'] = geometric_adstock(df['TV_Spend'], decay=0.7)
df['FB_Adstock'] = geometric_adstock(df['FB_Spend'], decay=0.3)

# 3. Train Model
X = df[['TV_Adstock', 'FB_Adstock', 'Competitor_Price']]
y = df['Policy_Sales']

model = Ridge(alpha=1.0)
model.fit(X, y)

print(f"TV Coefficient: {model.coef_[0]}")
```

---

## 5. Evaluation & Validation

### 5.1 Metrics

*   **MAPE (Mean Absolute Percentage Error):** Standard for forecasting.
*   **R-squared:** Goodness of fit.
*   **Decomposition Check:** Does the model say TV drove 90% of sales when we only spent 10% of budget on TV? (Unlikely).

### 5.2 Holdout Testing

*   **Time-based:** Train on Jan-Oct, Test on Nov-Dec.
*   **Geo-based:** Train on 40 states, Test on 10 states.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Confusing Correlation with Causation**
    *   "We spend more on Ads in December. We sell more in December."
    *   *Reality:* People buy in December anyway (Seasonality).
    *   *Fix:* Include "Seasonality" as a control variable.

2.  **Trap: Ignoring Lag**
    *   "We stopped TV ads yesterday, and sales didn't drop. TV is useless."
    *   *Reality:* Adstock effect keeps sales up for weeks.

### 6.2 Implementation Challenges

1.  **Data Granularity:**
    *   Daily data is noisy. Monthly data has too few rows.
    *   *Sweet Spot:* Weekly data.

2.  **Zero Spend:**
    *   If you never turn off ads, you can't measure the baseline.
    *   *Fix:* Geo-Lift Tests (Turn off ads in Ohio for 2 weeks).

---

## 7. Advanced Topics & Extensions

### 7.1 Hierarchical Bayesian MMM

*   Models parameters at the State level, shrinking them towards a National mean.
*   Allows for local variations (e.g., TV works better in Florida than New York).

### 7.2 Budget Optimization

*   **Input:** Total Budget = \$10M.
*   **Constraint:** TV Spend $\ge$ \$2M.
*   **Objective:** Maximize Sales.
*   **Solver:** `scipy.optimize` or Robyn's allocator.

---

## 8. Regulatory & Governance Considerations

### 8.1 Fair Lending

*   **Risk:** If the model says "Target Zip Code X" (which is minority-majority), is that Redlining?
*   **Marketing vs. Underwriting:** Marketing can target specific groups, but Underwriting cannot discriminate. The line is blurry.

---

## 9. Practical Example

### 9.1 Worked Example: The "Super Bowl" Ad

**Scenario:**
*   Insurer spends \$5M on a Super Bowl spot.
*   **Immediate Impact:** Website traffic spikes 500% on Sunday.
*   **Long-term Impact:** Brand awareness rises.
*   **MMM Analysis:**
    *   Adstock $\lambda = 0.8$ (Long decay).
    *   Attributed Sales: 5,000 policies over 3 months.
    *   CAC: \$1,000. (High).
*   **Decision:** "Prestige" play, not a direct response play.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Adstock** captures the "echo" of advertising.
2.  **Saturation** captures the "limit" of advertising.
3.  **MMM** is the source of truth for CMOs and CFOs.

### 10.2 When to Use This Knowledge
*   **Budgeting:** Deciding next year's marketing spend.
*   **Attribution:** When cookies disappear.

### 10.3 Critical Success Factors
1.  **Data Quality:** Garbage in, garbage out. Spend data must be accurate.
2.  **Experimentation:** Validate MMM results with Lift Tests.

### 10.4 Further Reading
*   **Jin et al. (Google):** "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects".

---

## Appendix

### A. Glossary
*   **CPP:** Cost Per Point (TV rating cost).
*   **GRP:** Gross Rating Point (Reach $\times$ Frequency).
*   **Impressions:** Number of times an ad was seen.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Geometric Adstock** | $x_t + \lambda A_{t-1}$ | Decay |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
