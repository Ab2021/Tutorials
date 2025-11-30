# Customer Lifetime Value (CLV) (Part 1) - Theoretical Foundations - Theoretical Deep Dive

## Overview
"It costs 5x more to acquire a new customer than to retain an existing one."
But not all customers are worth retaining.
**CLV (Customer Lifetime Value)** is the North Star metric that tells us exactly how much we should spend to acquire (CAC) and retain (Retention Cost) each individual.

---

## 1. Conceptual Foundation

### 1.1 What is CLV?

*   **Definition:** The present value of all future profits generated from a customer.
*   **Formula:**
    $$ \text{CLV} = \sum_{t=1}^{T} \frac{(R_t - C_t)}{(1 + d)^t} $$
    *   $R_t$: Revenue in period $t$ (Premium).
    *   $C_t$: Cost in period $t$ (Claims + Expenses).
    *   $d$: Discount Rate (e.g., 10%).
    *   $T$: Lifetime (Churn Date).

### 1.2 Historical vs. Predictive CLV

1.  **Historical CLV:** "How much *have* they spent?"
    *   *Use:* Reward programs (Gold Status).
    *   *Flaw:* Ignores future potential. A customer who just cancelled has High Historical CLV but Zero Predictive CLV.
2.  **Predictive CLV:** "How much *will* they spend?"
    *   *Use:* Acquisition bidding, Retention budgeting.
    *   *Method:* Probabilistic Models (BTYD).

---

## 2. Mathematical Framework

### 2.1 The "Buy 'Til You Die" (BTYD) Class

*   **Assumption:** Customers are "Alive" (Active) for a while, then they "Die" (Churn) and never come back (Non-contractual setting).
*   *Note:* Insurance is usually *contractual* (subscription), but BTYD models are still useful for *discretionary* purchases (Add-ons, Riders).

### 2.2 Pareto/NBD Model

*   **Transaction Process (NBD):** While alive, customers buy at a rate $\lambda$ (Poisson).
    *   $\lambda \sim Gamma(r, \alpha)$.
*   **Dropout Process (Pareto):** Customers die at a rate $\mu$ (Exponential).
    *   $\mu \sim Gamma(s, \beta)$.
*   **Result:** We can estimate $P(\text{Alive} | \text{History})$ and $E[\text{Transactions}]$.

### 2.3 BG/NBD (Beta-Geometric / Negative Binomial)

*   **Simplification:** Instead of continuous death risk, the customer flips a coin after every transaction.
    *   Heads = Stay Alive.
    *   Tails = Die (Churn).
*   **Pros:** Much faster to compute than Pareto/NBD.

---

## 3. Theoretical Properties

### 3.1 Gamma-Gamma Model (Monetary Value)

*   **Problem:** BG/NBD only predicts *frequency* (how many transactions). It doesn't predict *value* (\$ amount).
*   **Solution:** Gamma-Gamma.
    *   Assumption: The monetary value of transactions is independent of frequency.
    *   Assumption: Average transaction value is Gamma distributed across the population.
*   **Combined CLV:**
    $$ \text{CLV} = E[\text{Transactions}] \times E[\text{Value}] $$

### 3.2 The "Churn" Paradox in Insurance

*   In Retail, if you don't buy for 1 year, you might still be alive.
*   In Insurance, if you don't renew, you are **Dead** (Contractual Churn).
*   *Adaptation:* For the Core Policy, use **Survival Analysis** (Cox Proportional Hazards). For Add-ons, use **BG/NBD**.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Python Implementation (Lifetimes)

```python
from lifetimes import BetaGeoFitter, GammaGammaFitter

# 1. Data: RFM (Recency, Frequency, Monetary)
data = summary_data_from_transaction_data(df, 'customer_id', 'date', 'amount')

# 2. Fit BG/NBD (Frequency)
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])

# 3. Fit Gamma-Gamma (Monetary)
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(data['frequency'], data['monetary_value'])

# 4. Predict CLV
clv = ggf.customer_lifetime_value(
    bgf,
    data['frequency'],
    data['recency'],
    data['T'],
    data['monetary_value'],
    time=12, # Months
    discount_rate=0.01
)
print(clv.head())
```

### 4.2 Survival Analysis (Cox PH)

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='tenure', event_col='churned')
cph.print_summary()
```

---

## 5. Evaluation & Validation

### 5.1 Holdout Validation

*   **Method:**
    1.  Train on data from 2020-2022.
    2.  Predict transactions for 2023.
    3.  Compare Predicted vs. Actual.
*   **Metric:** RMSE or MAE of the aggregate transactions.

### 5.2 Decile Analysis

*   Rank customers by Predicted CLV.
*   Check if Top Decile actually generated the most revenue in the holdout period.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Ignoring Claims Cost**
    *   *Scenario:* A customer pays \$10k premium but files \$50k in claims.
    *   *Revenue CLV:* High.
    *   *Profit CLV:* Negative.
    *   *Fix:* Always model **Profit**, not Revenue. Use `Predicted_Premium - Predicted_Claims`.

2.  **Trap: The "Discount Rate" Debate**
    *   *Scenario:* Marketing uses 0% (to justify spending). Finance uses 15% (Cost of Capital).
    *   *Result:* Chaos.
    *   *Fix:* Align on a corporate standard (usually WACC ~ 8-10%).

---

## 7. Advanced Topics & Extensions

### 7.1 Deep CLV (RNNs/LSTMs)

*   **Idea:** Feed the sequence of customer interactions (Login, Call, Claim) into an LSTM.
*   **Output:** Predict time-to-next-event and value-of-next-event.
*   **Pros:** Captures temporal patterns (e.g., "Customers who call twice in a week usually churn").

### 7.2 Macro-Economic CLV

*   Adjusting CLV for Inflation and Interest Rates.
*   *Relevance:* In High-Inflation environments, future premiums are worth less.

---

## 8. Regulatory & Governance Considerations

### 8.1 Discrimination via CLV

*   **Risk:** You offer better service to High CLV customers.
*   **Problem:** If High CLV correlates with Race/Gender, you are violating Fair Access laws.
*   **Rule:** Service levels must be based on objective business criteria, not protected classes.

---

## 9. Practical Example

### 9.1 Worked Example: The "Whale" Hunter

**Scenario:**
*   **Data:** 1 Million customers.
*   **Model:** BG/NBD + Gamma-Gamma.
*   **Finding:**
    *   Top 1% of customers contribute 20% of future value.
    *   Bottom 50% are break-even.
*   **Action:**
    *   **VIP Team:** Call the Top 1% monthly. (Retention Budget = \$500/yr).
    *   **Self-Service:** Push the Bottom 50% to the App. (Retention Budget = \$5/yr).
*   **Result:** Retention of Top 1% increases by 5%. Net Profit +$10M.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **CLV** = Future Profit, discounted.
2.  **BG/NBD** predicts Frequency.
3.  **Gamma-Gamma** predicts Value.

### 10.2 When to Use This Knowledge
*   **Marketing:** Setting CAC limits (CAC < CLV / 3).
*   **Product:** Prioritizing features for High Value segments.

### 10.3 Critical Success Factors
1.  **Data Depth:** You need at least 3 repeat transactions for BTYD models to work well.
2.  **Profit Focus:** Revenue is vanity, Profit is sanity.

### 10.4 Further Reading
*   **Fader & Hardie:** "Probability Models for Customer-Base Analysis".

---

## Appendix

### A. Glossary
*   **CAC:** Customer Acquisition Cost.
*   **WACC:** Weighted Average Cost of Capital.
*   **RFM:** Recency, Frequency, Monetary.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **CLV** | $\sum (R-C)/(1+d)^t$ | Valuation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
