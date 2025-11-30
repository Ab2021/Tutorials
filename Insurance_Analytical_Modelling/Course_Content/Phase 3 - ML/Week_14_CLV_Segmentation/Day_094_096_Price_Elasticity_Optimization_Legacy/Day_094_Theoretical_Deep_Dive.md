# Price Elasticity & Optimization (Part 1) - Demand Modeling Fundamentals - Theoretical Deep Dive

## Overview
"Actuaries calculate the Cost. Data Scientists calculate the Price."
The **Technical Price** (Cost + Profit) is what we *need* to charge.
The **Market Price** is what the customer is *willing* to pay.
This day focuses on **Price Elasticity of Demand**, **Conversion Modeling**, and the gap between Cost and Price.

---

## 1. Conceptual Foundation

### 1.1 The Pricing Waterfall

1.  **Pure Premium:** Expected Loss (Frequency $\times$ Severity).
2.  **Technical Price:** Pure Premium + Expenses + Target Profit.
3.  **Street Price:** Technical Price adjusted for Competition and Elasticity.
    *   *Constraint:* Street Price $\ge$ Floor (Solvency).

### 1.2 Elasticity Defined

*   **Definition:** The % change in Demand for a 1% change in Price.
    $$ \epsilon = \frac{\% \Delta Q}{\% \Delta P} $$
*   **Inelastic ($\epsilon > -1$):** Raising price increases revenue. (e.g., Mandatory Auto Insurance).
*   **Elastic ($\epsilon < -1$):** Raising price decreases revenue. (e.g., Optional Pet Insurance).

---

## 2. Mathematical Framework

### 2.1 The Demand Function

We model the probability of purchase (Conversion) as a function of Price.
$$ P(Buy) = \frac{1}{1 + e^{-(\alpha + \beta \cdot Price + \gamma X)}} $$
*   **Logistic Regression:** The standard tool.
*   **Coefficient $\beta$:** Represents price sensitivity. Should be negative.
*   **Price:** Usually modeled as "Competitive Ratio" (My Price / Competitor Price).

### 2.2 Iso-Profit Curves

*   **Profit Equation:** $\pi = Q(P) \times (P - Cost)$.
*   **Optimization:** Find $P$ where $\frac{d\pi}{dP} = 0$.
*   **Condition:** $P^* = Cost \times \frac{\epsilon}{1+\epsilon}$. (The Inverse Elasticity Rule).

---

## 3. Theoretical Properties

### 3.1 The "Winner's Curse" in Pricing

*   **Scenario:** You and a Competitor bid on a risk. You bid \$100, they bid \$120.
*   **Result:** You win the customer.
*   **Risk:** Maybe you won because you *underestimated* the risk?
*   **Correction:** Incorporate "Competitive Position" into the risk model.

### 3.2 Asymmetric Elasticity

*   **New Business:** Highly elastic. People shop around.
*   **Renewal Business:** Inelastic. People are lazy (Inertia).
*   **Strategy:** "Dual Pricing" (Low intro rate, high renewal rate). *Note: Heavily regulated.*

---

## 4. Modeling Artifacts & Implementation

### 4.1 Logistic Demand Model (Python)

```python
import statsmodels.api as sm
import numpy as np

# Data: Quote_Price, Competitor_Price, Is_Bound (0/1)
df['price_ratio'] = df['quote_price'] / df['competitor_price']

X = df[['price_ratio', 'customer_age', 'credit_score']]
X = sm.add_constant(X)
y = df['is_bound']

# Logit Model
model = sm.Logit(y, X).fit()

print(model.summary())
# Look for negative coef on price_ratio
```

### 4.2 Calculating Elasticity

```python
# Elasticity = beta * Price * (1 - Probability)
beta = model.params['price_ratio']
price = 1.0 # At parity
prob = model.predict([1.0, 1.0, 30, 700]) # Example customer

elasticity = beta * price * (1 - prob)
print(f"Price Elasticity: {elasticity:.2f}")
```

---

## 5. Evaluation & Validation

### 5.1 The "Price Test" (Randomized Control Trial)

*   **Method:** Randomly perturb the price by $\pm 1\%$ for a small sample.
*   **Measure:** The actual change in conversion.
*   **Use:** Calibrate the Logistic Regression (which is observational) with real experimental data.

### 5.2 Off-Line Evaluation

*   **Metric:** AUC of the Conversion Model.
*   **Business Metric:** "Expected Revenue" on the test set.
    *   $\sum (Prob(Buy) \times Price)$.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Endogeneity

*   **Problem:** Prices are not random. High risk customers get high prices.
*   **Bias:** The model thinks "High Price = High Conversion" because we only charge high prices to people who *really* need insurance (high risk).
*   **Fix:** Instrumental Variables or RCTs (Price Tests).

### 6.2 Competitor Data Blindness

*   **Issue:** We rarely know the competitor's exact price.
*   **Proxy:** "Market Basket" scraping or Comparative Rater data (e.g., from aggregators).

---

## 7. Advanced Topics & Extensions

### 7.1 Non-Linear Elasticity

*   **Reality:** Elasticity is not constant.
    *   Small increase: No reaction.
    *   Big increase: Mass exodus.
*   **Model:** GAM (Generalized Additive Models) or Splines for the Price variable.

### 7.2 Portfolio Optimization

*   **Goal:** Maximize Portfolio Profit, subject to Volume constraints.
*   **Method:** Linear Programming (Simplex).
    *   Maximize $\sum \pi_i$
    *   Subject to $\sum Q_i \ge TargetVolume$.

---

## 8. Regulatory & Governance Considerations

### 8.1 Unfair Discrimination

*   **Rule:** You cannot charge two people with the same risk different prices just because one is "willing to pay more".
*   **US Regulation:** Most states ban "Price Optimization" based solely on elasticity.
*   **UK Regulation:** FCA ban on "Price Walking" (Renewal price cannot exceed New Business price).

---

## 9. Practical Example

### 9.1 The "Expense Loading" Optimization

**Scenario:** We have a fixed expense of \$100 per policy.
**Current:** We load 20% for expenses.
**Problem:** For small policies (\$200), 20% is \$40. We lose \$60 on every sale.
**Optimization:**
1.  **Change:** Switch to "Fixed Expense Constant" (\$100) + Variable Load.
2.  **Impact:** Price goes up for small policies (Demand drops). Price goes down for large policies (Demand rises).
3.  **Result:** We lose unprofitable volume and gain profitable volume.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Technical Price** is the floor.
2.  **Elasticity** determines the ceiling.
3.  **Conversion Models** link Price to Volume.

### 10.2 When to Use This Knowledge
*   **Pricing Committee:** Deciding on rate changes.
*   **Marketing:** Targeting price-sensitive segments.

### 10.3 Critical Success Factors
1.  **Competitor Intel:** You cannot optimize in a vacuum. You need to know the market rate.
2.  **Regulatory Compliance:** Always check if "Elasticity Pricing" is legal in your jurisdiction.

### 10.4 Further Reading
*   **Phillips:** "Pricing and Revenue Optimization".
*   **Casualty Actuarial Society:** "Price Optimization Monograph".

---

## Appendix

### A. Glossary
*   **Hit Ratio:** Conversion Rate (Policies Bound / Quotes Issued).
*   **Retention Ratio:** Policies Renewed / Policies Available to Renew.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Optimal Margin** | $-1 / \epsilon$ | Monopoly Pricing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
