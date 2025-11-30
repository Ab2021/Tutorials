# Customer Lifetime Value (Part 2) - Cross-Selling & Up-Selling - Theoretical Deep Dive

## Overview
"The easiest sale is to an existing customer."
Increasing the **Policies Per Household (PPH)** is the fastest way to increase CLV and Retention.
A customer with Auto + Home + Life is 3x less likely to churn than a monoline Auto customer.
This day focuses on **Market Basket Analysis**, **Association Rules**, and **Propensity Modeling** for Cross-Sell.

---

## 1. Conceptual Foundation

### 1.1 Cross-Sell vs. Up-Sell

*   **Cross-Sell:** Selling a *different* product. (Auto $\to$ Home).
*   **Up-Sell:** Selling a *better* version of the same product. (Liability Only $\to$ Full Coverage, or increasing limits).
*   **Goal:** Share of Wallet.

### 1.2 The "Sticky" Customer

*   **Theory:** Multi-line customers face higher **Switching Costs**.
    *   "It's a hassle to move three policies."
*   **Data:** Retention rates correlate linearly with policy count.

---

## 2. Mathematical Framework

### 2.1 Association Rules (Apriori)

*   **Concept:** "If {Auto, Home}, then {Umbrella}."
*   **Metrics:**
    *   **Support:** $P(A \cap B)$. How frequent is the combination?
    *   **Confidence:** $P(B | A)$. If they have A, how likely is B?
    *   **Lift:** $\frac{P(B|A)}{P(B)}$. How much more likely is B given A, compared to random chance?

### 2.2 Collaborative Filtering (Item-Based)

*   **Similarity:** Calculate similarity between products based on co-occurrence.
*   **Cosine Similarity:**
    $$ \text{sim}(i, j) = \frac{\vec{i} \cdot \vec{j}}{||\vec{i}|| \cdot ||\vec{j}||} $$
    Where vectors represent customer purchase history.

---

## 3. Theoretical Properties

### 3.1 The "Next Best Product" Propensity

*   **Model:** Multi-class Classification (or multiple Binary Classifiers).
*   **Target:** Which product will they buy next?
*   **Features:**
    *   **Life Stage:** Age 30 (Renters), Age 35 (Home), Age 40 (Life).
    *   **Asset Wealth:** Home Value (Umbrella).
    *   **Risk Profile:** Safe driver (Telematics).

### 3.2 Cannibalization

*   **Risk:** Up-selling a deductible buyback might increase claims frequency.
*   **Net Value:** Ensure the Cross-sell adds *Profit*, not just Revenue.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Market Basket Analysis (MLxtend)

```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Data: One-hot encoded transactions
# CustomerID | Auto | Home | Life | Pet
# 1          | 1    | 1    | 0    | 0

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(rules[['antecedents', 'consequents', 'lift', 'confidence']])
# Result: {Home} -> {Flood} Lift=5.0
```

### 4.2 Propensity Model (LightGBM)

```python
import lightgbm as lgb

# Target: Bought_Umbrella (Binary)
# Features: Home_Limit, Auto_Limit, Income, ZIP_Wealth

train_data = lgb.Dataset(X, label=y)
params = {'objective': 'binary', 'metric': 'auc'}
model = lgb.train(params, train_data)

# Predict
preds = model.predict(X_new)
```

---

## 5. Evaluation & Validation

### 5.1 Conversion Rate

*   **Metric:** Of the top 1000 customers targeted for "Pet Insurance", how many bought it?
*   **Benchmark:** Compare to a random sample.
*   **Lift:** Conversion(Target) / Conversion(Random).

### 5.2 PPH Growth

*   **Metric:** Average Policies Per Household over time.
*   **Goal:** Move from 1.5 to 2.0.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Timing is Everything

*   **Pitfall:** Offering "Homeowners Insurance" to someone who just renewed their Renters policy.
*   **Trigger:** Use external data (Zillow, MLS) to detect "New Home Purchase" events. That is the *moment* to cross-sell.

### 6.2 Channel Conflict

*   **Issue:** Agents own the relationship. They hate "Direct Marketing" emails to their clients.
*   **Solution:** "Next Best Action" for the *Agent*. Give the lead to the agent, don't bypass them.

---

## 7. Advanced Topics & Extensions

### 7.1 Sequence Mining

*   **Algorithm:** GSP (Generalized Sequential Patterns).
*   **Insight:** Order matters.
    *   Renters $\to$ Auto $\to$ Home (Common).
    *   Home $\to$ Renters (Rare - Downsizing).
*   **Use:** Predict the *path* of the customer journey.

### 7.2 Bundle Optimization

*   **Problem:** What discount to offer for the bundle?
*   **Economics:** The discount should be less than the Retention Benefit + Acquisition Savings.

---

## 8. Regulatory & Governance Considerations

### 8.1 Tying & Bundling

*   **Regulation:** In some jurisdictions, you cannot *force* a customer to buy Product B to get Product A (Tying).
*   **Allowed:** You *can* offer a discount for buying both (Bundling).

---

## 9. Practical Example

### 9.1 The "Life Event" Trigger

**Scenario:** Customer adds a "Teenful Driver" to their Auto policy.
**Insight:** This is a high-risk event for Auto, but a *trigger* for Life Insurance (Parents worry about protection).
**Action:**
1.  **Trigger:** New driver added age 16.
2.  **Offer:** "Protect your family's future with Term Life."
**Result:** 5% conversion rate (vs 0.5% baseline).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Lift** measures the strength of association.
2.  **Triggers** (Life Events) drive timing.
3.  **Bundling** creates stickiness.

### 10.2 When to Use This Knowledge
*   **Sales:** Scripting for agents.
*   **Marketing:** Email campaigns.

### 10.3 Critical Success Factors
1.  **Product Relevance:** Don't sell Snowmobile insurance in Florida.
2.  **Ease of Purchase:** "One-click" add-on is best. If they have to fill out a 20-page app, they won't buy.

### 10.4 Further Reading
*   **Agrawal et al.:** "Fast Algorithms for Mining Association Rules".
*   **Knott et al.:** "Next Best Product Models".

---

## Appendix

### A. Glossary
*   **Share of Wallet:** % of a customer's total insurance spend that goes to you.
*   **Monoline:** Customer with only 1 policy.
*   **Multiline:** Customer with >1 policy.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Lift** | $\frac{P(A \cap B)}{P(A)P(B)}$ | Association Strength |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
