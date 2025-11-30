# Customer Segmentation (Part 2) - Advanced Techniques & RFM - Theoretical Deep Dive

## Overview
"Demographics tell you *who* they are. Behavior tells you *what* they do."
Building on Day 94's clustering fundamentals, Day 95 dives into **Advanced Segmentation Techniques**.
We focus on **RFM Analysis** (Recency, Frequency, Monetary), **Behavioral Segmentation**, and **Psychographic Profiling**.
These techniques move beyond static attributes to capture the dynamic relationship between the customer and the insurer.

---

## 1. Conceptual Foundation

### 1.1 RFM Analysis in Insurance

*   **Recency (R):** How recently did they interact? (Claim, Login, Payment).
    *   *Insight:* High Recency = Engaged (or having a problem).
*   **Frequency (F):** How often do they interact?
    *   *Insight:* High Frequency of *payments* is normal. High Frequency of *calls* is a cost driver.
*   **Monetary (M):** How much value do they bring? (Premium - Claims).
    *   *Insight:* High Monetary = VIP.

### 1.2 Behavioral vs. Psychographic

*   **Behavioral:** Observed actions. "Logged in 5 times." "Added a driver."
*   **Psychographic:** Inferred motivations. "Risk Averse." "Price Sensitive." "Tech Savvy."
*   **The Link:** We use Behavior to infer Psychographics. (e.g., "High Deductible" behavior -> "Risk Tolerant" psychographic).

---

## 2. Mathematical Framework

### 2.1 RFM Scoring

*   **Method:** Quintile Scoring.
    1.  Rank all customers by Recency. Split into 5 groups (5=Best, 1=Worst).
    2.  Repeat for Frequency and Monetary.
    3.  Concatenate scores: "555" is a Champion. "111" is Lost.
*   **Insurance Adaptation:**
    *   *Monetary* should be **Profitability**, not just Premium.
    *   *Frequency* should distinguish between "Good Contacts" (buying) and "Bad Contacts" (complaining).

### 2.2 Latent Class Analysis (LCA)

*   **Use Case:** Segmentation with categorical data (e.g., Policy Type, Gender, Payment Method).
*   **Model:** Probabilistic model that assumes a latent "Class" variable generates the observed categorical patterns.
*   **Formula:** $P(Y|C) = \prod P(Y_j | C)$. (Independence assumption given class).

---

## 3. Theoretical Properties

### 3.1 The "Moveable Middle"

*   **Concept:** Segmentation often reveals a large "Middle" group that is neither loyal nor churning.
*   **Strategy:** This is the target for **Uplift Modeling**. Small nudges can move them to the "Loyal" segment.

### 3.2 Stability of Behavioral Segments

*   **Volatility:** Behavior changes faster than demographics. A "Safe Driver" becomes a "High Risk" driver overnight after a DUI.
*   **Implication:** Behavioral segments must be recalculated **daily** or **weekly**, whereas demographic segments can be static for months.

---

## 4. Modeling Artifacts & Implementation

### 4.1 RFM Implementation in Python

```python
import pandas as pd

# Data: CustomerID, TransactionDate, Amount
current_date = pd.to_datetime('today')

rfm = df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (current_date - x.max()).days, # Recency
    'TransactionID': 'count', # Frequency
    'Amount': 'sum' # Monetary
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Quintile Scoring
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
```

### 4.2 Hierarchical Clustering (Dendrogram)

*   **Tool:** `scipy.cluster.hierarchy`.
*   **Visual:** A tree diagram showing how customers merge into groups.
*   **Use:** Determining the optimal number of clusters visually.

---

## 5. Evaluation & Validation

### 5.1 The "Migrator" Analysis

*   **Method:** Track customers who moved from Segment A to Segment B over 6 months.
*   **Insight:**
    *   "New Parents" -> "Suburban Homeowners". (Natural progression).
    *   "Loyalists" -> "Price Shoppers". (Warning sign of churn).

### 5.2 Profitability Profiling

*   **Check:** Calculate the Loss Ratio for each segment.
*   **Goal:** Ensure that "High Value" segments actually have low Loss Ratios. If "VIPs" are unprofitable, your definition of "Value" is wrong.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Zero" Problem in Frequency

*   **Issue:** In Insurance, the ideal customer has *Zero* claims.
*   **RFM Fail:** Standard RFM treats "High Frequency" as good. For claims, High Frequency is bad.
*   **Fix:** Invert the Frequency score for Claims. Or separate "Purchase Frequency" (Good) from "Claim Frequency" (Bad).

### 6.2 Seasonality

*   **Issue:** "Recency" looks bad in Q1 if everyone renews in Q4.
*   **Fix:** Use "Days Since Renewal" or "Days Since Expected Interaction" instead of raw date.

---

## 7. Advanced Topics & Extensions

### 7.1 Sequence Mining (SPMF)

*   **Concept:** finding patterns in the *order* of events.
*   **Pattern:** "Add Car" -> "Add Teen Driver" -> "Increase Liability Limit".
*   **Segment:** "Growing Families".

### 7.2 Graph-Based Segmentation

*   **Concept:** Customers are nodes. Household/Referral links are edges.
*   **Community Detection:** Louvain algorithm finds "Households" or "Communities" of risk.

---

## 8. Regulatory & Governance Considerations

### 8.1 Behavioral Pricing

*   **Risk:** Using "Web Browsing History" or "Time of Day" for pricing (Price Optimization).
*   **Regulation:** Banned in many states (e.g., UK FCA ban on "Walking Tax").
*   **Rule:** Segmentation for *Marketing* is flexible. Segmentation for *Pricing* is strictly regulated.

---

## 9. Practical Example

### 9.1 The "Digital Engagement" Segments

**Goal:** Drive App Adoption.
**Features:** App Logins, Paperless Status, Email Opens.
**Segments:**
1.  **"Analog Anchors":** Paper bills, calls agent, 0 logins. (Costly).
2.  **"Hybrid Hoppers":** Gets email, but calls to pay. (Convertible).
3.  **"Digital Devotees":** App only, auto-pay. (Efficient).
**Strategy:**
*   **Analog:** Leave them alone (too hard to convert).
*   **Hybrid:** Email campaign: "Save \$5 by switching to Auto-Pay".
*   **Digital:** Push "Refer a Friend" offers.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **RFM** must be adapted for Insurance (Claims vs. Purchases).
2.  **Behavior** drives Psychographics.
3.  **Migration** analysis reveals lifecycle trends.

### 10.2 When to Use This Knowledge
*   **Retention:** Identifying "At-Risk" segments before they cancel.
*   **Operations:** Routing "High Maintenance" segments to specialized agents.

### 10.3 Critical Success Factors
1.  **Data Freshness:** Behavioral segments rot quickly.
2.  **Business Alignment:** Operations must be ready to treat segments differently.

### 10.4 Further Reading
*   **Kotler:** "Marketing Management" (Segmentation chapters).
*   **Berry & Linoff:** "Data Mining Techniques" (RFM section).

---

## Appendix

### A. Glossary
*   **Quintile:** Dividing a population into 5 equal groups.
*   **Dendrogram:** Tree diagram for hierarchical clustering.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **RFM Score** | $R \times 100 + F \times 10 + M$ | Customer Ranking |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
