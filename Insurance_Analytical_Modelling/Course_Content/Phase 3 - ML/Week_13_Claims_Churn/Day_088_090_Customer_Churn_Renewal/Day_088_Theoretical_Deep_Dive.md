# Customer Churn & Retention (Part 1) - Churn Prediction Models - Theoretical Deep Dive

## Overview
"It costs 5x more to acquire a new customer than to keep an old one."
In insurance, retention is the primary driver of profitability (due to high acquisition costs).
This day focuses on **Churn Prediction** (Binary Classification) and **Retention Modeling** (Survival Analysis).
We will explore why customers leave (Price, Service, Life Events) and how to stop them.

---

## 1. Conceptual Foundation

### 1.1 The Churn Definition

*   **Voluntary Churn:** Customer cancels (switches to Geico). **Target for ML.**
*   **Involuntary Churn:** Insurer cancels (Non-payment, Underwriting Risk). **Not a target.**
*   **Lapse:** Policy expires and is not renewed.

### 1.2 Drivers of Churn

1.  **Price Shock:** Renewal premium increased > 10%.
2.  **Service Failure:** Bad claims experience.
3.  **Life Event:** Moved state, sold car, got married.
4.  **Competitive Offer:** Competitor marketing.

---

## 2. Mathematical Framework

### 2.1 Binary Classification (Logistic / XGBoost)

*   **Target:** $Y=1$ if Churn, $Y=0$ if Renew.
*   **Features:**
    *   **Policy:** Tenure, Premium Change, Coverage Limits.
    *   **Behavior:** Calls to call center, Login to portal.
    *   **Demographics:** Age, Credit Score.
*   **Output:** Probability of Churn $P(Y=1|X)$.

### 2.2 Survival Analysis (Cox Proportional Hazards)

*   **Question:** *When* will they churn?
*   **Hazard Function:** $\lambda(t) = \lambda_0(t) e^{\beta X}$.
*   **Insight:** "Customers are most likely to churn at Month 12 (Renewal) and Month 6 (Mid-term)."
*   **Censoring:** Active customers are right-censored.

---

## 3. Theoretical Properties

### 3.1 The Price Elasticity of Churn

*   **Concept:** How sensitive is retention to price increases?
*   **Formula:** $\epsilon = \frac{\% \Delta \text{Retention}}{\% \Delta \text{Price}}$.
*   **Optimization:** We want to maximize $Price \times Retention(Price)$.
    *   If $\epsilon$ is low (inelastic), we can raise rates.
    *   If $\epsilon$ is high (elastic), we must discount.

### 3.2 The "U-Shape" of Tenure

*   **New Customers:** High churn (Price shoppers).
*   **Mid-Tenure:** Low churn (Loyal / Lazy).
*   **Old Customers:** Rising churn (Price walking / "Loyalty Penalty" realization).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Churn Prediction (XGBoost)

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Features: Tenure, Rate_Increase, Claims_Last_Year
X = df[['tenure', 'rate_increase_pct', 'claims_count']]
y = df['is_churn']

# Train
model = xgb.XGBClassifier(objective='binary:logistic')
model.fit(X_train, y_train)

# Feature Importance
print(model.feature_importances_)
# Usually 'rate_increase_pct' is #1
```

### 4.2 Survival Curve (Lifelines)

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='months_active', event_col='is_churn')

cph.print_summary()
# Look at exp(coef)
# If exp(coef) for 'rate_increase' is 1.5, then a 1 unit increase raises churn risk by 50%.
```

---

## 5. Evaluation & Validation

### 5.1 Lift Charts

*   **Method:** Decile the test set by Predicted Churn Probability.
*   **Check:** Does the top decile have 3x-5x the churn rate of the average?
*   **Actionability:** We can only afford to call the top 10%. The model must be accurate *at the top*.

### 5.2 The "Save Rate" Metric

*   **Metric:** Of the customers flagged by the model AND contacted, how many stayed?
*   **Control Group:** Compare to customers flagged but *not* contacted.
*   **Uplift:** Save Rate (Treated) - Save Rate (Control).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Sleeping Dog" Effect

*   **Risk:** Calling a low-risk customer to say "Please don't leave" might remind them to shop around.
*   **Solution:** Uplift Modeling (Day 103). Identify customers who are *persuadable*, not just those at risk.

### 6.2 Data Leakage

*   **Leak:** Using "Cancellation Date" as a feature.
*   **Leak:** Using "Refund Amount" (only happens after cancel).
*   **Fix:** Use only data available *30 days before* the renewal date.

---

## 7. Advanced Topics & Extensions

### 7.1 Sentiment Analysis for Churn

*   **Source:** Call center transcripts, Emails.
*   **Signal:** "Angry" sentiment or keywords like "Cancel", "Expensive", "Supervisor".
*   **Integration:** Add `sentiment_score` as a feature to the XGBoost model.

### 7.2 Competitor Pricing Integration

*   **Idea:** Churn depends on *Relative* Price.
*   **Feature:** `My_Price / Competitor_Price`.
*   **Source:** Comparative Raters (e.g., Compare.com data) or scraping.

---

## 8. Regulatory & Governance Considerations

### 8.1 Price Optimization Bans

*   **Regulation:** Some states (UK FCA, some US states) ban "Price Walking" (charging loyal customers more).
*   **Impact:** You cannot use Churn Propensity to set prices (e.g., charging high-retention customers more).
*   **Allowed:** You *can* use it for marketing (e.g., sending a "Thank You" card).

---

## 9. Practical Example

### 9.1 The "Renewal Center" Dashboard

**Scenario:** 10,000 policies renewing next month.
**Model:**
1.  **Risk Score:** High/Medium/Low Churn Risk.
2.  **Value Score:** High/Low CLV.
**Strategy:**
*   **High Risk + High Value:** "White Glove" treatment. Agent calls, offers discount/review.
*   **High Risk + Low Value:** Let them go (improves loss ratio).
*   **Low Risk:** Automated email.
**Impact:** Retention up 2%, Loss Ratio down 1%.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Rate Increase** is the #1 churn driver.
2.  **Survival Analysis** handles the time dimension.
3.  **Intervention** matters more than prediction.

### 10.2 When to Use This Knowledge
*   **Marketing:** Targeting retention campaigns.
*   **Pricing:** Understanding elasticity (if legal).

### 10.3 Critical Success Factors
1.  **Timing:** Predict churn 45-60 days before renewal (before the bill goes out).
2.  **Explainability:** Agents need to know *why* a customer is at risk (e.g., "Price" vs "Service") to have the right conversation.

### 10.4 Further Reading
*   **Berry & Linoff:** "Data Mining Techniques" (Chapter on Churn).
*   **Gupta:** "Managing Customers as Investments".

---

## Appendix

### A. Glossary
*   **Retention Rate:** % of customers who renew.
*   **Lapse Rate:** % of customers who leave.
*   **NPS:** Net Promoter Score (Proxy for loyalty).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Hazard Ratio** | $\frac{\lambda(t|x_1)}{\lambda(t|x_0)}$ | Relative Risk |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
