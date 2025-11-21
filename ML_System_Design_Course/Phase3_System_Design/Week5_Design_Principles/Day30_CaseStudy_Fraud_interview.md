# Day 30: Case Study: Fraud Detection - Interview Questions

> **Topic**: Anomaly Detection
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design a Credit Card Fraud Detection System.
**Answer:**
*   **Real-time**: Block transaction < 200ms.
*   **Imbalanced Data**: 0.1% Fraud.
*   **Features**: Transaction details, User history, Graph features.

### 2. How do you handle extreme Class Imbalance?
**Answer:**
*   **Resampling**: SMOTE, Undersampling.
*   **Class Weights**: High penalty for fraud class.
*   **Ensembles**: Balanced Random Forest.
*   **Metric**: PR-AUC (not ROC-AUC).

### 3. What features are important for Fraud?
**Answer:**
*   **Aggregates**: "Amount spent in last 1 hour".
*   **Velocity**: "Number of transactions in last 10 mins".
*   **Geospatial**: Distance from last transaction.
*   **Device**: IP change, Device fingerprint.

### 4. Explain Isolation Forest.
**Answer:**
*   Unsupervised Anomaly Detection.
*   Builds random trees.
*   Anomalies are easy to isolate (short path length).
*   Normal points are deep in the tree.

### 5. Explain One-Class SVM.
**Answer:**
*   Learns a boundary around the "Normal" data.
*   Anything outside is Anomaly.
*   Good for outlier detection.

### 6. How do Graph Neural Networks (GNN) help in Fraud?
**Answer:**
*   Fraudsters work in rings/syndicates.
*   Graph captures relationships (Shared Device, Shared IP).
*   **GraphSAGE**: Aggregates neighbor info to detect suspicious subgraphs.

### 7. What is "Active Learning" in Fraud?
**Answer:**
*   Labels are expensive (Analysts must investigate).
*   Model selects "most uncertain" or "most likely fraud" cases for human review.
*   Model learns faster with fewer labels.

### 8. How do you handle "Concept Drift" in Fraud?
**Answer:**
*   Fraudsters adapt strategies.
*   **Online Learning**: Update model frequently.
*   **Retrain**: Weekly.
*   **Monitor**: Feature drift.

### 9. What is the cost of False Positives vs False Negatives?
**Answer:**
*   **FP**: Block legit user. Frustration. Churn.
*   **FN**: Allow fraud. Financial loss. Chargebacks.
*   Usually, optimize for high Recall at acceptable Precision.

### 10. Explain "Device Fingerprinting".
**Answer:**
*   Identifying a device uniquely (Browser, OS, Screen Res, Battery Level).
*   Detects if one device is used for 100 accounts.

### 11. What is an Autoencoder for Anomaly Detection?
**Answer:**
*   Train to reconstruct "Normal" transactions.
*   **Inference**: Calculate Reconstruction Error.
*   High error = Anomaly (Model hasn't seen this pattern).

### 12. How do you handle Categorical Features with high cardinality (IP, Merchant)?
**Answer:**
*   **Target Encoding**: Replace MerchantID with "Average Fraud Rate of Merchant".
*   **Smoothed**: Add regularization to prevent overfitting rare merchants.

### 13. What is "Benford's Law"?
**Answer:**
*   Distribution of first digits in natural numbers.
*   1 appears 30% of time. 9 appears 5%.
*   Deviations indicate fabricated data (Accounting fraud).

### 14. Explain "Velocity Features".
**Answer:**
*   Speed of events.
*   "3 transactions in 1 minute".
*   Crucial for bot detection.

### 15. How do you evaluate a Fraud model?
**Answer:**
*   **Precision-Recall Curve**.
*   **Recall @ Top K**: If we can only review 100 cases, how many frauds are in top 100?

### 16. What is "Adversarial Attacks" in Fraud?
**Answer:**
*   Fraudsters probing the system to find thresholds.
*   "Camouflage": Making fraud look like normal behavior.

### 17. How do you explain a fraud prediction to an analyst? (SHAP).
**Answer:**
*   **SHAP values**: "Score is high because 'Amount > $5000' and 'IP is Nigeria'".
*   Transparency builds trust.

### 18. What is "Rule-based" vs "ML-based" fraud detection?
**Answer:**
*   **Rules**: "If Amount > 10k, Block". Simple, interpretable, rigid.
*   **ML**: Finds complex patterns. Adaptive. Harder to explain.
*   **Hybrid**: ML score + Hard Rules (Sanctions list).

### 19. How do you handle "Label Delay"?
**Answer:**
*   Chargeback comes 30 days later.
*   Use "Prediction Drift" monitoring.
*   Use "Short-term" labels (User blocked by analyst) as proxy.

### 20. What is "Link Analysis"?
**Answer:**
*   Visualizing the graph of Users, Cards, IPs.
*   Finding connected components (Fraud Rings).
