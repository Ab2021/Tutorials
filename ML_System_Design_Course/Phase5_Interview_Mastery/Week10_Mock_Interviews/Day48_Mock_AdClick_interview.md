# Day 48: Mock Interview: Ad Click Prediction - Interview Questions

> **Topic**: Computational Advertising
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design an Ad Click Prediction System.
**Answer:**
*   **Goal**: Predict $P(Click | User, Ad, Context)$.
*   **Metric**: Log Loss (Calibration matters). AUC.
*   **Scale**: High QPS, Low Latency (< 50ms).

### 2. Why is "Calibration" important?
**Answer:**
*   We bid based on Expected Value: $Bid = CTR \times Value$.
*   If model predicts 0.2 but actual is 0.1, we overbid and lose money.
*   **Isotonic Regression** / **Platt Scaling**.

### 3. What features are used?
**Answer:**
*   **User**: Demographics, History.
*   **Ad**: Creative type, Text, Advertiser.
*   **Context**: App, Time, Device.
*   **Cross**: User x Ad interaction.

### 4. Explain FTRL (Follow The Regularized Leader).
**Answer:**
*   Online learning algorithm for Logistic Regression.
*   Handles sparsity (L1) and keeps weights bounded.
*   Standard in industry.

### 5. What is "Feature Hashing" (Hashing Trick)?
**Answer:**
*   Map high-cardinality strings to fixed size vector via Hash function.
*   `hash("User123") % N`.
*   Saves memory. Handles new users.

### 6. How do you handle "Data Imbalance"?
**Answer:**
*   CTR is low (1%).
*   **Downsampling negatives**.
*   Recalibrate prediction: $p' = p / (p + (1-p)/w)$.

### 7. What is "Deep & Cross Network" (DCN)?
**Answer:**
*   Replaces manual feature crossing.
*   **Cross Network**: Explicitly learns interactions.
*   **Deep Network**: Learns implicit patterns.

### 8. How do you handle "Delayed Feedback"?
**Answer:**
*   Conversion happens days after click.
*   Train on "Click" (immediate).
*   Separate model for CVR (Conversion Rate).

### 9. What is the "Auction" mechanism?
**Answer:**
*   **Vickrey Auction (Second Price)**: Winner pays 2nd highest bid + 0.01.
*   Encourages truthful bidding.

### 10. How do you handle "Cold Start" Ads?
**Answer:**
*   Explore/Exploit (Bandits).
*   Boost bid for new ads to gather data.
*   Content-based features.

### 11. What is "Position Bias" in Ads?
**Answer:**
*   Top ad gets more clicks.
*   Train with position as feature.
*   Predict at position 0.

### 12. How do you store Feature History?
**Answer:**
*   **Feature Store**.
*   Sliding windows ("Clicks in last 7 days").

### 13. What is "Leakage" in Ad prediction?
**Answer:**
*   Using future data (e.g., "Conversion" feature in CTR model).
*   Time-based splitting is mandatory.

### 14. How do you evaluate offline?
**Answer:**
*   **Normalized Entropy (NE)**.
*   **AUC**.

### 15. What is "Wide & Deep"?
**Answer:**
*   Google's architecture.
*   **Wide**: Linear model with Cross-Product features (Memorization).
*   **Deep**: MLP (Generalization).

### 16. How do you handle "Ad Fatigue"?
**Answer:**
*   User sees same ad 10 times -> CTR drops.
*   Feature: "Times shown to user".
*   Frequency Capping.

### 17. What is "Budget Pacing"?
**Answer:**
*   Don't spend all budget in morning.
*   **PID Controller** to smooth spending throughout the day.

### 18. How do you handle "Fraudulent Clicks"?
**Answer:**
*   Filter bots.
*   IP blacklists.
*   Anomaly detection.

### 19. What is "Multi-Task Learning" in Ads?
**Answer:**
*   Predict CTR and CVR together.
*   Shared embedding layer.
*   ESMM (Entire Space Multi-Task Model).

### 20. How do you update the model?
**Answer:**
*   **Online Learning**: Update gradients on every batch of logs.
*   Freshness is critical.
