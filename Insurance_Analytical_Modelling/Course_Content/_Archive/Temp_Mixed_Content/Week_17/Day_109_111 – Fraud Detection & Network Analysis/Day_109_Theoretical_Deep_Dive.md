# Fraud Detection & Network Analysis (Part 1) - Anomaly Detection - Theoretical Deep Dive

## Overview
"Fraud is not a classification problem; it's a needle-in-a-haystack problem."
In Insurance, 99% of claims are legitimate. If you predict "Legitimate" for everyone, you get 99% accuracy but lose millions. Today, we explore **Unsupervised Anomaly Detection** to find the unknown unknowns.

---

## 1. Conceptual Foundation

### 1.1 The Imbalance Problem

*   **Supervised Learning:** Requires labeled data ("Fraud" vs "Non-Fraud").
*   **The Reality:**
    *   Labels are scarce (SIU investigators are expensive).
    *   Labels are noisy (Many frauds go undetected).
    *   Labels are outdated (Fraudsters change tactics).
*   **Unsupervised Solution:** Assume Fraud is "Rare" and "Different".

### 1.2 Isolation Forest

*   **Logic:** It's easier to isolate an anomaly than a normal point.
*   **Mechanism:** Randomly split the data.
    *   *Normal Point:* Deep in the tree (requires many splits to isolate).
    *   *Anomaly:* Near the root (requires few splits).
*   **Score:** Inverse of Path Length.

### 1.3 Autoencoders (Deep Anomaly Detection)

*   **Logic:** A Neural Network tries to compress data and reconstruct it.
*   **Mechanism:** Train on *Normal* claims only.
    *   The network learns the "Pattern of Normality".
    *   When a *Fraud* claim comes in, the network fails to reconstruct it.
*   **Score:** Reconstruction Error (MSE).

---

## 2. Mathematical Framework

### 2.1 Isolation Forest Score

$$ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} $$

*   $h(x)$: Path length of point $x$.
*   $E(h(x))$: Average path length across trees.
*   $c(n)$: Average path length of a generic BST (Normalization factor).
*   If $s \to 1$, it's an anomaly. If $s \to 0.5$, it's normal.

### 2.2 Autoencoder Loss

$$ \mathcal{L} = || x - \text{Decoder}(\text{Encoder}(x)) ||^2 $$

*   We minimize this loss on the training set (Normal claims).
*   At inference time, we use $\mathcal{L}$ as the Anomaly Score.

---

## 3. Theoretical Properties

### 3.1 The Curse of Dimensionality

*   In high dimensions, "distance" becomes meaningless. Everything is far from everything.
*   **Isolation Forest:** Handles high dimensions well because it selects features randomly.
*   **Autoencoder:** Handles high dimensions by learning a low-dimensional manifold (Latent Space).

### 3.2 Contamination

*   **Assumption:** The training data is "mostly" clean.
*   If your training data has 20% fraud, Unsupervised methods will learn that fraud is "normal".
*   *Rule of Thumb:* Works best if fraud < 5%.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Isolation Forest (Scikit-Learn)

```python
from sklearn.ensemble import IsolationForest

# 1. Data (Features: Claim Amount, Time Since Policy Start, etc.)
X = df[['amount', 'days_since_inception', 'num_past_claims']]

# 2. Train Model
# contamination='auto' or estimate (e.g., 0.01)
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
clf.fit(X)

# 3. Predict
# -1 = Anomaly, 1 = Normal
preds = clf.predict(X)
scores = clf.decision_function(X) # Raw scores
```

### 4.2 Autoencoder (Keras)

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Architecture
input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation="relu")(input_layer) # Compress
decoder = Dense(input_dim, activation="linear")(encoder) # Reconstruct

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# 2. Train (on Normal data only!)
X_normal = X[y == 0]
autoencoder.fit(X_normal, X_normal, epochs=50, batch_size=32)

# 3. Detect
reconstructions = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructions, 2), axis=1)
# High MSE = Fraud
```

---

## 5. Evaluation & Validation

### 5.1 Precision-Recall vs. ROC

*   **ROC Curve:** Misleading for imbalanced data. (High True Negative rate inflates AUC).
*   **PR Curve (AUPRC):** Focuses on the minority class.
*   **Precision@K:** If we send the top 100 suspicious claims to SIU, how many are actually fraud?

### 5.2 Lift Curve

*   Compare the model against Random Selection.
*   "In the top 1% of scores, we find 10x more fraud than random." (Lift = 10).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: One-Hot Encoding High Cardinality**
    *   "Zip Code" has 10,000 values. One-Hot encoding creates sparse vectors.
    *   *Result:* Distance-based methods fail.
    *   *Fix:* Use Target Encoding or Embeddings.

2.  **Trap: Concept Drift**
    *   Fraudsters adapt. A rule that works today ("Staged Accident in Miami") won't work tomorrow.
    *   *Fix:* Retrain frequently. Monitor score distribution.

---

## 7. Advanced Topics & Extensions

### 7.1 Variational Autoencoders (VAE)

*   Instead of a point in latent space, map input to a *distribution*.
*   Better for detecting "Out of Distribution" samples.

### 7.2 Semi-Supervised Learning

*   We have *some* labels (100 confirmed frauds).
*   **Deep SVDD (Support Vector Data Description):** Train a network to map normal data to a sphere. Minimize the sphere's volume.

---

## 8. Regulatory & Governance Considerations

### 8.1 False Positives & Customer Friction

*   **Risk:** Flagging a loyal customer as a fraudster is a disaster.
*   **Process:** The model output is a *referral* to SIU (Special Investigative Unit), not an automatic denial.
*   **Explainability:** SIU needs to know *why*. (Use SHAP).

---

## 9. Practical Example

### 9.1 Worked Example: Staged Accidents

**Scenario:**
*   **Pattern:** New policy, high premium vehicle, accident within 3 days, 4 passengers claiming whiplash.
*   **Isolation Forest:**
    *   "Days since inception" = 3 (Rare).
    *   "Num Passengers" = 4 (Rare).
    *   *Result:* Very short path length -> Anomaly Score 0.9.
*   **Action:** Flag for SIU interview.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Unsupervised Learning** finds patterns without labels.
2.  **Isolation Forest** is fast and effective.
3.  **Autoencoders** capture complex non-linear anomalies.

### 10.2 When to Use This Knowledge
*   **Claims Triage:** Fast-tracking simple claims, flagging complex ones.
*   **Underwriting:** Detecting "Ghost Brokers" (Fake agents).

### 10.3 Critical Success Factors
1.  **Feature Engineering:** "Ratio of Claim to Premium" is better than raw numbers.
2.  **Feedback Loop:** SIU feedback must retrain the model.

### 10.4 Further Reading
*   **Liu et al.:** "Isolation Forest".

---

## Appendix

### A. Glossary
*   **SIU:** Special Investigative Unit.
*   **Reconstruction Error:** Difference between Input and Output.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **MSE** | $\frac{1}{n} \sum (x_i - \hat{x}_i)^2$ | Autoencoder Loss |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
