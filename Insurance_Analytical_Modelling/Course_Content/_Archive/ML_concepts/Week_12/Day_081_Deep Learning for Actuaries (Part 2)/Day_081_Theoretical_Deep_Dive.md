# Deep Learning for Actuaries (Part 2) - Theoretical Deep Dive

## Overview
We continue our journey into Deep Learning with advanced actuarial applications. Today, we cover the **Deep Triangle** (Reserving), **Telematics** (CNNs for driving data), and **Fraud Detection** using Autoencoders.

---

## 1. Conceptual Foundation

### 1.1 Deep Triangle (Reserving)

*   **Problem:** Chain Ladder assumes a fixed development pattern. It fails when there are trend changes or complex interactions.
*   **Solution:** Treat the Run-off Triangle as a sequence prediction problem.
*   **Architecture:** Use Recurrent Neural Networks (GRU/LSTM) to predict the next payment based on the history of payments.
*   **Benefit:** Can handle multiple lines of business simultaneously and learn cross-line correlations.

### 1.2 Telematics & CNNs

*   **Data:** Second-by-second speed, acceleration, and G-force.
*   **Insight:** A trip is just a 1D "image" of driving behavior.
*   **CNN (Convolutional Neural Network):**
    *   Normally used for 2D images (Cats vs. Dogs).
    *   **1D CNN:** Slides a filter over the time-series data to detect patterns like "Hard Braking" or "Rapid Acceleration".
    *   *Result:* Automatically extracts features from raw sensor data.

### 1.3 Autoencoders for Fraud

*   **Problem:** Fraud is rare. We don't have enough "Fraud" labels to train a classifier.
*   **Solution:** Unsupervised Anomaly Detection.
*   **Mechanism:**
    1.  Train an Autoencoder to compress and reconstruct *normal* claims.
    2.  Feed a *fraudulent* claim.
    3.  The model will fail to reconstruct it accurately (High Reconstruction Error).
    4.  *Flag:* High Error = Potential Fraud.

---

## 2. Mathematical Framework

### 2.1 RNN/GRU for Reserving

*   **Input:** Sequence of cumulative payments $C_{i,0}, C_{i,1}, \dots, C_{i,t}$.
*   **Hidden State:** $h_t = \text{GRU}(C_{i,t}, h_{t-1})$.
*   **Output:** Predicted payment $C_{i,t+1}$.
*   **Loss:** Mean Squared Error on the observed part of the triangle.

### 2.2 Autoencoder Loss

*   Input $x$. Encoder $z = E(x)$. Decoder $\hat{x} = D(z)$.
*   **Loss:** $L = ||x - \hat{x}||^2$.
*   The model is forced to learn the "essence" of the data in the bottleneck layer $z$.

---

## 3. Theoretical Properties

### 3.1 Long Short-Term Memory (LSTM)

*   Standard RNNs forget early history (Vanishing Gradient).
*   **LSTM:** Has a "Cell State" (Long-term memory) and "Gates" (Input, Output, Forget).
*   *Actuarial Use:* Modeling a claim history that spans 10 years. The LSTM remembers the initial injury type from Year 1 when predicting the payment in Year 10.

### 3.2 1D Convolutions

*   Filter size $k$ (e.g., 5 seconds).
*   Stride $s$ (e.g., 1 second).
*   The filter learns shapes. One filter might learn "Sudden Stop", another "High Speed Turn".

---

## 4. Modeling Artifacts & Implementation

### 4.1 Simple LSTM for Sequence Data (Python)

```python
import torch
import torch.nn as nn

class ClaimLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (Batch, Sequence Length, Features)
        out, (hn, cn) = self.lstm(x)
        
        # We only care about the last time step for prediction
        last_out = out[:, -1, :] 
        prediction = self.fc(last_out)
        return prediction

# Example: Predict next payment based on last 5 years
model = ClaimLSTM(input_size=1, hidden_size=32, output_size=1)
```

### 4.2 Autoencoder for Anomaly Detection

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Compress 50 features to 5
        self.encoder = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        # Decoder: Reconstruct 5 to 50
        self.decoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 50)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Detection Logic
# error = torch.mean((x - model(x))**2, dim=1)
# if error > threshold: print("Fraud!")
```

---

## 5. Evaluation & Validation

### 5.1 Backtesting Reserving Models

*   Cut the triangle at year $T-k$.
*   Predict the diagonal $T-k+1 \dots T$.
*   Compare with actuals.
*   *Metric:* RMSE, MAPE.

### 5.2 Telematics Validation

*   **ROC Curve:** Can the CNN distinguish between "Safe Driver" and "Stunt Driver"?
*   *Challenge:* Labeling data. Who decides what is a "Safe" trip? (Usually based on claims history).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Overfitting the Triangle**
    *   Run-off triangles are small (e.g., $10 \times 10$ cells). Deep Learning needs Big Data.
    *   *Fix:* Train on *individual claims* data, not aggregated triangles. (Granular Reserving).

2.  **Trap: Autoencoder Threshold**
    *   Setting the fraud threshold is subjective.
    *   *Fix:* Use a Precision-Recall curve to balance False Positives vs. False Negatives.

### 6.2 Implementation Challenges

1.  **Variable Sequence Lengths:**
    *   Some claims settle in 1 year, some in 10.
    *   *Fix:* Padding (add zeros to short sequences) and Masking (tell the LSTM to ignore zeros).

---

## 7. Advanced Topics & Extensions

### 7.1 Transformer for Reserving

*   Replace LSTM with Self-Attention.
*   "Attention" mechanism can highlight *which* past payment is most relevant for the future prediction.
*   *Paper:* "Transformer-based Reserving".

### 7.2 Graph Neural Networks (GNN)

*   Model the relationships between policyholders (e.g., Family, Business partners).
*   Fraud rings often form connected subgraphs. GNNs are perfect for detecting rings.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Black Box" Reserving

*   **Auditor:** "I can't audit a Neural Network."
*   **Actuary:** "We run Chain Ladder alongside it as a benchmark. If they diverge, we investigate."
*   **Best Practice:** Use Deep Learning as a "Challenger" model, not the primary model (yet).

---

## 9. Practical Example

### 9.1 Worked Example: The "Deep Triangle"

**Scenario:**
*   Workers Comp line. Long tail.
*   **Chain Ladder:** Under-reserves because it misses a recent trend in medical inflation.
*   **Deep Triangle:** Picks up the non-linear trend in the recent calendar years.
*   **Result:** Increases reserves by 5%. Prevents future insolvency.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Deep Triangle** modernizes reserving.
2.  **CNNs** read telematics like a book.
3.  **Autoencoders** find the needle in the haystack (Fraud).

### 10.2 When to Use This Knowledge
*   **Reserving:** When standard methods fail (e.g., changing mix of business).
*   **Fraud:** When you have lots of unlabeled data.

### 10.3 Critical Success Factors
1.  **Data Granularity:** Move from Triangles to Individual Claims.
2.  **Hybrid Approach:** Combine NN with Actuarial Judgment.

### 10.4 Further Reading
*   **Gabrielli & Wüthrich:** "A Neural Network Approach to Chain-Ladder Reserving".
*   **Wüthrich:** "Machine Learning in Insurance".

---

## Appendix

### A. Glossary
*   **Run-off Triangle:** Matrix of claim payments (Accident Year vs. Development Year).
*   **Latent Space:** The compressed representation in an Autoencoder.
*   **Padding:** Adding zeros to make sequences equal length.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Reconstruction Error** | $||x - \hat{x}||^2$ | Anomaly Detection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
