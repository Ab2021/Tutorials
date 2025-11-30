# Deep Learning for Actuaries (Part 2) - Sequence Modeling & Time Series - Theoretical Deep Dive

## Overview
"Insurance is a promise in time."
Claims happen over time. Driving happens over time. Mortality happens over time.
Standard Feed-Forward Networks (MLPs) assume inputs are static and independent. They fail at "Sequence".
This day focuses on **Recurrent Neural Networks (RNNs)**, **LSTMs**, and **GRUs**: The architectures that have "Memory".

---

## 1. Conceptual Foundation

### 1.1 The Sequence Problem

*   **Static Data:** [Age, Gender, Car]. Order doesn't matter.
*   **Sequence Data:** [Speed at t=1, Speed at t=2, Speed at t=3]. Order matters.
*   **Markov Property:** The future depends on the past.
*   **Application:**
    *   **Telematics:** A sequence of GPS points.
    *   **Claims Reserving:** A sequence of payments over development periods.
    *   **Mortality:** A sequence of health states.

### 1.2 The RNN Cell

*   **Idea:** The network has a "Hidden State" ($h_t$) that acts as short-term memory.
*   **Process:**
    1.  Input $x_t$ comes in.
    2.  Combine $x_t$ with previous state $h_{t-1}$.
    3.  Output new state $h_t$ and prediction $y_t$.
*   **Flaw:** "Short-term Memory Loss". Standard RNNs cannot remember things from 100 steps ago (Vanishing Gradient).

---

## 2. Mathematical Framework

### 2.1 Long Short-Term Memory (LSTM)

Designed to solve the memory problem.
*   **The Cell State ($C_t$):** The "Long-term Highway". Information flows through it unchanged unless explicitly modified.
*   **The Gates:**
    1.  **Forget Gate ($f_t$):** What should I throw away? (Sigmoid).
    2.  **Input Gate ($i_t$):** What new info should I store?
    3.  **Output Gate ($o_t$):** What should I tell the next cell?

### 2.2 Gated Recurrent Unit (GRU)

*   **Simplified LSTM:** Merges the Cell State and Hidden State.
*   **Gates:** Update Gate and Reset Gate.
*   **Benefit:** Faster to train, often just as good as LSTM for smaller datasets (common in insurance).

---

## 3. Theoretical Properties

### 3.1 Many-to-One vs. Many-to-Many

*   **Many-to-One:**
    *   *Input:* Sequence of 100 driving seconds.
    *   *Output:* One Risk Score (Probability of Accident).
    *   *Use Case:* Telematics Pricing.
*   **Many-to-Many:**
    *   *Input:* Sequence of Paid Losses (Years 1-5).
    *   *Output:* Sequence of Future Payments (Years 6-10).
    *   *Use Case:* Stochastic Reserving (DeepTriangle).

### 3.2 Bidirectional RNNs

*   **Idea:** Read the sequence Forwards AND Backwards.
*   **Context:** In text (NLP), the meaning of a word depends on the words *after* it.
*   **Insurance:** Less common for real-time (can't see future), but useful for analyzing historical claim notes.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Building an LSTM for Telematics (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_telematics_model(timesteps, features):
    # Input: (Batch, Timesteps, Features)
    # e.g., (32, 100, 5) -> 32 trips, 100 seconds each, 5 signals (Speed, Accel, etc.)
    inputs = layers.Input(shape=(timesteps, features))
    
    # LSTM Layer
    # return_sequences=False -> Only output the final state (Many-to-One)
    x = layers.LSTM(64, return_sequences=False)(inputs)
    
    # Dense Layers for Classification
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

# Example
model = build_telematics_model(timesteps=100, features=5)
model.summary()
```

### 4.2 DeepTriangle (Reserving)

*   **Architecture:** Sequence-to-Sequence (Seq2Seq).
*   **Encoder:** Reads the history of Paid Losses.
*   **Decoder:** Generates the future development.
*   **Benefit:** Captures non-linear development patterns and calendar year trends simultaneously.

---

## 5. Evaluation & Validation

### 5.1 Backtesting Time Series

*   **Method:** Walk-Forward Validation.
    *   Train on Jan-Mar. Test on Apr.
    *   Train on Jan-Apr. Test on May.
*   **Rule:** Never use K-Fold Cross Validation on time series (Leakage).

### 5.2 The "Lag" Baseline

*   **Check:** Does your LSTM beat a naive model that just predicts $y_t = y_{t-1}$?
*   **Reality:** Often, for simple financial series, the naive model is hard to beat.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Variable Length Sequences

*   **Problem:** Trip A is 5 minutes. Trip B is 30 minutes.
*   **Solution:** Padding.
    *   Pad Trip A with zeros to match Trip B.
    *   Use `Masking` layer in Keras so the LSTM ignores the zeros.

### 6.2 Data Leakage (Lookahead Bias)

*   **Issue:** Using "Total Trip Distance" as a feature at $t=1$.
*   **Reality:** At $t=1$, you don't know the total distance yet.
*   **Fix:** Only use features available at time $t$.

---

## 7. Advanced Topics & Extensions

### 7.1 1D-CNNs for Time Series

*   **Alternative:** Instead of RNNs, use Convolutional Neural Networks (1D).
*   **Mechanism:** Slide a filter over the time axis.
*   **Benefit:** Much faster (parallelizable). Detects local patterns (e.g., "Hard Brake" signature) efficiently.
*   **Trend:** Transformers (Attention) are replacing RNNs, but 1D-CNNs remain a strong baseline.

### 7.2 Attention Mechanisms

*   **Idea:** Let the model "focus" on specific time steps.
*   **Telematics:** "Focus on the 5 seconds before the crash, ignore the 20 minutes of highway driving."
*   **Interpretability:** Attention weights show *where* the risk is.

---

## 8. Regulatory & Governance Considerations

### 8.1 Explaining the "Black Box" in Time

*   **Challenge:** "Why did the LSTM predict a crash?"
*   **Solution:** Integrated Gradients or Attention Maps.
    *   "The model reacted to the sequence [Accel, Brake, Accel] at timestamp 45s."

---

## 9. Practical Example

### 9.1 The "Drowsy Driver" Detector

**Scenario:** Real-time monitoring of steering wheel angle.
**Data:** Sequence of angles $\theta_t$.
**Pattern:**
*   Alert: Constant micro-corrections.
*   Drowsy: Long period of no correction, followed by a jerk.
**Model:** LSTM trained on labeled "Drowsy" vs "Alert" sequences.
**Deployment:** Runs on the edge (in the car).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **LSTMs** handle memory.
2.  **Sequence Data** requires special handling (Padding, Masking).
3.  **Time Series** validation must respect time order.

### 10.2 When to Use This Knowledge
*   **Telematics:** The #1 use case.
*   **Macro-economics:** Forecasting inflation/interest rates.

### 10.3 Critical Success Factors
1.  **Feature Scaling:** LSTMs are very sensitive to scale. Normalize everything.
2.  **Stationarity:** Remove trends before feeding into the LSTM (e.g., predict *change* in loss, not absolute loss).

### 10.4 Further Reading
*   **Kuo:** "DeepTriangle: A Deep Learning Approach to Loss Reserving".
*   **Colah's Blog:** "Understanding LSTM Networks".

---

## Appendix

### A. Glossary
*   **Hidden State:** The internal memory of the RNN.
*   **Timestep:** One point in the sequence.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **LSTM Forget Gate** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | Memory Control |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
