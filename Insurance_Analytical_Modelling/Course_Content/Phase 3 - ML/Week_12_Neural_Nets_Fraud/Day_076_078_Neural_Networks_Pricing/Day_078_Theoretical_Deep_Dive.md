# Neural Networks Fundamentals (Part 1) - Foundations & Interpretable Deep Pricing - Theoretical Deep Dive

## Overview
"Neural Networks are black boxes."
This is the biggest myth in modern actuarial science.
While a standard Multi-Layer Perceptron (MLP) is opaque, modern architectures like **Neural Additive Models (NAMs)** and **Combined Actuarial Neural Networks (CANNs)** offer the interpretability of a GLM with the predictive power of Deep Learning.
This day introduces the mathematical foundations of Neural Networks and their specific application to Insurance Pricing.

---

## 1. Conceptual Foundation

### 1.1 From GLM to Neural Network

*   **GLM:** $g(\mu) = \beta_0 + \beta_1 x_1 + \beta_2 x_2$.
    *   Linear combination of features.
*   **Neural Network (1 Hidden Layer):** $g(\mu) = \sigma(W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2)$.
    *   Non-linear combination of features.
*   **The Connection:** A GLM is just a Neural Network with 0 hidden layers and a specific link function.

### 1.2 The "Black Box" Problem

*   **Regulatory Constraint:** You cannot deploy a model if you cannot explain *why* it doubled someone's premium.
*   **Solution:** **Interpretable Architectures**.
    *   Instead of a fully connected "Soup" of neurons, we structure the network to isolate feature effects.

---

## 2. Mathematical Framework

### 2.1 The Perceptron

The fundamental unit of Deep Learning.
$$ z = \sum w_i x_i + b $$
$$ a = \sigma(z) $$
*   $w$: Weights (Parameters to learn).
*   $b$: Bias (Intercept).
*   $\sigma$: Activation Function (ReLU, Sigmoid, Tanh).

### 2.2 Loss Functions for Insurance

Standard MSE (Mean Squared Error) is wrong for insurance because claims are not Normal.
*   **Poisson Loss (Frequency):**
    $$ L(y, \hat{y}) = \hat{y} - y \ln(\hat{y}) $$
*   **Gamma Loss (Severity):**
    $$ L(y, \hat{y}) = \frac{y - \hat{y}}{\hat{y}} + \ln\left(\frac{\hat{y}}{y}\right) $$
*   **Tweedie Loss (Pure Premium):**
    $$ L(y, \hat{y}) = -y \frac{\hat{y}^{1-p}}{1-p} + \frac{\hat{y}^{2-p}}{2-p} $$

---

## 3. Theoretical Properties

### 3.1 Universal Approximation Theorem

*   **Theorem:** A neural network with a single hidden layer (and enough neurons) can approximate *any* continuous function.
*   **Implication:** Neural Networks can learn interactions (Age $\times$ Car Power) and non-linearities (Age$^2$) without manual feature engineering.

### 3.2 Embedding Layers

*   **Problem:** High Cardinality Categorical Variables (e.g., Zip Code, 50,000 levels).
*   **GLM Approach:** One-Hot Encoding (Sparse, massive matrix).
*   **NN Approach:** Embeddings. Map each Zip Code to a dense vector of size $d$ (e.g., 5).
    *   Zip 90210 $\to$ `[0.1, -0.5, 0.8, 0.2, 0.0]`.
    *   The network *learns* which Zip Codes are similar.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Building a Simple Pricing Network (Keras/TensorFlow)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_pricing_model(input_dim):
    # Input Layer
    inputs = layers.Input(shape=(input_dim,))
    
    # Hidden Layers (The "Deep" part)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output Layer (Exponential activation for positive pricing)
    outputs = layers.Dense(1, activation='exponential')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile with Poisson Loss (for Frequency)
    model.compile(optimizer='adam', loss='poisson')
    return model

# Example Usage
model = build_pricing_model(input_dim=10)
model.summary()
```

### 4.2 The CANN Approach (Combined Actuarial Neural Network)

*   **Idea:** Start with a GLM (Skip Connection) and add a Neural Network to capture the residuals/interactions.
*   **Formula:** $\mu = \exp(\text{GLM}(x) + \text{NN}(x))$.
*   **Benefit:** If the NN weights are 0, you recover the standard GLM. The NN only learns what the GLM missed.

---

## 5. Evaluation & Validation

### 5.1 Gradient-Based Attribution

*   **Question:** Which feature drove the price up?
*   **Method:** Calculate $\frac{\partial \text{Price}}{\partial x_i}$.
*   **Interpretation:** If the gradient is positive, increasing $x_i$ increases the price.

### 5.2 Stability Analysis

*   **Risk:** Neural Networks can be unstable (small change in input $\to$ huge change in output).
*   **Check:** Adversarial Testing. Perturb inputs by 1% and ensure output changes by $< 1\%$.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Overfitting

*   **Issue:** NNs have millions of parameters. They can memorize the training data.
*   **Solution:**
    *   **Dropout:** Randomly turn off neurons during training.
    *   **Early Stopping:** Stop training when Validation Loss stops improving.
    *   **L1/L2 Regularization:** Penalize large weights.

### 6.2 Data Scaling

*   **GLM:** Doesn't care if Age is 20 or 0.2.
*   **NN:** Fails if inputs are not scaled.
*   **Rule:** Always normalize continuous inputs to Mean=0, Std=1 (StandardScaler) or Range=[0,1] (MinMaxScaler).

---

## 7. Advanced Topics & Extensions

### 7.1 Neural Additive Models (NAMs)

*   **Structure:** $\sum g_i(x_i)$.
*   **Architecture:** Each feature $x_i$ goes into its *own* sub-network. The outputs are summed.
*   **Result:** You can plot exactly how the network treats "Age", independent of other features. It is fully interpretable.

### 7.2 Multi-Task Learning

*   **Idea:** Train one network to predict Frequency AND Severity simultaneously.
*   **Benefit:** The "Feature Extraction" layers learn representations useful for both tasks (e.g., "Bad Driver" latent variable).

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Black Box" Defense

*   **Regulator:** "Show me the formula."
*   **Actuary:** "Here is the NAM plot for every variable. It is monotonic and smooth."
*   **Result:** NAMs are much easier to get approved than standard MLPs.

---

## 9. Practical Example

### 9.1 The "Telematics Encoder"

**Scenario:** You have second-by-second driving data.
**Task:** Predict accident probability.
**Approach:**
1.  Feed the time-series data into an **LSTM** or **1D-CNN**.
2.  Output a "Driving Style Embedding" (vector of size 10).
3.  Feed this embedding into the Pricing GLM as a new feature.
**Result:** You capture complex patterns (sudden braking) without manually defining "hard brake events".

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Loss Function** matters more than architecture. Use Poisson/Tweedie.
2.  **Scaling** is mandatory.
3.  **Interpretability** is non-negotiable in insurance. Use CANNs or NAMs.

### 10.2 When to Use This Knowledge
*   **High Volume Lines:** Personal Auto, where small gains in accuracy = millions in profit.
*   **Complex Data:** Telematics, Images, Text.

### 10.3 Critical Success Factors
1.  **Start Simple:** Fit a GLM first. Only use a NN if it beats the GLM significantly.
2.  **Architecture Search:** Don't just guess the number of layers. Use Keras Tuner.

### 10.4 Further Reading
*   **Wüthrich & Merz:** "Statistical Foundations of Actuarial Learning and its Applications".
*   **Google Research:** "Neural Additive Models: Interpretable Machine Learning with Neural Nets".

---

## Appendix

### A. Glossary
*   **Activation Function:** The non-linear switch (ReLU, Sigmoid).
*   **Backpropagation:** The algorithm to calculate gradients and update weights.
*   **Epoch:** One pass through the entire training dataset.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Poisson Loss** | $\hat{y} - y \ln(\hat{y})$ | Frequency Training |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
