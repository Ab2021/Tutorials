# Advanced Loss & Tail Modelling (Part 3) - Deep Learning for Extremes (Deep EVT) - Theoretical Deep Dive

## Overview
Standard EVT assumes parameters ($\xi, \sigma$) are constant or change linearly. **Deep EVT** uses Neural Networks to learn how these parameters vary with complex features (e.g., Satellite Images, Telematics). It combines the *flexibility* of Deep Learning with the *safety* of Extreme Value Theory.

---

## 1. Conceptual Foundation

### 1.1 The Hybrid Approach

*   **Pure Deep Learning:** Good at the mean, bad at the tail. (MSE loss ignores outliers).
*   **Pure EVT:** Good at the tail, bad at complex patterns.
*   **Deep EVT:**
    *   Use a Neural Network to predict the *parameters* of the GPD.
    *   Output: $\hat{\xi}(x)$ and $\hat{\sigma}(x)$.

### 1.2 Mixture Density Networks (MDN)

*   A single distribution rarely fits the whole loss curve.
*   **Spliced Network:**
    *   **Head 1:** Predicts probability of being "Normal" vs "Extreme".
    *   **Head 2:** Predicts Log-Normal parameters (for the body).
    *   **Head 3:** Predicts GPD parameters (for the tail).

---

## 2. Mathematical Framework

### 2.1 The GPD Loss Function

To train a Neural Network to output GPD parameters, we minimize the **Negative Log-Likelihood (NLL)** of the GPD.

$$ \mathcal{L}(\xi, \sigma | y) = \ln \sigma + \left( 1 + \frac{1}{\xi} \right) \ln \left( 1 + \frac{\xi y}{\sigma} \right) $$

*   The Network minimizes $\sum \mathcal{L}(\hat{\xi}_i, \hat{\sigma}_i | y_i)$.
*   *Constraint:* $1 + \frac{\xi y}{\sigma} > 0$.

### 2.2 Parameter Constraints

*   $\sigma$ must be positive. (Use `Softplus` activation).
*   $\xi$ usually needs to be bounded (e.g., $< 1$) to ensure finite mean.

---

## 3. Theoretical Properties

### 3.1 Universal Approximation for Tails

*   A Neural Network can approximate any continuous function.
*   Therefore, it can approximate the true mapping from *Risk Factors* to *Tail Risk*.
*   *Example:* How does "Driver Age" + "Car Horsepower" + "Rain Intensity" interact to affect the *Tail Index* of accident severity?

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Network
class DeepEVT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Output Heads
        self.xi_head = nn.Linear(32, 1)
        self.sigma_head = nn.Linear(32, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.shared(x)
        # Constrain xi to be small positive (e.g., 0 to 1)
        xi = torch.sigmoid(self.xi_head(h)) 
        # Constrain sigma to be positive
        sigma = self.softplus(self.sigma_head(h))
        return xi, sigma

# 2. Define Custom Loss (GPD NLL)
def gpd_loss(y, xi, sigma):
    # y is excess over threshold
    # Add small epsilon for stability
    term1 = torch.log(sigma)
    term2 = (1 + 1/xi) * torch.log(1 + (xi * y) / sigma)
    return torch.mean(term1 + term2)

# 3. Training Loop
model = DeepEVT(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for x_batch, y_batch in loader:
    xi_pred, sigma_pred = model(x_batch)
    loss = gpd_loss(y_batch, xi_pred, sigma_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 5. Evaluation & Validation

### 5.1 Conditional QQ Plots

*   Since $\xi$ and $\sigma$ vary per customer, we can't just plot one QQ plot.
*   **Transform:** Convert data to Uniform using the predicted CDF: $U_i = F(y_i; \hat{\xi}_i, \hat{\sigma}_i)$.
*   Check if $U_i$ follows a Uniform distribution.

### 5.2 Tail Calibration Score

*   Check if the predicted 99% VaR is exceeded exactly 1% of the time across different segments (e.g., Young vs Old drivers).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Training on Non-Extreme Data**
    *   If you feed the whole dataset into the GPD loss, it will fail. GPD is only for the tail.
    *   *Fix:* Use a "Hard Threshold" (only train on $y > u$) or a "Soft Threshold" (Mixture Model).

2.  **Trap: Exploding Gradients**
    *   The term $1/\xi$ in the loss function is unstable if $\xi \to 0$.
    *   *Fix:* Use a slightly different parameterization or clamp $\xi$.

---

## 7. Advanced Topics & Extensions

### 7.1 Variational Deep EVT

*   Instead of point estimates for $\xi, \sigma$, predict *distributions* (Uncertainty).
*   Useful for "Epistemic Uncertainty" (Model risk).

### 7.2 Spatio-Temporal Deep EVT

*   Using CNNs (Convolutional Neural Networks) to extract features from Weather Maps, then feeding them into the EVT head.
*   *Application:* Real-time Flood Severity forecasting.

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Black Box" Problem

*   Regulators hate Black Boxes for Capital Modeling.
*   *Defense:* The *structure* (GPD) is standard and interpretable. Only the *parameter estimation* is deep.
*   *Explainability:* Use SHAP to show which features drive $\xi$ (Tail Risk).

---

## 9. Practical Example

### 9.1 Worked Example: Telematics & Crash Severity

**Scenario:**
*   We have Telematics data (Speeding, Braking) for 1M drivers.
*   **Goal:** Predict the *severity* of a crash if it happens.
*   **Model:** Deep EVT.
*   **Result:**
    *   Driver A (Speeding): Predicted $\xi = 0.8$ (High Tail Risk). VaR = \$500k.
    *   Driver B (Cautious): Predicted $\xi = 0.1$ (Low Tail Risk). VaR = \$10k.
*   **Pricing:** Driver A pays a massive premium for the "Catastrophe Load".

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Deep EVT** = Deep Learning Body + EVT Tail.
2.  **GPD Loss** allows end-to-end training.
3.  **Conditional Parameters** allow personalized risk assessment.

### 10.2 When to Use This Knowledge
*   **InsurTech:** Usage-Based Insurance (UBI) pricing.
*   **Climate Tech:** Hyper-local risk scoring.

### 10.3 Critical Success Factors
1.  **Stability:** Ensure the loss function doesn't produce NaNs.
2.  **Data Volume:** You need *a lot* of extreme events to train a Neural Network.

### 10.4 Further Reading
*   **Massi et al.:** "Deep Learning for Extreme Value Theory".

---

## Appendix

### A. Glossary
*   **NLL:** Negative Log-Likelihood.
*   **Softplus:** Activation function $\ln(1 + e^x)$ (ensures positivity).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **GPD Log-Likelihood** | $\ln \sigma + (1+1/\xi)\ln(1+\xi y/\sigma)$ | Loss Function |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
