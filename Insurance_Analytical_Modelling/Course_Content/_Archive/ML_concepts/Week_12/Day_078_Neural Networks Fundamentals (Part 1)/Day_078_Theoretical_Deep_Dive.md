# Neural Networks Fundamentals (Part 1) - Theoretical Deep Dive

## Overview
We now enter the world of **Deep Learning**. While GBMs dominate tabular data, Neural Networks are the engines of AI (Images, Text, Speech). This session covers the building blocks: **Perceptrons**, **Activation Functions**, and the magic of **Backpropagation**.

---

## 1. Conceptual Foundation

### 1.1 The Biological Inspiration

*   **Neuron:** Receives signals (dendrites), processes them (soma), and fires an output (axon) if the signal is strong enough.
*   **Perceptron:** The mathematical equivalent.
    $$ y = f(\sum w_i x_i + b) $$
    *   $w_i$: Weights (Synaptic strength).
    *   $b$: Bias (Activation threshold).
    *   $f$: Activation Function.

### 1.2 Multi-Layer Perceptron (MLP)

*   A single perceptron can only solve linear problems.
*   **MLP:** Stacking layers of neurons.
*   **Universal Approximation Theorem:** An MLP with just one hidden layer (and enough neurons) can approximate *any* continuous function.
*   *Actuarial Use:* Modeling highly non-linear mortality curves or claim severity distributions.

### 1.3 Activation Functions

1.  **Sigmoid:** $\sigma(z) = \frac{1}{1+e^{-z}}$.
    *   Squashes output to $(0, 1)$.
    *   *Use:* Output layer for Binary Classification (Fraud vs. Legit).
    *   *Problem:* Vanishing Gradient (derivatives become zero for large inputs).
2.  **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$.
    *   *Use:* Hidden layers.
    *   *Benefit:* Fast, solves Vanishing Gradient.
3.  **Softmax:**
    *   *Use:* Multi-class classification (Preferred, Standard, Substandard).
    *   Converts raw scores (logits) into probabilities that sum to 1.

---

## 2. Mathematical Framework

### 2.1 Forward Propagation

*   Passing the data through the network to get a prediction.
*   **Matrix Form:**
    $$ Z^{[1]} = W^{[1]} X + b^{[1]} $$
    $$ A^{[1]} = g(Z^{[1]}) $$
    $$ Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} $$
    $$ \hat{y} = g(Z^{[2]}) $$
*   It's just a series of Matrix Multiplications and Non-linearities.

### 2.2 Loss Functions

1.  **Binary Cross-Entropy (Log Loss):**
    $$ L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})] $$
    *   Used for probabilities (Bernoulli).
2.  **MSE (Mean Squared Error):**
    $$ L = (y - \hat{y})^2 $$
    *   Used for regression (Gaussian).

### 2.3 Backpropagation (The Chain Rule)

*   **Goal:** Find how much the Loss $L$ changes if we change a weight $w$. (i.e., $\frac{\partial L}{\partial w}$).
*   **Chain Rule:**
    $$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial z} \times \frac{\partial z}{\partial w} $$
*   We calculate the error at the output and "propagate" it backwards to update the weights.

---

## 3. Theoretical Properties

### 3.1 Non-Linearity

*   Why do we need activation functions?
*   If we didn't have them, $f(W_2(W_1 x)) = (W_2 W_1) x = W_{new} x$.
*   A deep network without non-linearities is just a single Linear Regression model.
*   **ReLU** introduces the "kinks" that allow us to model complex shapes.

### 3.2 Gradient Descent

*   **Update Rule:**
    $$ w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} $$
*   We descend the loss landscape to find the minimum (lowest error).

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyTorch Autograd Demo

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define Data (XOR Problem - Non-linear)
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2. Define Model (MLP)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4) # Input 2 -> Hidden 4
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 1) # Hidden 4 -> Output 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN()

# 3. Loss and Optimizer
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training Loop
for epoch in range(1000):
    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward
    optimizer.zero_grad() # Clear old gradients
    loss.backward()       # Calculate new gradients
    optimizer.step()      # Update weights
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Prediction
print("Predictions:", model(X).detach().numpy().round())
```

### 4.2 Manual Backprop (Conceptual)

```python
# Forward
z = w * x + b
a = sigmoid(z)
L = (y - a)**2

# Backward
dL_da = -2 * (y - a)
da_dz = a * (1 - a) # Derivative of sigmoid
dz_dw = x

dL_dw = dL_da * da_dz * dz_dw
```

---

## 5. Evaluation & Validation

### 5.1 The Loss Curve

*   Plot Loss vs. Epochs.
*   **Smooth Drop:** Good learning rate.
*   **Jittery/No Drop:** Learning rate too high or data not normalized.
*   **Slow Drop:** Learning rate too low.

### 5.2 Data Normalization

*   **Critical:** Neural Networks expect inputs to be scaled (Mean 0, Std 1).
*   If "Age" is 50 and "Income" is 50,000, the gradients for Income will explode.
*   *Action:* Always use `StandardScaler` before feeding data to an NN.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Overfitting**
    *   NNs have thousands of parameters. They can memorize anything.
    *   *Fix:* Regularization (Dropout, L2) - covered in Day 79.

2.  **Trap: Local Minima**
    *   The loss landscape is bumpy. Gradient Descent might get stuck in a small valley (Local Minimum) instead of the deepest valley (Global Minimum).
    *   *Reality:* In high dimensions, local minima are rare. Saddle points are the real issue.

### 6.2 Implementation Challenges

1.  **Vanishing Gradients:**
    *   If you use Sigmoid in deep networks, gradients become $0.25 \times 0.25 \times 0.25 \dots \approx 0$. The front layers stop learning.
    *   *Fix:* Use ReLU.

---

## 7. Advanced Topics & Extensions

### 7.1 Deep vs. Wide

*   **Wide:** One hidden layer with 1000 neurons. (Good at memorization).
*   **Deep:** 10 hidden layers with 100 neurons. (Good at generalization and hierarchy).
*   *Trend:* Deep is better.

### 7.2 Initialization

*   If you initialize all weights to 0, the neurons all learn the same thing (Symmetry).
*   *Fix:* Xavier/Glorot Initialization (Random small numbers).

---

## 8. Regulatory & Governance Considerations

### 8.1 Interpretability

*   NNs are the ultimate Black Box.
*   **Integrated Gradients:** A method (like SHAP) to attribute predictions to inputs in NNs.
*   **Regulators:** Skeptical of NNs for pricing, but accepting for Fraud/Claims processing.

---

## 9. Practical Example

### 9.1 Worked Example: The "Fraud" Detector

**Scenario:**
*   Input: 50 features of a claim.
*   Output: Probability of Fraud.
*   **Model:** MLP with 3 hidden layers.
*   **Result:** 95% Accuracy.
*   **Insight:** The model learned that "Claims filed at 2 AM on a Sunday" are highly suspicious. (Non-linear interaction of Time and Day).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **MLP** approximates functions.
2.  **Backpropagation** calculates gradients.
3.  **ReLU** allows deep learning.

### 10.2 When to Use This Knowledge
*   **Unstructured Data:** Images, Text.
*   **Complex Tabular Data:** When GBMs plateau.

### 10.3 Critical Success Factors
1.  **Scale your data.**
2.  **Start small.** Don't build a 100-layer network for a regression problem.

### 10.4 Further Reading
*   **Goodfellow et al.:** "Deep Learning" (The Textbook).
*   **3Blue1Brown:** "Neural Networks" (YouTube Series - Essential visualization).

---

## Appendix

### A. Glossary
*   **Epoch:** One pass through the entire dataset.
*   **Batch:** A chunk of data used for one update.
*   **Logits:** Raw output before activation.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | Activation |
| **ReLU** | $\max(0, z)$ | Activation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
