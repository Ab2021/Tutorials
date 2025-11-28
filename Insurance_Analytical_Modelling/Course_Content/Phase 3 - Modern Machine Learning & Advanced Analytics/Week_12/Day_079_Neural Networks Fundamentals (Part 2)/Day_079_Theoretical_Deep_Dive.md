# Neural Networks Fundamentals (Part 2) - Theoretical Deep Dive

## Overview
Training a Neural Network is an art. You need the right **Optimizer** to find the minimum, **Regularization** to prevent overfitting, and **Batch Normalization** to keep the training stable. This session covers the "Tricks of the Trade" for modern Deep Learning.

---

## 1. Conceptual Foundation

### 1.1 The Optimization Landscape

*   **SGD (Stochastic Gradient Descent):** Taking small steps downhill.
    *   *Problem:* Gets stuck in local minima or saddle points. Slow.
*   **Momentum:** Like a ball rolling down a hill. It builds up speed.
    *   *Benefit:* Plows through small bumps.
*   **Adam (Adaptive Moment Estimation):**
    *   Combines Momentum (First moment) and RMSprop (Second moment).
    *   *Benefit:* Adapts the learning rate for *each parameter* individually.
    *   *Verdict:* The default choice for 99% of problems.

### 1.2 Regularization: Dropout

*   **Idea:** Randomly "kill" (set to zero) a percentage of neurons (e.g., 50%) during each training pass.
*   **Effect:**
    1.  Prevents neurons from co-adapting (relying too much on each other).
    2.  Acts like training an ensemble of $2^N$ different networks.
*   **Test Time:** Turn off Dropout and scale weights by $(1-p)$.

### 1.3 Batch Normalization

*   **Problem:** Internal Covariate Shift.
    *   As weights change, the distribution of inputs to the next layer changes. The next layer has to constantly adapt.
*   **Solution:** Normalize the inputs *inside* the network (Mean 0, Std 1) for every batch.
*   *Benefit:* Allows higher learning rates, faster convergence.

---

## 2. Mathematical Framework

### 2.1 Adam Update Rule

1.  Calculate Gradients $g_t$.
2.  Update Momentum $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$.
3.  Update Variance $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.
4.  Correct Bias $\hat{m}_t, \hat{v}_t$.
5.  Update Weights: $w_t = w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$.

### 2.2 L2 Regularization (Weight Decay)

*   Add $\frac{1}{2} \lambda ||w||^2$ to the Loss.
*   Gradient: $\frac{\partial L}{\partial w} + \lambda w$.
*   Update: $w_{new} = w_{old} - \eta (\text{Gradient} + \lambda w_{old}) = (1 - \eta \lambda) w_{old} - \dots$
*   *Effect:* Decays the weights towards zero at every step.

---

## 3. Theoretical Properties

### 3.1 Learning Rate Schedulers

*   **Constant LR:** Good start, but might oscillate around the minimum.
*   **Step Decay:** Drop LR by 10x every 30 epochs.
*   **ReduceLROnPlateau:** Watch the Validation Loss. If it stops improving, drop the LR. (The "Patience" strategy).
*   **Cyclic LR:** Oscillate LR between a min and max. Helps escape sharp minima.

### 3.2 Batch Size Dynamics

*   **Small Batch (32):** Noisy gradients. Acts as regularization. Good generalization.
*   **Large Batch (1024):** Stable gradients. Faster (GPU parallelization). Can converge to sharp minima (bad generalization).
*   *Actuarial Standard:* 32 to 256.

---

## 4. Modeling Artifacts & Implementation

### 4.1 PyTorch Optimization Demo

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a deeper model with Dropout and BatchNorm
class RobustNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64) # Batch Norm
        self.dropout1 = nn.Dropout(0.5) # 50% Dropout
        
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.5)
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        return self.output(x)

model = RobustNN()

# Optimizer: Adam with Weight Decay (L2)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# Scheduler: Reduce LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training Loop Snippet
# ... inside loop ...
# optimizer.step()
# ... end of epoch ...
# val_loss = validate(model, val_loader)
# scheduler.step(val_loss)
```

### 4.2 Visualizing Dropout

*   **Training:** Output is random (due to dropout).
*   **Evaluation:** Output is deterministic (`model.eval()` turns off dropout).
*   *Tip:* Always remember to switch modes!

---

## 5. Evaluation & Validation

### 5.1 The "Double Descent" Phenomenon

*   Classical Stats: Overfitting gets worse as complexity increases.
*   Deep Learning: Overfitting gets worse, then *better* as model gets huge.
*   *Actuarial Takeaway:* Don't be afraid of large models, provided you use Regularization.

### 5.2 Checking Gradient Flow

*   If gradients are `NaN` or `Inf`, check:
    1.  Learning Rate too high?
    2.  Exploding Gradients? (Use Gradient Clipping).
    3.  Division by zero in loss?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Dropout before Batch Norm**
    *   Usually, the order is: `Linear -> BatchNorm -> ReLU -> Dropout`.
    *   Putting Dropout before BatchNorm shifts the statistics, confusing the BatchNorm layer.

2.  **Trap: Adam vs. SGD for Generalization**
    *   Adam converges fast.
    *   SGD (properly tuned) often generalizes slightly better.
    *   *Start with Adam.* If you need that last 0.1% accuracy, try SGD.

### 6.2 Implementation Challenges

1.  **Reproducibility:**
    *   NN training is non-deterministic (GPU operations).
    *   `torch.manual_seed(42)` helps, but isn't perfect.

---

## 7. Advanced Topics & Extensions

### 7.1 Lookahead Optimizer

*   "k steps forward, 1 step back".
*   Maintains a "Slow Weights" set and "Fast Weights" set.
*   More stable than Adam.

### 7.2 Weight Standardization

*   Normalizing the *weights* (not just inputs).
*   Helps when batch size is very small (Micro-Batching).

---

## 8. Regulatory & Governance Considerations

### 8.1 "Why did the model change?"

*   **Scenario:** You retrain the model on the *same* data, and predictions change by 5%.
*   **Reason:** Random Initialization + GPU non-determinism.
*   **Fix:** Model Versioning. Save the *weights* (`model.pth`), not just the code.

---

## 9. Practical Example

### 9.1 Worked Example: The "Noisy" Claims Data

**Scenario:**
*   Predicting bodily injury severity (very noisy).
*   **Baseline MLP:** Overfits immediately. Val Loss goes up.
*   **Robust MLP:** Added Dropout (0.5) and L2 Regularization.
*   **Result:** Training accuracy dropped, but Validation accuracy improved significantly.
*   **Insight:** The model was forced to ignore the noise and learn the underlying "Physics" of the crash.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Adam** is the go-to optimizer.
2.  **Dropout** prevents memorization.
3.  **Batch Norm** speeds up training.

### 10.2 When to Use This Knowledge
*   **Deep Learning:** Always. These are standard components.

### 10.3 Critical Success Factors
1.  **Monitor LR:** If loss bounces, lower LR.
2.  **Use Callbacks:** Save the best model, not the last model.

### 10.4 Further Reading
*   **Srivastava et al.:** "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".
*   **Ioffe & Szegedy:** "Batch Normalization".

---

## Appendix

### A. Glossary
*   **Momentum:** Velocity of updates.
*   **Scheduler:** LR adjuster.
*   **Covariate Shift:** Changing input distribution.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Weight Decay** | $w_{new} = (1-\eta\lambda)w$ | Regularization |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
