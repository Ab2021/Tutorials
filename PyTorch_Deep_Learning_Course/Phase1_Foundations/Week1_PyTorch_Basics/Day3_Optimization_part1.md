# Day 3: Optimization - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Adam Internals, Weight Decay, and Second-Order Methods

## 1. Inside the Adam Optimizer

Adam maintains two state buffers for *every* parameter:
1.  **Momentum ($m_t$)**: Exponential Moving Average (EMA) of gradients.
2.  **Variance ($v_t$)**: EMA of squared gradients.

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$

**Bias Correction**:
Since $m_0, v_0$ are initialized to 0, they are biased towards 0 initially.
$$ \hat{m}_t = m_t / (1-\beta_1^t) $$
$$ \hat{v}_t = v_t / (1-\beta_2^t) $$

**Update Rule**:
$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

This means Adam adapts the step size for each parameter individually based on its gradient history.

## 2. Weight Decay vs L2 Regularization

In standard SGD:
$$ L_{reg}(\theta) = L(\theta) + \frac{\lambda}{2} ||\theta||^2 $$
Gradient: $\nabla L_{reg} = \nabla L + \lambda \theta$.
Update: $\theta \leftarrow \theta - \eta (\nabla L + \lambda \theta) = (1 - \eta \lambda)\theta - \eta \nabla L$.
This is literally decaying the weight by a factor.

**In Adam**:
L2 Regularization (adding to Loss) and Weight Decay (modifying update rule) are **NOT** equivalent!
Because Adam scales the gradient by $1/\sqrt{v_t}$, the L2 term gets scaled too, which is wrong.
**AdamW** (Adam with Weight Decay) fixes this by decoupling weight decay from the gradient update.
**Always use AdamW instead of Adam.**

```python
# Correct way
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

## 3. Second-Order Methods (L-BFGS)

SGD uses only the Gradient (First Derivative).
Newton's Method uses the Hessian (Second Derivative) to jump directly to the minimum (assuming quadratic).
$$ \theta_{t+1} = \theta_t - H^{-1} \nabla L $$

*   **Pros**: Converges in very few steps.
*   **Cons**: Computing inverse Hessian $H^{-1}$ is $O(N^3)$. Impossible for Neural Nets.

**L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)**:
Approximates the inverse Hessian using history of updates.
Supported in PyTorch (`optim.LBFGS`).
*   Used for small problems or style transfer.
*   Requires a closure (function that re-evaluates loss).

```python
optimizer = optim.LBFGS(model.parameters())

def closure():
    optimizer.zero_grad()
    loss = model(input)
    loss.backward()
    return loss

optimizer.step(closure)
```

## 4. Gradient Clipping

To prevent Exploding Gradients (common in RNNs), we clip the norm of the gradient vector.

```python
# Clip global norm of all gradients to max 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
This preserves the *direction* of the gradient but limits its *magnitude*.

## 5. Sparse Optimizers

For Embedding layers with millions of rows, gradients are sparse (only accessed rows have grads).
`optim.SparseAdam` handles sparse tensors efficiently.
