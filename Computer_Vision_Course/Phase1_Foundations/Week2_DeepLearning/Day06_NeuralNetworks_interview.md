# Day 6 Interview Questions: Neural Networks

## Q1: Explain the vanishing gradient problem.
**Answer:**
**Problem:** In deep networks with sigmoid/tanh activations, gradients become exponentially small in early layers.

**Mathematical cause:**
$$ \frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial z^{[L]}} \cdot \prod_{l=2}^{L} \frac{\partial z^{[l]}}{\partial z^{[l-1]}} \cdot \frac{\partial z^{[1]}}{\partial W^{[1]}} $$

For sigmoid: $\sigma'(z) \leq 0.25$, so product $\rightarrow 0$ exponentially.

**Solutions:**
1. **ReLU activation:** $f'(z) = 1$ for $z > 0$
2. **Batch normalization:** Normalize layer inputs
3. **Residual connections:** Skip connections (ResNet)
4. **Better initialization:** He/Xavier initialization

## Q2: Why is ReLU preferred over sigmoid?
**Answer:**

| Aspect | Sigmoid | ReLU |
|--------|---------|------|
| **Range** | (0, 1) | [0, ∞) |
| **Gradient** | $\sigma'(z) \leq 0.25$ | 1 (if $z > 0$) |
| **Computation** | Expensive (exp) | Cheap (max) |
| **Vanishing gradient** | Yes | No (for $z > 0$) |
| **Sparsity** | No | Yes (~50% zeros) |

**ReLU problems:**
- **Dead neurons:** If $z \leq 0$ always, gradient = 0
- **Solution:** Leaky ReLU, ELU

## Q3: Derive backpropagation for a 2-layer network.
**Answer:**
**Network:**
$$ z^{[1]} = W^{[1]} x + b^{[1]}, \quad a^{[1]} = \text{ReLU}(z^{[1]}) $$
$$ z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}, \quad \hat{y} = \text{softmax}(z^{[2]}) $$
$$ L = -\sum_c y_c \log \hat{y}_c $$

**Backward:**
$$ \frac{\partial L}{\partial z^{[2]}} = \hat{y} - y \quad \text{(softmax + cross-entropy)} $$

$$ \frac{\partial L}{\partial W^{[2]}} = \frac{1}{m} \frac{\partial L}{\partial z^{[2]}} (a^{[1]})^T $$

$$ \frac{\partial L}{\partial a^{[1]}} = (W^{[2]})^T \frac{\partial L}{\partial z^{[2]}} $$

$$ \frac{\partial L}{\partial z^{[1]}} = \frac{\partial L}{\partial a^{[1]}} \odot \mathbb{1}_{z^{[1]} > 0} \quad \text{(ReLU derivative)} $$

$$ \frac{\partial L}{\partial W^{[1]}} = \frac{1}{m} \frac{\partial L}{\partial z^{[1]}} x^T $$

## Q4: Compare SGD, Momentum, and Adam.
**Answer:**

**SGD:**
$$ \theta_t = \theta_{t-1} - \alpha \nabla L $$
- Simple, but slow convergence
- Oscillates in ravines

**Momentum:**
$$ v_t = \beta v_{t-1} + \nabla L, \quad \theta_t = \theta_{t-1} - \alpha v_t $$
- Dampens oscillations
- Faster convergence
- $\beta \approx 0.9$

**Adam:**
$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L $$
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2 $$
$$ \theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon} $$
- Adaptive learning rates per parameter
- Combines momentum + RMSprop
- Default: $\beta_1=0.9, \beta_2=0.999, \alpha=0.001$

**Recommendation:** Adam for most cases, SGD+Momentum for fine-tuning.

## Q5: What is batch normalization and why does it help?
**Answer:**
**Batch Normalization:** Normalize layer inputs to have mean 0, variance 1.

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$
$$ y_i = \gamma \hat{x}_i + \beta \quad \text{(learnable scale/shift)} $$

**Benefits:**
1. **Reduces internal covariate shift:** Stabilizes layer input distributions
2. **Higher learning rates:** Less sensitive to initialization
3. **Regularization:** Adds noise (batch statistics)
4. **Faster convergence:** 10-14× speedup reported

**Placement:** After linear layer, before activation.

## Q6: Explain dropout and its effect during training vs inference.
**Answer:**
**Dropout:** Randomly set activations to 0 with probability $p$ during training.

**Training:**
```python
mask = (np.random.rand(*A.shape) < keep_prob)
A = A * mask / keep_prob  # Inverted dropout
```

**Inference:**
```python
# No dropout, use all neurons
A = A  # (already scaled during training)
```

**Why it works:**
- Prevents co-adaptation of neurons
- Ensemble effect (training many sub-networks)
- Typical: $p = 0.5$ for hidden layers, $p = 0.1$ for input

**Trade-off:** Slower training (need more epochs).

## Q7: How to initialize weights properly?
**Answer:**

**Xavier/Glorot (tanh/sigmoid):**
$$ W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right) $$

**He initialization (ReLU):**
$$ W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right) $$

**Reason:** Maintain variance of activations and gradients across layers.

```python
# He initialization
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)

# Xavier initialization
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))
```

**Bad initialization:**
- All zeros: Symmetry problem (all neurons learn same thing)
- Too large: Exploding activations/gradients
- Too small: Vanishing activations/gradients

## Q8: What is the universal approximation theorem?
**Answer:**
**Theorem:** A feedforward network with:
- Single hidden layer
- Finite number of neurons
- Non-polynomial activation

can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary accuracy.

**Implications:**
- **Existence:** Neural networks are powerful function approximators
- **Not constructive:** Doesn't tell us how many neurons or how to train
- **Depth helps:** Deep networks can be exponentially more efficient than shallow ones

**Example:** XOR requires 2 hidden neurons, but complex functions may need exponentially many in a single layer vs logarithmic depth.

## Q9: Implement gradient checking.
**Answer:**
```python
def gradient_check(params, grads, X, Y, epsilon=1e-7):
    """Numerical gradient verification."""
    params_flat = np.concatenate([p.flatten() for p in params.values()])
    grads_flat = np.concatenate([g.flatten() for g in grads.values()])
    
    num_grads = np.zeros_like(params_flat)
    
    for i in range(len(params_flat)):
        # Perturb parameter
        params_flat[i] += epsilon
        loss_plus = compute_loss(params_flat, X, Y)
        
        params_flat[i] -= 2 * epsilon
        loss_minus = compute_loss(params_flat, X, Y)
        
        params_flat[i] += epsilon  # Restore
        
        # Numerical gradient
        num_grads[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Relative difference
    diff = np.linalg.norm(num_grads - grads_flat) / \
           (np.linalg.norm(num_grads) + np.linalg.norm(grads_flat))
    
    return diff < 1e-7  # Should be very small
```

## Q10: Why use mini-batch instead of full-batch gradient descent?
**Answer:**

**Full-batch GD:**
- Uses entire dataset per update
- **Pros:** Stable, deterministic
- **Cons:** Slow, doesn't fit in memory for large datasets

**Stochastic GD (batch=1):**
- One sample per update
- **Pros:** Fast updates, can escape local minima
- **Cons:** Noisy, unstable

**Mini-batch GD:**
- Batch size: 32-512
- **Pros:** 
  - Vectorization speedup (GPU)
  - Regularization effect (noise)
  - Faster convergence than full-batch
  - More stable than SGD
- **Cons:** Hyperparameter to tune

**Typical:** Batch size = 32, 64, 128, or 256 (powers of 2 for GPU efficiency).

**Learning rate scaling:** When increasing batch size, increase learning rate proportionally (linear scaling rule).
