# Day 6 Deep Dive: Backpropagation and Optimization

## 1. Backpropagation Derivation

### Chain Rule Foundation
For composite function $f(g(x))$:
$$ \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} $$

### Computational Graph
**Example:** $L = (w_1 x_1 + w_2 x_2 + b)^2$

```
x1 ──┐
     ├─→ [×] ──┐
w1 ──┘         │
               ├─→ [+] ──┐
x2 ──┐         │         │
     ├─→ [×] ──┘         ├─→ [+] ──→ [²] ──→ L
w2 ──┘                   │
                         │
b ───────────────────────┘
```

**Forward pass:** Compute outputs
**Backward pass:** Compute gradients using chain rule

### Multi-Layer Network

**Network:**
$$ z^{[1]} = W^{[1]} x + b^{[1]} $$
$$ a^{[1]} = \sigma(z^{[1]}) $$
$$ z^{[2]} = W^{[2]} a^{[1]} + b^{[2]} $$
$$ \hat{y} = \sigma(z^{[2]}) $$
$$ L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})] $$

**Backward pass:**
$$ \frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} $$

$$ \frac{\partial L}{\partial z^{[2]}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} = \hat{y} - y $$

$$ \frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial z^{[2]}} \cdot (a^{[1]})^T $$

$$ \frac{\partial L}{\partial b^{[2]}} = \frac{\partial L}{\partial z^{[2]}} $$

$$ \frac{\partial L}{\partial a^{[1]}} = (W^{[2]})^T \cdot \frac{\partial L}{\partial z^{[2]}} $$

$$ \frac{\partial L}{\partial z^{[1]}} = \frac{\partial L}{\partial a^{[1]}} \odot \sigma'(z^{[1]}) $$

$$ \frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial z^{[1]}} \cdot x^T $$

### Vectorized Implementation

```python
class BackpropNetwork:
    """Detailed backpropagation implementation."""
    
    def __init__(self, layer_dims):
        self.params = {}
        self.L = len(layer_dims) - 1
        
        # Initialize parameters
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.params[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    def forward_propagation(self, X):
        """Forward pass with caching."""
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            
            Z = W @ A_prev + b
            
            if l < self.L:
                A = np.maximum(0, Z)  # ReLU
            else:
                A = 1 / (1 + np.exp(-Z))  # Sigmoid
            
            cache = (A_prev, W, b, Z)
            caches.append(cache)
        
        return A, caches
    
    def backward_propagation(self, AL, Y, caches):
        """Backward pass."""
        grads = {}
        m = Y.shape[1]
        
        # Output layer gradient
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Last layer (sigmoid)
        current_cache = caches[self.L - 1]
        A_prev, W, b, Z = current_cache
        
        dZ = AL - Y  # Simplified for sigmoid + binary cross-entropy
        grads[f'dW{self.L}'] = (1/m) * (dZ @ A_prev.T)
        grads[f'db{self.L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ
        
        # Hidden layers (ReLU)
        for l in reversed(range(1, self.L)):
            current_cache = caches[l - 1]
            A_prev, W, b, Z = current_cache
            
            dZ = dA_prev * (Z > 0)  # ReLU derivative
            grads[f'dW{l}'] = (1/m) * (dZ @ A_prev.T)
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                dA_prev = W.T @ dZ
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """Gradient descent update."""
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']
```

## 2. Gradient Checking

**Numerical gradient:**
$$ \frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon} $$

```python
def gradient_check(model, X, Y, epsilon=1e-7):
    """Verify backpropagation implementation."""
    # Compute analytical gradients
    AL, caches = model.forward_propagation(X)
    grads = model.backward_propagation(AL, Y, caches)
    
    # Flatten parameters and gradients
    params_vector = []
    grads_vector = []
    
    for l in range(1, model.L + 1):
        params_vector.append(model.params[f'W{l}'].flatten())
        params_vector.append(model.params[f'b{l}'].flatten())
        grads_vector.append(grads[f'dW{l}'].flatten())
        grads_vector.append(grads[f'db{l}'].flatten())
    
    params_vector = np.concatenate(params_vector)
    grads_vector = np.concatenate(grads_vector)
    
    # Compute numerical gradients
    num_grads = np.zeros_like(params_vector)
    
    for i in range(len(params_vector)):
        theta_plus = params_vector.copy()
        theta_plus[i] += epsilon
        
        theta_minus = params_vector.copy()
        theta_minus[i] -= epsilon
        
        # Compute loss for both
        # (Simplified - need to reshape and set parameters)
        loss_plus = compute_loss_with_params(model, X, Y, theta_plus)
        loss_minus = compute_loss_with_params(model, X, Y, theta_minus)
        
        num_grads[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare
    difference = np.linalg.norm(num_grads - grads_vector) / (np.linalg.norm(num_grads) + np.linalg.norm(grads_vector))
    
    if difference < 1e-7:
        print("✓ Gradient check passed!")
    else:
        print(f"✗ Gradient check failed: difference = {difference}")
    
    return difference
```

## 3. Optimization Algorithms

### Momentum
**Idea:** Accumulate gradients to dampen oscillations.

$$ v_t = \beta v_{t-1} + (1 - \beta) \nabla L $$
$$ \theta_t = \theta_{t-1} - \alpha v_t $$

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v = {}
    
    def update(self, params, grads):
        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])
        
        for key in params:
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * grads[key]
            params[key] -= self.lr * self.v[key]
```

### RMSprop
**Idea:** Adaptive learning rates per parameter.

$$ s_t = \beta s_{t-1} + (1 - \beta) (\nabla L)^2 $$
$$ \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla L $$

```python
class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}
    
    def update(self, params, grads):
        if not self.s:
            for key in params:
                self.s[key] = np.zeros_like(params[key])
        
        for key in params:
            self.s[key] = self.beta * self.s[key] + (1 - self.beta) * grads[key]**2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.s[key]) + self.epsilon)
```

### Adam (Adaptive Moment Estimation)
**Combines momentum + RMSprop:**

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L)^2 $$

**Bias correction:**
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

**Update:**
$$ \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params, grads):
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        
        for key in params:
            # Update biased moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

## 4. Learning Rate Schedules

### Step Decay
$$ \alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / k \rfloor} $$

### Exponential Decay
$$ \alpha_t = \alpha_0 e^{-\lambda t} $$

### Cosine Annealing
$$ \alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t\pi}{T})) $$

```python
class CosineAnnealingLR:
    def __init__(self, lr_max, lr_min, T_max):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
    
    def get_lr(self, epoch):
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max))
```

## 5. Regularization

### L2 Regularization (Weight Decay)
$$ L_{total} = L_{data} + \frac{\lambda}{2m} \sum_l ||W^{[l]}||_F^2 $$

**Gradient:**
$$ \frac{\partial L_{total}}{\partial W} = \frac{\partial L_{data}}{\partial W} + \frac{\lambda}{m} W $$

### Dropout
**Training:** Randomly set activations to 0 with probability $p$.
$$ a^{[l]} = a^{[l]} \odot mask, \quad mask \sim \text{Bernoulli}(1-p) $$

**Inference:** Scale activations by $(1-p)$ or use inverted dropout during training.

```python
def dropout_forward(A, keep_prob=0.8, training=True):
    """Dropout forward pass."""
    if training:
        mask = (np.random.rand(*A.shape) < keep_prob).astype(float)
        A = A * mask / keep_prob  # Inverted dropout
        return A, mask
    else:
        return A, None

def dropout_backward(dA, mask, keep_prob=0.8):
    """Dropout backward pass."""
    return dA * mask / keep_prob
```

## Summary
Backpropagation efficiently computes gradients via the chain rule. Modern optimizers (Adam) and techniques (dropout, learning rate schedules) enable effective training of deep networks.
