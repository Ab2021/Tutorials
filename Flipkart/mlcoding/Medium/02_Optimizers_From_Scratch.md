# 🟡 Medium: Optimizers from Scratch
> **Platform:** TensorTonic + Deep-ML | **Difficulty:** Medium | **⭐ DMM + HO Round Critical**

---

## WHY OPTIMIZERS MATTER IN INTERVIEWS

Optimizers are where theory (gradient descent) meets practice (numerics, adaptive learning rates, momentum). Flipkart's DMM round explicitly tests your ability to derive and implement these. TensorTonic has dedicated problems for every major optimizer.

---

## 🟡 PROBLEM 1: SGD with Momentum

### Theory
**Plain SGD:** $w \leftarrow w - \alpha g_t$

**SGD + Momentum:** Accumulates a velocity vector in directions of persistent gradient.

$$v_t = \beta v_{t-1} + g_t$$
$$w_t = w_{t-1} - \alpha v_t$$

Typical β = 0.9. **Intuition:** Like a ball rolling downhill with momentum — builds up speed in consistent directions, dampens oscillations.

**Nesterov Momentum (NAG):** "Look ahead" before computing gradient.
$$v_t = \beta v_{t-1} + g(w_{t-1} - \alpha \beta v_{t-1})$$
$$w_t = w_{t-1} - \alpha v_t$$

Converges faster than standard momentum (closer to optimal step).

### Things to Focus On
- ✅ Momentum β=0.9 means 90% of previous velocity carries forward
- ✅ Learning rate effectively becomes α/(1-β) for consistent gradients
- ✅ Nesterov: gradient computed at "lookahead" position, not current
- ✅ SGD+Momentum can escape shallow local minima (due to accumulated velocity)
- ✅ Still requires careful LR tuning unlike Adam

### Implementation
```python
import numpy as np

class SGDMomentum:
    """
    SGD with optional Nesterov momentum.
    
    Standard: v = β*v + g; w -= lr * v
    Nesterov:  g computed at w - lr*β*v (lookahead position)
    """
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, 
                 nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = {}  # param_id → velocity array
    
    def step(self, params: list, grads: list) -> list:
        """
        Update parameters.
        
        params: list of parameter arrays
        grads: list of gradient arrays (same order)
        Returns: list of updated parameter arrays
        """
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(p)
            
            v = self.velocity[i]
            
            if self.nesterov:
                # Nesterov: update velocity first, then compute effective update
                v_prev = v.copy()
                v = self.momentum * v + g
                p_new = p - self.lr * (self.momentum * v + (1 - self.momentum) * g)
                # Equivalently: p -= lr * (-momentum * v_prev + (1+momentum) * v)
            else:
                # Standard momentum
                v = self.momentum * v + g
                p_new = p - self.lr * v
            
            self.velocity[i] = v
            updated.append(p_new)
        
        return updated
```

---

## 🟡 PROBLEM 2: Adagrad

### Theory
Adapts learning rate per parameter based on historical gradients.

$$G_t = G_{t-1} + g_t^2 \quad (\text{sum of squared gradients})$$
$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{G_t + \epsilon}} g_t$$

**Key insight:** Parameters that receive large gradients get smaller learning rates; sparse parameters that rarely update get larger effective LR.

**Problem:** $G_t$ keeps growing → learning rate shrinks to zero → learning stops.

### Things to Focus On
- ✅ Good for sparse features (NLP, embeddings) — rarely updated features get boosted LR
- ✅ Problem: learning rate becomes vanishingly small over time (G_t is monotone increasing)
- ✅ RMSProp fixes this with exponential decay instead of accumulation
- ✅ ε: typically 1e-8 to prevent division by zero

### Implementation
```python
class Adagrad:
    """
    Adagrad: accumulate squared gradients, adapt LR per parameter.
    Good for sparse problems; LR decays over time.
    """
    
    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.G = {}  # Accumulated squared gradients
    
    def step(self, params: list, grads: list) -> list:
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.G:
                self.G[i] = np.zeros_like(p)
            
            # Accumulate squared gradient (monotonically increasing!)
            self.G[i] += g ** 2
            
            # Adaptive update
            p_new = p - (self.lr / np.sqrt(self.G[i] + self.eps)) * g
            updated.append(p_new)
        
        return updated
```

---

## 🟡 PROBLEM 3: RMSProp

### Theory
Fixes Adagrad's decaying LR by using exponential moving average of squared gradients:

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t^2$$
$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

Typical ρ = 0.9. The window effectively looks at recent gradients, not all historical ones.

### Things to Focus On
- ✅ RMSProp divides by EMA of squared gradients (not cumulative sum like Adagrad)
- ✅ Stable LR throughout training (doesn't decay to zero)
- ✅ Good for non-stationary problems (RNNs)
- ✅ ρ = decay rate (0.9-0.99); similar to momentum in SGD

### Implementation
```python
class RMSProp:
    """
    RMSProp: exponential moving average of squared gradients.
    Fixes Adagrad's vanishing LR problem.
    """
    
    def __init__(self, lr: float = 0.001, rho: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.E_g2 = {}  # EMA of squared gradients
    
    def step(self, params: list, grads: list) -> list:
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.E_g2:
                self.E_g2[i] = np.zeros_like(p)
            
            # Exponential moving average of squared gradients
            self.E_g2[i] = self.rho * self.E_g2[i] + (1 - self.rho) * g ** 2
            
            # Adaptive update
            p_new = p - (self.lr / np.sqrt(self.E_g2[i] + self.eps)) * g
            updated.append(p_new)
        
        return updated
```

---

## 🔥 PROBLEM 4: Adam (MUST KNOW COLD)

### Theory
Adam = **Adaptive Moment Estimation**. Combines:
1. Momentum (first moment / mean of gradients)
2. RMSProp (second moment / EMA of squared gradients)
3. Bias correction (accounts for initialization at zero)

**Algorithm:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(2nd moment)}$$

**Bias correction** (critical in early steps when m and v are near zero):
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update:**
$$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Defaults (Kingma & Ba 2015):** α=0.001, β₁=0.9, β₂=0.999, ε=1e-8

### Things to Focus On
- ✅ WHY bias correction: m₀=v₀=0 → first steps are biased toward zero → correct by dividing by (1-βᵗ)
- ✅ β₁^t → 0 fast (0.9^100 ≈ 0.00003) → bias correction disappears quickly
- ✅ Adam bad for generalization? Yes — SGD+Momentum can generalize better but Adam converges faster. Use Adam for LLMs, SGD for image classifiers.
- ✅ AdamW = Adam + decoupled weight decay (fixes Adam's L2 regularization issue)
- ✅ ε is typically 1e-8; some papers use 1e-6 for stability

### Implementation
```python
class Adam:
    """
    Adam optimizer: combines momentum + RMSProp + bias correction.
    
    Defaults from Kingma & Ba (2015):
    - lr = 0.001
    - beta1 = 0.9 (momentum decay)
    - beta2 = 0.999 (RMS decay) 
    - eps = 1e-8
    """
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m = {}   # 1st moment (mean of grads)
        self.v = {}   # 2nd moment (EMA of squared grads)
        self.t = 0    # Timestep (for bias correction)
    
    def step(self, params: list, grads: list) -> list:
        """
        Perform one Adam update step.
        
        params: list of parameter arrays
        grads: list of gradient arrays
        Returns: list of updated parameter arrays
        """
        self.t += 1  # Increment timestep
        updated = []
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Initialize moments
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            
            # === STEP 1: Update biased first moment (momentum) ===
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # === STEP 2: Update biased second moment (RMSProp) ===
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            # === STEP 3: Bias correction ===
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # === STEP 4: Parameter update ===
            p_new = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updated.append(p_new)
        
        return updated
    
    def reset(self):
        """Reset moments and timestep."""
        self.m.clear()
        self.v.clear()
        self.t = 0


class AdamW(Adam):
    """
    AdamW: Adam with DECOUPLED weight decay.
    
    Standard Adam applies weight decay through the gradient (L2 regularization),
    which interacts with the adaptive learning rate scaling.
    
    AdamW decouples weight decay from gradient update:
        w -= lr * (adam_update + weight_decay * w)
    
    This is the correct way to add L2 regularization to Adam.
    Used by most modern transformers (BERT, GPT).
    """
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8, 
                 weight_decay: float = 0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay
    
    def step(self, params: list, grads: list) -> list:
        updated = super().step(params, grads)  # Get Adam updates
        
        # Apply weight decay SEPARATELY (decoupled)
        return [p_new - self.lr * self.weight_decay * p 
                for p_new, p in zip(updated, params)]


class Nadam(Adam):
    """
    Nadam: Adam + Nesterov momentum.
    Uses lookahead gradient for the momentum term.
    """
    
    def step(self, params: list, grads: list) -> list:
        self.t += 1
        updated = []
        
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Nesterov term: use beta1 * m_hat + (1-beta1)*g/correction
            m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * g / (1 - self.beta1 ** self.t)
            
            p_new = p - self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)
            updated.append(p_new)
        
        return updated


# === DEMONSTRATION: Compare all optimizers on a simple quadratic loss ===
def demo_optimizers():
    """
    Minimize f(w) = w² (minimum at w=0) starting from w=10.
    Shows convergence behavior of each optimizer.
    """
    def loss_and_grad(w):
        return w ** 2, 2 * w  # f(w), f'(w)
    
    optimizers = {
        'SGD':      SGDMomentum(lr=0.1),
        'SGD+Momentum': SGDMomentum(lr=0.1, momentum=0.9),
        'Adagrad':  Adagrad(lr=0.5),
        'RMSProp':  RMSProp(lr=0.1),
        'Adam':     Adam(lr=0.1),
        'AdamW':    AdamW(lr=0.1, weight_decay=0.01),
    }
    
    for name, opt in optimizers.items():
        w = np.array([10.0])
        for step in range(100):
            loss, grad = loss_and_grad(w)
            w = opt.step([w], [grad])[0]
        print(f"{name:20s}: w = {w[0]:.6f}, loss = {w[0]**2:.8f}")

demo_optimizers()
# All should converge to w ≈ 0
```

---

## 📊 OPTIMIZER COMPARISON TABLE

| Optimizer | Learning Rate | Adaptive? | Momentum | Key Hyperparams | Best For |
|---|---|---|---|---|---|
| SGD | Fixed | ❌ | ❌ | lr | Simple, needs careful tuning |
| SGD+Momentum | Fixed | ❌ | ✅ | lr, β | Image classification, CNNs |
| Nesterov | Fixed | ❌ | ✅ lookahead | lr, β | Faster than standard momentum |
| Adagrad | Decreasing | ✅ | ❌ | lr | Sparse features, NLP |
| RMSProp | Stable | ✅ | ❌ | lr, ρ | RNNs, non-stationary |
| Adam | Stable | ✅ | ✅ | lr, β₁, β₂, ε | Default for most tasks |
| AdamW | Stable | ✅ | ✅ | + weight_decay | Transformers, BERT, GPT |
| Nadam | Stable | ✅ | ✅ Nesterov | lr, β₁, β₂, ε | Sometimes better than Adam |

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"Why does Adam need bias correction?"** → m₀=v₀=0. Without correction, first steps underestimate the true gradient. (1-β^t) corrects this until t is large enough that β^t ≈ 0.
2. **"What's wrong with Adam for generalization?"** → Adaptive learning rates can cause model to fit memorizable patterns. SGD+Momentum with good LR schedule often generalizes better.
3. **"Why AdamW instead of Adam with L2 regularization?"** → In Adam, L2 gradient gets scaled by the same adaptive LR as other gradients. AdamW's decoupled decay treats all weights equally regardless of gradient history.
4. **"What happens if epsilon in Adam is too large?"** → Effectively reduces the adaptive component — approaches SGD with momentum. If too small, can cause numerical instability when v_hat ≈ 0.
