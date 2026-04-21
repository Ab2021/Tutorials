# ⚡ Easy: Activation Functions from Scratch
> **Platforms:** Deep-ML + TensorTonic | **Difficulty:** Easy | **Must Know Cold ❄️**

---

## WHY ACTIVATION FUNCTIONS MATTER

Without nonlinear activations, a neural network — no matter how many layers — collapses to a single linear transformation. Activations introduce the nonlinearity that allows networks to learn complex patterns.

**Interview one-liner:** *"Activation functions introduce nonlinearity. Without them, stacking linear layers is mathematically equivalent to a single linear layer."*

---

---

## 🟢 PROBLEM 1: Sigmoid

### Theory
Maps any real number to (0, 1). Used in binary classification output layers.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Gradient (for backprop):**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Problems:**
- **Vanishing gradient:** When |x| is large, σ'(x) ≈ 0. Gradients vanish in deep networks.
- **Not zero-centered:** Outputs always positive → all gradients same sign → zigzag optimization.
- **Computationally expensive:** exp() is slow.

### Things to Focus On
- ✅ Numerical stability: use `1 / (1 + exp(-x))` not `exp(x) / (1 + exp(x))`
- ✅ Use `np.clip(x, -500, 500)` to prevent overflow in naive implementations
- ✅ Know when NOT to use sigmoid: hidden layers of deep nets (prefer ReLU)
- ✅ When to use: binary classification output, attention gates in LSTM

### Implementation
```python
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    
    Numerically stable implementation using:
    - For x >= 0: 1 / (1 + exp(-x))
    - For x < 0:  exp(x) / (1 + exp(x))  ← avoids overflow of exp(-x) when x is very negative
    
    Args:
        x: Input array of any shape
    Returns:
        Array of same shape with values in (0, 1)
    """
    # Simple version (fine for most interview answers):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    # Numerically optimal version (mention this if asked about stability):
    # pos_mask = x >= 0
    # result = np.zeros_like(x, dtype=float)
    # result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    # exp_x = np.exp(x[~pos_mask])
    # result[~pos_mask] = exp_x / (1 + exp_x)
    # return result

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative for backprop: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

# Test
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(sigmoid(x))   # [0.119, 0.269, 0.5, 0.731, 0.881]
print(sigmoid(0))   # 0.5 — always passes this sanity check
```

---

## 🟢 PROBLEM 2: ReLU (Rectified Linear Unit)

### Theory
$$\text{ReLU}(x) = \max(0, x)$$

**Gradient:**
$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**Advantages over sigmoid:**
- No vanishing gradient for positive values
- Computationally very fast (just a comparison)
- Sparse activation: ~50% neurons inactive → efficient

**Problems:**
- **Dying ReLU:** Neurons can permanently output 0 if weights become very negative. Those neurons never recover (gradient = 0 → no weight update).
- **Not zero-centered**

### Things to Focus On
- ✅ Know the dying ReLU problem and its fix (Leaky ReLU, batch norm)
- ✅ Gradient is undefined at x=0; convention: set to 0 (or 1, doesn't matter practically)
- ✅ Most common activation in hidden layers of modern networks

### Implementation
```python
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x). Most common hidden layer activation."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Gradient: 1 if x > 0, else 0.
    Note: at x=0, gradient is undefined; convention sets it to 0.
    """
    return (x > 0).astype(float)

# Test
x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print(relu(x))             # [0. 0. 0. 1. 3.]
print(relu_derivative(x))  # [0. 0. 0. 1. 1.]
```

---

## 🟢 PROBLEM 3: Leaky ReLU

### Theory
Fixes the Dying ReLU problem by allowing a small gradient when x < 0.

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

Typical α = 0.01. **Parametric ReLU (PReLU)** learns α during training.

### Things to Focus On
- ✅ Always mention WHY you'd choose it over ReLU (dying neuron fix)
- ✅ α is a hyperparameter (usually 0.01); if learned → PReLU
- ✅ Still not zero-centered, but avoids dead neurons

### Implementation
```python
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    LeakyReLU: x if x > 0, else alpha * x.
    Default alpha = 0.01 (1% leak).
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Gradient: 1 if x > 0, else alpha."""
    return np.where(x > 0, 1.0, alpha)

# Test
x = np.array([-2.0, 0.0, 2.0])
print(leaky_relu(x))             # [-0.02  0.    2.  ]
print(leaky_relu_derivative(x))  # [0.01  0.01  1.  ]
```

---

## 🟢 PROBLEM 4: Tanh (Hyperbolic Tangent)

### Theory
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

Range: (-1, 1). **Zero-centered** → faster convergence than sigmoid.

**Gradient:**
$$\tanh'(x) = 1 - \tanh^2(x)$$

Still has vanishing gradient for large |x|.

### Things to Focus On
- ✅ Zero-centered (advantage over sigmoid)
- ✅ Still has vanishing gradient (disadvantage vs ReLU)
- ✅ Used in: LSTM gates, GRU gates, output layer for regression with (-1,1) range
- ✅ Relationship to sigmoid: tanh(x) = 2σ(2x) - 1

### Implementation
```python
def tanh_activation(x: np.ndarray) -> np.ndarray:
    """
    Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    numpy has np.tanh built in, but show the formula:
    """
    return np.tanh(x)  # numpy's implementation is numerically stable

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Gradient: 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2

# Test
x = np.array([-2.0, 0.0, 2.0])
print(tanh_activation(x))   # [-0.964  0.     0.964]
print(tanh_derivative(x))   # [ 0.071  1.     0.071]  ← vanishing at extremes
```

---

## 🟢 PROBLEM 5: Softmax

### Theory
Converts a vector of raw scores (logits) into a probability distribution.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Properties:**
- Outputs sum to 1
- All outputs in (0, 1)
- Amplifies the largest value (soft argmax)

**Numerical stability trick:** Subtract max before exponentiation.
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

### Things to Focus On
- ✅ ALWAYS subtract max for numerical stability (prevents overflow)
- ✅ Softmax + Cross-Entropy is used together — know the combined gradient
- ✅ Combined gradient of softmax CE: `ŷ - y` (very clean)
- ✅ Temperature: `softmax(x/T)` — T→0 makes it argmax, T→∞ makes it uniform
- ✅ Axis matters in implementation: `axis=-1` for last dimension (typical)

### Implementation
```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    The -max(x) trick is CRITICAL for numerical stability.
    Without it: softmax([1000, 1001]) → exp(1000) = overflow (inf)
    With it:    softmax([1000, 1001]) → softmax([-1, 0]) = [0.269, 0.731] ✓
    """
    # Subtract max for numerical stability (doesn't change the output mathematically)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Test
logits = np.array([1.0, 2.0, 3.0])
probs = softmax(logits)
print(probs)       # [0.090, 0.245, 0.665]
print(probs.sum()) # 1.0 ✓

# Batched (2 samples, 3 classes)
batch = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
print(softmax(batch, axis=-1))
# [[0.090, 0.245, 0.665],
#  [0.333, 0.333, 0.333]]
```

---

## 🟢 PROBLEM 6: ELU (Exponential Linear Unit)

### Theory
$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

α is usually 1.0. **Advantages:**
- Zero-centered for negative inputs (smooth approach to 0)
- Pushes mean activation closer to zero → faster convergence
- No dead neurons (like Leaky ReLU)
- Saturates for very negative inputs → robustness to noise

### Things to Focus On
- ✅ At x=0: ELU(0) = 0 and is continuous AND differentiable (smooth at 0)
- ✅ More expensive than ReLU (exp computation)
- ✅ Gradient at x=0 from left: α (continuity ensured)

### Implementation
```python
def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    ELU: x if x > 0, else alpha * (exp(x) - 1)
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Gradient: 1 if x > 0, else ELU(x) + alpha = alpha * exp(x)
    """
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)

# Test
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(elu(x))    # [-0.865, -0.632,  0., 1., 2.]
```

---

## 🟢 PROBLEM 7: SELU (Scaled ELU)

### Theory
$$\text{SELU}(x) = \lambda \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

where λ ≈ 1.0507, α ≈ 1.6733 (derived mathematically).

**Key insight:** SELU is **self-normalizing** — with proper weight init (LeCun normal), activations maintain mean≈0 and variance≈1 throughout the network, eliminating the need for Batch Normalization.

### Things to Focus On
- ✅ The specific λ and α values are fixed constants (not hyperparameters)
- ✅ Requires LeCun normal weight initialization to work correctly
- ✅ Only works well with fully connected layers; less common with CNNs

### Implementation
```python
def selu(x: np.ndarray) -> np.ndarray:
    """
    SELU with fixed constants λ and α derived to achieve self-normalization.
    """
    alpha = 1.6732632423543772
    lam   = 1.0507009873554805
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

---

## 🟢 PROBLEM 8: GELU (Gaussian Error Linear Unit)

### Theory
$$\text{GELU}(x) = x \cdot \Phi(x)$$

where Φ(x) is the CDF of the standard normal distribution.

**Approximation** (used in practice):
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**Why GELU?** Used in BERT, GPT-2, GPT-3, and most modern transformers. It's a smooth, stochastic version of ReLU that weights inputs by their probability of being positive.

### Things to Focus On
- ✅ GELU ≈ ReLU × probability of being positive (probabilistic interpretation)
- ✅ Smooth (differentiable everywhere unlike ReLU)
- ✅ Used in virtually all modern LLMs — know this one!
- ✅ At x=0: GELU(0) = 0; unlike ReLU, smoothly approaches 0

### Implementation
```python
from scipy.special import erf

def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU: x * Phi(x) where Phi is the standard normal CDF.
    Two versions: exact and fast approximation.
    """
    # Exact version (uses error function)
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))

def gelu_approximate(x: np.ndarray) -> np.ndarray:
    """
    Fast approximation used in transformers (from original BERT paper).
    Error < 0.0001 for all x.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Test — both should give very similar results
x = np.array([-1.0, 0.0, 1.0, 2.0])
print(gelu(x))             # [-0.159,  0.,  0.841,  1.954]
print(gelu_approximate(x)) # [-0.159,  0.,  0.841,  1.954]  ✓ close
```

---

## 🟢 PROBLEM 9: Swish

### Theory
$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

With β=1 (default): **Swish(x) = x · σ(x)**

Discovered by Google Brain via automated search. Used in EfficientNet, MobileNetV3.

**Properties:**
- Smooth, non-monotonic (can decrease before increasing)
- Unbounded above, bounded below (≈ -0.28 at minimum)
- Self-gated: the sigmoid gate is derived from x itself

### Things to Focus On
- ✅ When β→∞: Swish → ReLU
- ✅ When β=0: Swish(x) = x/2 (linear)
- ✅ Non-monotonic → can model more complex functions than ReLU

### Implementation
```python
def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish: x * sigmoid(beta * x)
    Default beta=1.
    """
    return x * sigmoid(beta * x)

# Test
x = np.array([-2.0, 0.0, 2.0])
print(swish(x))  # [-0.238,  0.,  1.762]
```

---

## 📊 QUICK REFERENCE TABLE

| Activation | Range | Zero-centered | Vanishing Gradient | Dead Neurons | Use Case |
|---|---|---|---|---|---|
| Sigmoid | (0,1) | ❌ | ✅ Yes | ❌ | Binary output, LSTM gates |
| Tanh | (-1,1) | ✅ | ✅ Yes | ❌ | LSTM/GRU gates |
| ReLU | [0,∞) | ❌ | ❌ (positive) | ✅ Yes | Hidden layers default |
| Leaky ReLU | (-∞,∞) | ❌ | ❌ | ❌ | When dying ReLU is concern |
| ELU | (-α,∞) | ~✅ | ❌ | ❌ | Better than Leaky ReLU |
| SELU | (-λα,∞) | ✅ self-norm | ❌ | ❌ | Self-normalizing networks |
| GELU | unbounded | ~✅ | ❌ | ❌ | Transformers, BERT, GPT |
| Swish | unbounded | ~✅ | ❌ | ❌ | EfficientNet, MobileNet |
| Softmax | (0,1) sum=1 | ❌ | — | — | Multi-class output only |

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"Why do we subtract max in softmax?"** → Numerical stability: prevents exp(x) overflow for large x.
2. **"Why is ReLU preferred over sigmoid in hidden layers?"** → No vanishing gradient, computationally fast, sparse activation.
3. **"What is the dying ReLU problem?"** → Neurons with negative inputs always output 0, receive 0 gradient, never recover.
4. **"When would you use GELU vs ReLU?"** → GELU for transformers/NLP (smooth, probabilistic); ReLU for CNNs (fast, simple).
5. **"What's special about SELU?"** → Self-normalizing. With LeCun init, no BatchNorm needed.
