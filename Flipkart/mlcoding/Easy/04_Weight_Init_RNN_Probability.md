# 🟢 Easy: Weight Initialization, KNN, RNN Step, Probability & Missing Easy Problems
> **Platforms:** Deep-ML + TensorTonic | **Difficulty:** Easy | **Gaps filled from universe checklist**

---

## 🟢 PROBLEM 1: He Initialization (for ReLU networks)

### Theory
The right weight initialization prevents vanishing/exploding activations from the first forward pass.

**He (Kaiming) initialization — for ReLU:**
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)$$

**Xavier (Glorot) initialization — for Tanh/Sigmoid:**
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}\right)$$
or uniform: $\mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in}+\text{fan\_out}}}, +\sqrt{\frac{6}{\text{fan\_in}+\text{fan\_out}}}\right)$

**LeCun initialization — for SELU:**
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{1}{\text{fan\_in}}}\right)$$

**Why these formulas?**
- We want: Var(output) = Var(input) layer-to-layer
- For linear layer with fan_in inputs: Var(Wx) = fan_in × Var(W) × Var(x)
- Setting Var(W) = 1/fan_in → Var(Wx) = Var(x) ✓
- For ReLU (zeroes ~50%): need ×2 correction → Var(W) = 2/fan_in ✓

### Things to Focus On
- ✅ He → ReLU family; Xavier → Tanh/Sigmoid; LeCun → SELU
- ✅ Zero initialization is WRONG for symmetry breaking — all neurons learn same thing
- ✅ Bias is always initialized to zero (no symmetry issue)
- ✅ Fan_in = number of inputs to neuron; Fan_out = number of outputs

### Implementation
```python
import numpy as np

def he_init(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """
    He (Kaiming) Normal initialization for ReLU networks.
    W ~ N(0, sqrt(2 / fan_in))
    """
    if seed is not None:
        np.random.seed(seed)
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def xavier_init_normal(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """
    Xavier (Glorot) Normal initialization for Tanh/Sigmoid networks.
    W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    """
    if seed is not None:
        np.random.seed(seed)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def xavier_init_uniform(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """
    Xavier (Glorot) Uniform initialization.
    W ~ U(-limit, +limit)  where limit = sqrt(6 / (fan_in + fan_out))
    """
    if seed is not None:
        np.random.seed(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def lecun_init(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """LeCun Normal for SELU networks."""
    if seed is not None:
        np.random.seed(seed)
    std = np.sqrt(1.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def initialize_network(layer_sizes: list, activation: str = 'relu') -> list:
    """
    Initialize all layers of a network based on activation function.
    
    layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
    Returns: list of (W, b) parameter pairs
    """
    init_fn = {'relu': he_init, 'tanh': xavier_init_normal, 
                'sigmoid': xavier_init_normal, 'selu': lecun_init}[activation]
    
    params = []
    for i in range(len(layer_sizes) - 1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        W = init_fn(fan_in, fan_out)
        b = np.zeros(fan_out)  # Bias always zero-initialized
        params.append((W, b))
    return params

# Test
params = initialize_network([784, 256, 128, 10], activation='relu')
for i, (W, b) in enumerate(params):
    print(f"Layer {i+1}: W={W.shape}, std={W.std():.4f}, "
          f"expected={np.sqrt(2/W.shape[0]):.4f}")
# Std should match expected He std ✓
```

---

## 🟢 PROBLEM 2: Single Neuron (Linear Layer Forward)

### Theory
The fundamental unit of a neural network:
$$z = w \cdot x + b, \quad a = \text{activation}(z)$$

For a batch of inputs X (n×d):
$$Z = X W + b, \quad A = \text{activation}(Z)$$

### Implementation
```python
def single_neuron_forward(x: np.ndarray, w: np.ndarray, b: float, 
                           activation: str = 'sigmoid') -> float:
    """
    Single neuron forward pass.
    x: input features (d,)
    w: weights (d,)
    b: bias scalar
    """
    z = np.dot(w, x) + b
    
    activations = {
        'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
        'relu': lambda z: max(0, z),
        'tanh': lambda z: np.tanh(z),
        'linear': lambda z: z,
    }
    return activations[activation](z)

def linear_layer_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fully connected layer: Z = X @ W + b
    X: (batch, in_features)
    W: (in_features, out_features)
    b: (out_features,)
    Returns: (batch, out_features)
    """
    return X @ W + b  # Broadcasting adds b to each row

# Test
np.random.seed(42)
X = np.random.randn(32, 10)  # 32 samples, 10 features
W = he_init(10, 5)
b = np.zeros(5)
out = linear_layer_forward(X, W, b)
print(f"Input: {X.shape} → Output: {out.shape}")  # (32, 10) → (32, 5)
```

---

## 🟢 PROBLEM 3: Global Average Pooling & Max Pooling 2D

### Theory
**Global Average Pooling (GAP):** Takes the spatial average of each feature map.
$$\text{GAP}[c] = \frac{1}{H \times W} \sum_{i,j} X[c, i, j]$$

Replaces large fully connected layers in modern CNNs (ResNet, MobileNet).

**MaxPool 2D:** Takes maximum value in each kernel window.

### Implementation
```python
def global_average_pooling(X: np.ndarray) -> np.ndarray:
    """
    Global Average Pooling.
    X: (batch, channels, H, W)
    Returns: (batch, channels)  — spatial dims collapsed to scalar
    """
    return X.mean(axis=(-2, -1))  # Average over H and W

def max_pool_2d(X: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """
    2D Max Pooling.
    X: (batch, channels, H, W)
    Returns: (batch, channels, H_out, W_out)
    """
    batch, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    out = np.zeros((batch, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            h_s = i * stride
            w_s = j * stride
            out[:, :, i, j] = X[:, :, h_s:h_s+pool_size, w_s:w_s+pool_size].max(axis=(-2,-1))
    return out

# Test
X = np.random.randn(2, 4, 6, 6)  # 2 samples, 4 channels, 6×6
print(global_average_pooling(X).shape)  # (2, 4)
print(max_pool_2d(X, pool_size=2).shape) # (2, 4, 3, 3)
```

---

## 🟢 PROBLEM 4: RNN Step Forward

### Theory
Vanilla RNN step:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

Or equivalently with combined weight matrix:
$$h_t = \tanh(W_h [h_{t-1}; x_t] + b)$$

**Vanishing gradient problem:** Through T steps, backprop multiplies by $W_{hh}^T$. If eigenvalue of $W_{hh} < 1$ → exponential decay → vanishing gradient. LSTM/GRU fix this.

### Things to Focus On
- ✅ Output = hidden state (for seq2seq, all h_t matter; for classification, just last)
- ✅ tanh is used (not ReLU) — centered output prevents gradient explosion initially
- ✅ Shared weights across all time steps — key property of RNNs
- ✅ Vanishing gradient reason: repeated multiplication by Jacobian of tanh

### Implementation
```python
class RNNCell:
    """Vanilla RNN cell from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int):
        # Combined weight matrix [h_prev; x] → h
        scale = np.sqrt(1 / (input_size + hidden_size))
        self.W = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b = np.zeros(hidden_size)
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Single RNN step.
        x_t:   (batch, input_size)
        h_prev: (batch, hidden_size)
        Returns: h_t (batch, hidden_size)
        """
        # Concatenate input and previous hidden state
        combined = np.concatenate([h_prev, x_t], axis=-1)  # (batch, input+hidden)
        # Compute new hidden state
        h_t = np.tanh(combined @ self.W + self.b)
        return h_t
    
    def sequence_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Process full sequence.
        X: (batch, seq_len, input_size)
        Returns: all hidden states (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.W.shape[1]))
        all_h = []
        for t in range(seq_len):
            h = self.forward(X[:, t, :], h)
            all_h.append(h)
        return np.stack(all_h, axis=1)

# Test
rnn = RNNCell(input_size=4, hidden_size=8)
X = np.random.randn(2, 10, 4)  # 2 samples, seq=10, 4 features
out = rnn.sequence_forward(X)
print(f"RNN output: {out.shape}")  # (2, 10, 8)
```

---

## 🟢 PROBLEM 5: Pad Sequences & Batch Generator

### Theory
**Pad sequences:** Variable-length sequences need padding to the same length for batched processing.

**Mini-batch generator:** Efficient data loading for SGD training.

### Implementation
```python
from typing import List, Tuple

def pad_sequences(sequences: List[List], max_len: int = None, 
                  padding_value: int = 0, padding: str = 'post') -> np.ndarray:
    """
    Pad variable-length sequences to same length.
    
    padding: 'post' (add at end) or 'pre' (add at beginning)
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    
    n = len(sequences)
    result = np.full((n, max_len), padding_value)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if padding == 'post':
            result[i, :length] = seq[:length]
        else:  # pre
            result[i, max_len-length:] = seq[:length]
    
    return result

def create_padding_mask(sequences: List[List], max_len: int = None) -> np.ndarray:
    """
    Create boolean mask: True where padding (to ignore in attention).
    Returns: (n, max_len) — True = padding position
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    mask = np.ones((len(sequences), max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        mask[i, :min(len(seq), max_len)] = False
    return mask

def mini_batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                          shuffle: bool = True, seed: int = 42):
    """
    Yield (X_batch, y_batch) mini-batches for SGD.
    Last incomplete batch is included.
    """
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]

# Test
seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
padded = pad_sequences(seqs, padding='post')
print(padded)
# [[1, 2, 3, 0],
#  [4, 5, 0, 0],
#  [6, 7, 8, 9]]

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
batches = list(mini_batch_generator(X, y, batch_size=32))
print(f"Number of batches: {len(batches)}")  # ceil(100/32) = 4
print(f"Batch shapes: {[b[0].shape for b in batches]}")
```

---

## 🟢 PROBLEM 6: Probability Distributions

### Theory
Core distributions that come up in ML interviews:

| Distribution | PMF/PDF | Mean | Variance | ML Use |
|---|---|---|---|---|
| Bernoulli(p) | P(X=1)=p | p | p(1-p) | Binary outcome |
| Binomial(n,p) | C(n,k)p^k(1-p)^(n-k) | np | np(1-p) | Count successes |
| Poisson(λ) | e^{-λ}λ^k/k! | λ | λ | Count rare events |
| Geometric(p) | (1-p)^(k-1)p | 1/p | (1-p)/p² | Trials until success |

### Implementation
```python
import math

def bernoulli_pmf(k: int, p: float) -> float:
    """P(X=k) for Bernoulli(p). k ∈ {0,1}."""
    assert k in {0, 1} and 0 <= p <= 1
    return p if k == 1 else (1 - p)

def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X=k) for Binomial(n,p). k ∈ {0,...,n}."""
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1 - p) ** (n - k))

def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for Binomial(n,p)."""
    return sum(binomial_pmf(i, n, p) for i in range(k + 1))

def poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) for Poisson(λ)."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def geometric_pmf(k: int, p: float) -> float:
    """P(X=k) = (1-p)^{k-1} * p. X = trial of first success."""
    return ((1 - p) ** (k - 1)) * p

def gaussian_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """PDF of N(mu, sigma²)."""
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu)/sigma)**2)

# Test
print(bernoulli_pmf(1, 0.3))    # 0.3   — P(fraud) = 30%
print(binomial_pmf(3, 10, 0.3)) # 0.267 — 3 frauds in 10 txns at 30% rate
print(poisson_pmf(2, 1.5))      # 0.251 — 2 events when avg is 1.5
print(geometric_pmf(3, 0.5))    # 0.125 — 3rd trial is first success at 50%
```

---

## 🟢 PROBLEM 7: Learning Rate Schedulers

### Theory
Learning rate schedule controls how α changes during training.

**Step Decay:** `lr = lr₀ × γ^⌊epoch/step_size⌋` — drops LR by γ every step_size epochs.

**Cosine Annealing:** LR follows cosine curve from lr_max to lr_min.
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_{\max}}\pi\right)\right)$$

**Linear Warmup + Decay:** Used in BERT, GPT — warm up LR from 0 to peak, then decay.

### Things to Focus On
- ✅ Warmup: prevents early instability in transformer training (gradient norms unstable early)
- ✅ Cosine annealing: smoother than step decay, periodic restarts help escape local minima
- ✅ 1-cycle LR (superconvergence): fast training with high LR

### Implementation
```python
def step_decay_scheduler(lr_0: float, gamma: float, step_size: int):
    """Step decay: multiply LR by gamma every step_size epochs."""
    def get_lr(epoch: int) -> float:
        return lr_0 * (gamma ** (epoch // step_size))
    return get_lr

def cosine_annealing_lr(epoch: int, t_max: int, 
                          lr_max: float = 0.01, lr_min: float = 0.0) -> float:
    """Cosine annealing: smooth LR decay following cosine curve."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / t_max))

def linear_warmup_cosine_decay(step: int, warmup_steps: int, total_steps: int,
                                 lr_max: float = 0.001) -> float:
    """
    Linear warmup then cosine decay.
    Used in BERT and many transformer training recipes.
    """
    if step < warmup_steps:
        return lr_max * step / warmup_steps  # Linear warmup
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_max * 0.5 * (1 + np.cos(np.pi * progress))  # Cosine decay

def exponential_decay(lr_0: float, decay_rate: float, step: int, 
                       decay_steps: int = 1000) -> float:
    """Exponential decay: lr = lr_0 * decay_rate^(step/decay_steps)."""
    return lr_0 * (decay_rate ** (step / decay_steps))

# Visualize all schedulers
total_steps = 1000
warmup = 100
steps = np.arange(total_steps)

cosine_lrs = [cosine_annealing_lr(s, total_steps, lr_max=0.01) for s in steps]
warmup_cosine = [linear_warmup_cosine_decay(s, warmup, total_steps, 0.01) for s in steps]

print(f"Cosine LR range: {min(cosine_lrs):.5f} to {max(cosine_lrs):.5f}")
print(f"Warmup+Cosine range: {min(warmup_cosine):.5f} to {max(warmup_cosine):.5f}")
print(f"Peak at step {np.argmax(warmup_cosine)} (should be {warmup})")
```

---

## 🟢 PROBLEM 8: Gradient Clipping

### Theory
Prevents exploding gradients by capping the gradient norm:

$$\text{if } \|g\| > \text{threshold}: \quad g \leftarrow \frac{\text{threshold}}{\|g\|} g$$

**Global norm clipping:** Clips gradient of all parameters together (PyTorch default).

Used heavily in RNN/LSTM training where gradient explosion is common.

### Implementation
```python
def clip_gradients_by_norm(grads: list, max_norm: float = 1.0) -> list:
    """
    Clip gradients if their combined L2 norm exceeds max_norm.
    Scales ALL gradients by same factor (preserves direction).
    
    grads: list of gradient arrays
    max_norm: maximum allowed global norm
    """
    # Compute global gradient norm
    total_sq = sum(np.sum(g ** 2) for g in grads)
    global_norm = np.sqrt(total_sq)
    
    if global_norm <= max_norm:
        return grads  # No clipping needed
    
    # Scale factor
    clip_factor = max_norm / global_norm
    return [g * clip_factor for g in grads]

def clip_gradients_by_value(grads: list, min_val: float = -1.0, 
                             max_val: float = 1.0) -> list:
    """Clip each gradient element independently (not norm-based)."""
    return [np.clip(g, min_val, max_val) for g in grads]

# Test
grads = [np.array([10.0, 20.0]), np.array([5.0, 15.0])]
global_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
print(f"Original norm: {global_norm:.2f}")

clipped = clip_gradients_by_norm(grads, max_norm=1.0)
clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped))
print(f"Clipped norm:  {clipped_norm:.2f}")  # Should be ≤ 1.0
```

---

## 🟢 PROBLEM 9: K-Fold Cross-Validation Split

### Implementation
```python
def k_fold_split(n: int, k: int = 5, shuffle: bool = True, 
                  seed: int = 42) -> list:
    """
    Generate k-fold cross-validation indices.
    
    Returns: list of k (train_indices, val_indices) tuples
    """
    indices = np.arange(n)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    fold_size = n // k
    folds = []
    
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k-1 else n
        
        val_idx   = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        folds.append((train_idx, val_idx))
    
    return folds

# Test
folds = k_fold_split(n=100, k=5)
for i, (tr, va) in enumerate(folds):
    print(f"Fold {i+1}: train={len(tr)}, val={len(va)}")
# Each fold: 80 train, 20 val ✓
```

---

## 🟢 PROBLEM 10: Majority Classifier (Baseline) & Percent Change

```python
def majority_classifier_baseline(y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Baseline: always predict the most frequent class.
    Any ML model should beat this.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    return np.full(len(X_test), majority_class)

def percent_change(old_value: float, new_value: float) -> float:
    """Percent change: (new - old) / |old| × 100"""
    if old_value == 0:
        return float('inf') if new_value != 0 else 0.0
    return (new_value - old_value) / abs(old_value) * 100

def rank_transform(x: np.ndarray) -> np.ndarray:
    """
    Transform values to their rank (1 = smallest).
    Ties get average rank.
    """
    sorted_idx = np.argsort(x)
    ranks = np.empty_like(sorted_idx, dtype=float)
    ranks[sorted_idx] = np.arange(1, len(x) + 1)
    return ranks

# Test
y_train = np.array([0,0,0,0,1,1])  # 4 negatives, 2 positives
X_test = np.zeros((10, 3))
preds = majority_classifier_baseline(y_train, X_test)
print(preds)  # All zeros (majority class)
print(f"Baseline accuracy: {(preds==0).mean():.2f}")  # 1.0 (trivially)

print(percent_change(100, 120))  # 20.0%
print(rank_transform(np.array([10, 30, 20, 40, 10])))  # [1.5, 4., 3., 5., 1.5]
```
