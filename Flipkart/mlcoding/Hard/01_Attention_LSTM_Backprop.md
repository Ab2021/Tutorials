# 🔴 Hard: Backpropagation & Advanced Deep Learning
> **Platform:** Deep-ML + TensorTonic | **Difficulty:** Hard | **🔥 DMM + HO Round — Senior Level**

---

## 🔴 PROBLEM 1: Scaled Dot-Product Attention & Multi-Head Attention

### Theory
**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Why scale by √d_k?**
- Q and K have entries ~N(0,1)
- Dot product `q·k` has variance `d_k` (sum of d_k independent unit-variance products)
- Without scaling: large d_k → huge dot products → softmax saturates → vanishing gradients
- Dividing by √d_k brings variance back to 1

**Multi-Head Attention:**
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head attends to different subspaces → model captures different types of relationships simultaneously.

### Things to Focus On
- ✅ Q, K, V projections: each head uses different learned W^Q, W^K, W^V
- ✅ d_k = d_model / n_heads (per head dimension)
- ✅ Computational complexity: O(n² × d) — quadratic in sequence length (bottleneck for long sequences)
- ✅ Flash Attention: IO-efficient implementation, same result in O(n) memory
- ✅ Cross-attention vs self-attention: Q from decoder, K,V from encoder (in encoder-decoder)

### Implementation
```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: np.ndarray = None) -> tuple:
    """
    Scaled Dot-Product Attention.
    
    Q: (..., seq_q, d_k)
    K: (..., seq_k, d_k)
    V: (..., seq_k, d_v)
    mask: (..., seq_q, seq_k) — optional, True means "mask this position"
    
    Returns: (output, attention_weights)
    """
    d_k = Q.shape[-1]
    
    # Similarity scores: (batch, heads, seq_q, seq_k)
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    
    # Apply mask (set masked positions to -inf → softmax → 0)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    
    # Attention weights
    weights = softmax(scores, axis=-1)  # (batch, heads, seq_q, seq_k)
    
    # Weighted values
    output = weights @ V  # (batch, heads, seq_q, d_v)
    
    return output, weights


class MultiHeadAttention:
    """
    Multi-Head Attention from scratch.
    
    Each head attends in a different d_k-dimensional subspace.
    Outputs are concatenated and projected.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Per-head key/query dimension
        self.d_v = d_model // n_heads  # Per-head value dimension
        
        # Projection matrices (random init for demo)
        self.W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_O = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape for multi-head: (batch, seq, d_model) → (batch, heads, seq, d_k)
        """
        batch, seq, d = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape back: (batch, heads, seq, d_k) → (batch, seq, d_model)
        """
        batch, heads, seq, dk = x.shape
        x = x.transpose(0, 2, 1, 3)     # (batch, seq, heads, d_k)
        return x.reshape(batch, seq, -1) # (batch, seq, d_model)
    
    def forward(self, Q_in: np.ndarray, K_in: np.ndarray, V_in: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        """
        Multi-head attention forward pass.
        
        Q_in, K_in, V_in: (batch, seq, d_model)
        Returns: (batch, seq, d_model)
        """
        # Linear projections
        Q = Q_in @ self.W_Q  # (batch, seq, d_model)
        K = K_in @ self.W_K
        V = V_in @ self.W_V
        
        # Split into heads
        Q = self._split_heads(Q)  # (batch, heads, seq, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Attention for all heads simultaneously
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (batch, heads, seq, d_v)
        
        # Combine heads
        concat = self._combine_heads(attn_out)  # (batch, seq, d_model)
        
        # Output projection
        return concat @ self.W_O


# Test
batch, seq_len, d_model, n_heads = 2, 10, 64, 8
x = np.random.randn(batch, seq_len, d_model)

mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
output = mha.forward(x, x, x)  # Self-attention (Q=K=V)
print(f"Input shape:  {x.shape}")      # (2, 10, 64)
print(f"Output shape: {output.shape}") # (2, 10, 64) ✓
```

---

## 🔴 PROBLEM 2: LSTM Cell Forward Pass

### Theory
LSTM solves the vanishing gradient problem of RNNs via a cell state (Ct) that carries information through time with minimal interference.

**Four gates:**
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(Candidate cell state)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}$$

**Cell and hidden state updates:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$h_t = o_t \odot \tanh(C_t)$$

### Things to Focus On
- ✅ Forget gate: decides what to erase from cell state (f_t ≈ 1 → keep, ≈ 0 → forget)
- ✅ Input gate: decides what new information to add
- ✅ Cell state C_t is the "memory highway" — gets additive updates only (no multiplicative path)
- ✅ h_t is the "short-term memory"; C_t is the "long-term memory"
- ✅ GRU is a simplified LSTM (2 gates, no cell state) — 1/3 fewer parameters

### Implementation
```python
class LSTMCell:
    """
    LSTM Cell forward pass.
    Processes one time step at a time.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        I = input_size
        
        # All 4 gates share weight matrices for [h_{t-1}, x_t] input
        # Combined dim: (I + H) → 4H (one set of weights for all gates)
        scale = np.sqrt(2 / (I + H))
        self.W = np.random.randn(I + H, 4 * H) * scale  # Input → 4 gates
        self.b = np.zeros(4 * H)
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, 
                C_prev: np.ndarray) -> tuple:
        """
        One LSTM step.
        
        x_t:   (batch, input_size)
        h_prev: (batch, hidden_size)
        C_prev: (batch, hidden_size)
        
        Returns: (h_t, C_t) — next hidden and cell state
        """
        # Concatenate [h_{t-1}, x_t] → (batch, input_size + hidden_size)
        combined = np.concatenate([h_prev, x_t], axis=-1)
        
        # All 4 gates in one matrix multiply
        gates = combined @ self.W + self.b  # (batch, 4 * hidden_size)
        
        H = self.hidden_size
        
        # Split into individual gates
        f_raw  = gates[:, 0*H : 1*H]  # Forget gate
        i_raw  = gates[:, 1*H : 2*H]  # Input gate
        C_raw  = gates[:, 2*H : 3*H]  # Candidate cell state
        o_raw  = gates[:, 3*H : 4*H]  # Output gate
        
        # Apply activations
        f_t = self._sigmoid(f_raw)   # (0,1): how much to forget
        i_t = self._sigmoid(i_raw)   # (0,1): how much to input
        g_t = np.tanh(C_raw)         # (-1,1): candidate values
        o_t = self._sigmoid(o_raw)   # (0,1): how much to output
        
        # Cell state: forget old + add new
        C_t = f_t * C_prev + i_t * g_t  # Element-wise operations
        
        # Hidden state: filter through output gate
        h_t = o_t * np.tanh(C_t)
        
        # Cache for backpropagation
        self._cache = (x_t, h_prev, C_prev, f_t, i_t, g_t, o_t, C_t, h_t)
        
        return h_t, C_t
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def run_sequence(self, X: np.ndarray) -> tuple:
        """
        Run LSTM over full sequence.
        
        X: (batch, seq_len, input_size)
        Returns: (all_h, final_C)
        """
        batch_size, seq_len, _ = X.shape
        H = self.hidden_size
        
        h = np.zeros((batch_size, H))
        C = np.zeros((batch_size, H))
        all_h = []
        
        for t in range(seq_len):
            h, C = self.forward(X[:, t, :], h, C)
            all_h.append(h)
        
        # Stack: (seq_len, batch, hidden) → (batch, seq_len, hidden)
        all_h = np.stack(all_h, axis=1)
        return all_h, C


class GRUCell:
    """
    GRU Cell: simplified LSTM with 2 gates and no separate cell state.
    
    Update gate: z_t (like 1-f in LSTM)
    Reset gate:  r_t (controls how much past to expose when building candidate)
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        I = input_size
        
        scale = np.sqrt(2 / (I + H))
        # Two gates (update, reset): (I+H) → 2H
        self.W_zr = np.random.randn(I + H, 2 * H) * scale
        self.b_zr = np.zeros(2 * H)
        
        # Candidate hidden: (I+H) → H
        self.W_h = np.random.randn(I + H, H) * scale
        self.b_h = np.zeros(H)
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """One GRU step."""
        combined = np.concatenate([h_prev, x_t], axis=-1)
        H = self.hidden_size
        
        # Gates
        gates = self._sigmoid(combined @ self.W_zr + self.b_zr)
        z_t = gates[:, :H]   # Update gate
        r_t = gates[:, H:]   # Reset gate
        
        # Candidate hidden state (reset previous state first)
        candidate_input = np.concatenate([r_t * h_prev, x_t], axis=-1)
        h_tilde = np.tanh(candidate_input @ self.W_h + self.b_h)
        
        # Mix old and new hidden states
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# Test both
batch_size, seq_len, input_size, hidden_size = 4, 10, 8, 16

lstm = LSTMCell(input_size, hidden_size)
gru  = GRUCell (input_size, hidden_size)
X = np.random.randn(batch_size, seq_len, input_size)

all_h, final_C = lstm.run_sequence(X)
print(f"LSTM output: {all_h.shape}")   # (4, 10, 16)
print(f"LSTM cell:   {final_C.shape}") # (4, 16)

h = np.zeros((batch_size, hidden_size))
for t in range(seq_len):
    h = gru.forward(X[:, t, :], h)
print(f"GRU final h: {h.shape}")  # (4, 16)
```

---

## 🔴 PROBLEM 3: Backpropagation (Single Hidden Layer MLP)

### Theory
Backpropagation = chain rule applied recursively to compute gradients of loss w.r.t. all parameters.

**Forward pass** (2-layer MLP):
$$z_1 = XW_1 + b_1, \quad a_1 = \text{ReLU}(z_1)$$
$$z_2 = a_1 W_2 + b_2, \quad \hat{y} = \text{softmax}(z_2)$$

**Loss:** Cross-entropy $\mathcal{L}$

**Backward pass:**
$$\frac{\partial \mathcal{L}}{\partial z_2} = \hat{y} - y \quad \text{(softmax + CE gradient is clean)}$$
$$\frac{\partial \mathcal{L}}{\partial W_2} = a_1^T \frac{\partial \mathcal{L}}{\partial z_2}$$
$$\frac{\partial \mathcal{L}}{\partial a_1} = \frac{\partial \mathcal{L}}{\partial z_2} W_2^T$$
$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial a_1} \odot \mathbb{1}[z_1 > 0] \quad \text{(ReLU gradient)}$$
$$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \frac{\partial \mathcal{L}}{\partial z_1}$$

### Things to Focus On
- ✅ The "softmax + CE" gradient δ₂ = ŷ - y is the KEY result to know cold
- ✅ Shape checking: dimensions must align (use matrix shapes as guide)
- ✅ Gradient flows backward: output layer → hidden layer → input
- ✅ ReLU derivative is just the mask (which neurons were active in forward pass)
- ✅ With L2 reg: add λW to weight gradients

### Implementation
```python
class TwoLayerMLP:
    """
    Two-layer MLP with manual backpropagation.
    Architecture: Input → ReLU → Output (Softmax)
    Loss: Cross-Entropy
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.01, lambda_reg: float = 0.0):
        scale1 = np.sqrt(2 / input_size)    # He init for ReLU
        scale2 = np.sqrt(2 / hidden_size)
        
        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros(output_size)
        
        self.lr = lr
        self.lambda_reg = lambda_reg
        self._cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through 2-layer MLP.
        X: (batch, input_size)
        Returns: probabilities (batch, output_size)
        """
        # Layer 1: linear + ReLU
        z1 = X @ self.W1 + self.b1       # (batch, hidden_size)
        a1 = np.maximum(0, z1)            # ReLU activation
        
        # Layer 2: linear
        z2 = a1 @ self.W2 + self.b2       # (batch, output_size)
        
        # Softmax output
        z2_stable = z2 - z2.max(axis=-1, keepdims=True)
        exp_z2 = np.exp(z2_stable)
        y_hat = exp_z2 / exp_z2.sum(axis=-1, keepdims=True)  # (batch, output_size)
        
        # Cache for backprop
        self._cache = (X, z1, a1, z2, y_hat)
        return y_hat
    
    def cross_entropy_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-entropy loss.
        y: one-hot encoded labels (batch, output_size)
        """
        eps = 1e-8
        n = len(y)
        # L2 regularization penalty
        l2 = self.lambda_reg * 0.5 * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return -np.sum(y * np.log(y_hat + eps)) / n + l2
    
    def backward(self, y: np.ndarray) -> dict:
        """
        Backpropagation through the network.
        y: one-hot labels (batch, output_size)
        Returns: gradients dict
        """
        X, z1, a1, z2, y_hat = self._cache
        n = len(X)
        
        # ===== OUTPUT LAYER GRADIENTS =====
        # δ₂ = ŷ - y  (combined softmax+CE gradient)
        delta2 = (y_hat - y) / n               # (batch, output_size)
        
        # Gradient w.r.t. W2 and b2
        dW2 = a1.T @ delta2                    # (hidden, output)
        db2 = delta2.sum(axis=0)               # (output,)
        
        # Add L2 regularization to weight gradients (NOT bias)
        dW2 += self.lambda_reg * self.W2
        
        # ===== HIDDEN LAYER GRADIENTS =====
        # Gradient w.r.t. a1 (before ReLU)
        da1 = delta2 @ self.W2.T               # (batch, hidden)
        
        # ReLU gradient: pass through only where z1 > 0
        delta1 = da1 * (z1 > 0).astype(float)  # (batch, hidden) — element-wise mask
        
        # Gradient w.r.t. W1 and b1
        dW1 = X.T @ delta1                     # (input, hidden)
        db1 = delta1.sum(axis=0)               # (hidden,)
        
        dW1 += self.lambda_reg * self.W1
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update(self, grads: dict):
        """Gradient descent weight update."""
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']
    
    def fit(self, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 500):
        """Train the network."""
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.cross_entropy_loss(y_hat, y_onehot)
            grads = self.backward(y_onehot)
            self.update(grads)
            if epoch % 100 == 0:
                preds = self.predict(X)
                acc = (preds == y_onehot.argmax(axis=1)).mean()
                print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)


# Test on XOR problem
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[1,0],[0,1],[0,1],[1,0]])  # One-hot

mlp = TwoLayerMLP(input_size=2, hidden_size=8, output_size=2, lr=0.1)
mlp.fit(X_xor, y_xor, epochs=1000)
print(f"XOR predictions: {mlp.predict(X_xor)}")  # Should be [0, 1, 1, 0]
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"What makes LSTM better than vanilla RNN?"** → Cell state C_t provides an additive gradient highway, avoiding the multiplicative gradient vanishing of H chain in RNN. BPTT gradient through C_t is approximately constant.
2. **"Why do we need query, key, value projections in attention?"** → Without projections, Q, K, V would all be identical input representations. Projections allow the model to learn different roles for the same input (what to query, what key to expose, what value to output).
3. **"What does the 'forget gate' forget?"** → It gates the previous cell state C_{t-1}. f_t ≈ 0 → completely forget past; f_t ≈ 1 → carry forward everything. A series with regime changes (e.g., resetting at sentence boundaries) needs the forget gate to work well.
4. **"Why use He initialization instead of random?"** → For ReLU networks, each layer zeroes ~50% of neurons. Without special init, variance contracts through depth → vanishing activations. He init: W ~ N(0, 2/fan_in) preserves variance after ReLU.
