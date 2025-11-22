# Day 12: Positional Encodings & Embeddings
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. The Mathematics of Rotary Positional Embeddings (RoPE)

**Goal:** Find a transformation $f(x, pos)$ such that the dot product between two vectors depends only on their relative distance $m - n$.
$$ \langle f(q, m), f(k, n) \rangle = g(q, k, m - n) $$

**Derivation (2D Case):**
In 2D, we can represent a vector $x$ as a complex number.
Let $q = q_0 + iq_1$ and $k = k_0 + ik_1$.
We define the positional encoding as a rotation by angle $m\theta$:
$$ f(q, m) = q \cdot e^{im\theta} $$
$$ f(k, n) = k \cdot e^{in\theta} $$

The dot product is the real part of the product of one complex number with the conjugate of the other:
$$ \langle f(q, m), f(k, n) \rangle = \text{Re}(f(q, m) \cdot \overline{f(k, n)}) $$
$$ = \text{Re}((q \cdot e^{im\theta}) \cdot (\overline{k \cdot e^{in\theta}})) $$
$$ = \text{Re}(q \cdot e^{im\theta} \cdot \overline{k} \cdot e^{-in\theta}) $$
$$ = \text{Re}(q \cdot \overline{k} \cdot e^{i(m-n)\theta}) $$

Notice the term $e^{i(m-n)\theta}$. It depends **only** on the relative position $m-n$.

**Generalizing to d-dimensions:**
We split the d-dimensional vector into $d/2$ pairs. We rotate each pair with a different frequency $\theta_i$.
$$ \theta_i = 10000^{-2i/d} $$

**Matrix Form:**
For a pair of features $(x_1, x_2)$ at position $m$:
$$
\begin{pmatrix} x'_1 \\ x'_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

**Implementation Efficiency:**
Instead of full matrix multiplication (which is sparse), we implement it element-wise:
```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**Why RoPE Wins:**
1.  **Decay with distance:** The dot product naturally decays as relative distance increases (for high frequency components), matching linguistic intuition (closer words matter more).
2.  **Training Stability:** No learnable parameters means gradients flow directly to semantic embeddings.
3.  **Extrapolation:** Since it's based on rotation, it handles unseen positions better than learned embeddings (though "NTK-Aware scaling" is often needed for massive extension).

### 2. ALiBi (Attention with Linear Biases)

**The Hypothesis:**
Positional embeddings are unnecessary if we bias the attention mechanism to prefer local context.

**Mechanism:**
Subtract a penalty from the attention score based on distance.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V $$
Where $B$ is a matrix:
$$ B_{ij} = \begin{cases} -m \cdot (i - j) & \text{if } i \ge j \\ -\infty & \text{if } i < j \text{ (causal)} \end{cases} $$

**The Slope $m$:**
For 8 heads, slopes are geometric sequence: $\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$.
Head 1 has weak penalty (long range), Head 8 has strong penalty (very local).

**Why ALiBi Extrapolates:**
Because the penalty is linear, the model learns "relative distance" as a fundamental concept. When it sees a distance of 2000 (never seen in training), it just applies a larger penalty. The attention mechanism remains well-behaved.

**Comparison:**
- **Sinusoidal:** Fails to extrapolate because the "wiggles" of high frequencies become unpredictable at unseen positions.
- **RoPE:** Extrapolates well, but can suffer from "out of distribution" rotation angles without scaling.
- **ALiBi:** Extrapolates "out of the box" to 2x-4x training length.

### 3. Long Context & Extrapolation Techniques

**The Problem:**
Training on 32k tokens is expensive ($O(N^2)$). We want to train on 4k and run on 32k.

**RoPE Scaling (NTK-Aware):**
If we simply extend the sequence, the rotation angles $m\theta$ become larger than seen during training.
**Solution:** Interpolate the position indices.
Instead of positions $0, 1, \dots, L$, we map them to $0, \frac{1}{s}, \frac{2}{s}, \dots, \frac{L}{s}$ where $s$ is the scale factor.
This squeezes the longer sequence into the trained range of rotation angles.
*High-frequency components are interpolated (preserving resolution), low-frequency are extrapolated.*

**YaRN (Yet another RoPE extensioN):**
Refines NTK-aware scaling by handling high and low frequencies differently to prevent "resolution loss" in attention.

### 4. Positional Embeddings in Vision Transformers (ViT)

**2D Positional Embeddings:**
Images have 2D structure.
- **Option 1:** Flatten patches and use 1D learnable embeddings (Standard ViT). Surprisingly, the model learns the 2D grid structure on its own!
- **Option 2:** 2D Sinusoidal. $PE(x, y) = [PE_{sin}(x) || PE_{sin}(y)]$.

**Interpolation:**
When fine-tuning ViT on higher resolution images, the sequence length increases (more patches).
We **bicubic interpolate** the pre-trained positional embeddings to the new size. This works because the relative positions are preserved.

### 5. No Positional Encoding?

Some research suggests that for causal decoder-only models, explicit PE might be redundant because the causal mask leaks some positional information (first token attends to 1, second to 2, etc.). However, convergence is much slower and performance usually degrades.

### Summary of Mechanics

| Feature | Sinusoidal | Learned | RoPE | ALiBi |
| :--- | :--- | :--- | :--- | :--- |
| **Injection** | Add to Input | Add to Input | Multiply Q,K | Add to Attn Score |
| **Parameters** | 0 | $L \times d$ | 0 | 0 |
| **Operation** | Addition | Addition | Rotation | Subtraction |
| **Decay** | Implicit | Learned | Implicit | Explicit |
| **Best For** | Generic | Short/Fixed | Modern LLMs | Long Context |

### Code: RoPE Implementation (PyTorch)

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, n_heads, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_tables(seq_len, device=x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )
```
