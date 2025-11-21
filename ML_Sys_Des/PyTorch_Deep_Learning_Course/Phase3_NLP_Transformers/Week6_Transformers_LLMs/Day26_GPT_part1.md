# Day 26: GPT - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: KV Cache, RoPE, and Sampling Strategies

## 1. KV Cache (Key-Value Cache)

In generation, we predict token $t+1$ using $t$.
Naive approach recomputes attention for $0...t$ every time. $O(N^3)$ total cost.
**KV Cache**:
*   Save the $K$ and $V$ matrices of past tokens.
*   At step $t$, only compute $Q_t, K_t, V_t$.
*   Append $K_t, V_t$ to cache.
*   Attend to $[K_{past}, K_t]$.
*   Reduces complexity to $O(N^2)$ total.

## 2. Rotary Positional Embeddings (RoPE)

Used in LLaMA/PaLM.
Instead of adding $P$ to $X$, we **rotate** $X$ in the complex plane.
$$ f(x, m) = x e^{im\theta} $$
*   **Relative Position**: The dot product $f(x, m) \cdot f(y, n)$ depends only on $m-n$.
*   Allows better extrapolation to longer sequences than Sinusoidal.

## 3. Sampling Strategies

How to pick the next token from `probs`?
1.  **Greedy**: Pick max prob. Boring, repetitive.
2.  **Temperature**: $p_i = \exp(z_i/T) / \sum ...$.
    *   $T < 1$: Sharpen (More confident).
    *   $T > 1$: Flatten (More random).
3.  **Top-K**: Sample from top $K$ tokens only.
4.  **Top-P (Nucleus)**: Sample from smallest set of tokens whose cumulative prob > $P$ (e.g., 0.9).
    *   Dynamic vocabulary size based on confidence. Best balance.

## 4. SwiGLU Activation

Used in LLaMA/PaLM.
$$ \text{SwiGLU}(x) = \text{Swish}(xW) \otimes (xV) $$
*   Gated Linear Unit with Swish activation.
*   Empirically better than ReLU/GELU.

## 5. Flash Attention

IO-Aware Attention.
*   Standard Attention reads/writes $N \times N$ matrix to HBM (High Bandwidth Memory). Slow.
*   **Flash Attention**: Tiling optimization. Computes attention in SRAM (fast cache) without materializing the full matrix.
*   Speedup: 2-4x. Memory: Linear $O(N)$.
