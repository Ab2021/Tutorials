# Day 31: Transformer Architecture Deep Dive

> **Phase**: 4 - LLMs & GenAI
> **Week**: 7 - LLM Foundations
> **Focus**: Attention is All You Need
> **Reading Time**: 60 mins

---

## 1. The Attention Mechanism

Before 2017, RNNs processed text sequentially. Transformers process it in parallel.

### 1.1 Self-Attention
"How much does word A relate to word B?"
*   **Query (Q)**: What am I looking for?
*   **Key (K)**: What do I have?
*   **Value (V)**: What is the content?
*   **Formula**: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    *   $QK^T$: Similarity matrix (Dot product).
    *   $\sqrt{d_k}$: Scaling factor to prevent vanishing gradients.
    *   Softmax: Normalize to probabilities.

### 1.2 Multi-Head Attention
*   Instead of one attention head, use $H$ heads.
*   **Why?**: Each head learns different relationships (e.g., Head 1 learns grammar, Head 2 learns names, Head 3 learns dates).

---

## 2. Architecture Variants

### 2.1 Encoder-Decoder (T5, BART)
*   **Encoder**: Bi-directional. Sees the whole sentence. Good for translation/summarization.
*   **Decoder**: Auto-regressive. Generates one token at a time.

### 2.2 Decoder-Only (GPT, Llama)
*   **Structure**: Just the Decoder stack.
*   **Causal Masking**: Token $t$ can only attend to tokens $1 \dots t-1$. It cannot see the future.
*   **Dominance**: This architecture won the scaling war.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Context Window Limit
**Scenario**: $O(N^2)$ complexity. Doubling context length (4k -> 8k) quadruples memory/compute.
**Solution**:
*   **Flash Attention**: IO-aware exact attention. Reduces memory reads/writes. Speedup 2-4x.
*   **Sparse Attention**: Only attend to local neighbors + random distant tokens.

### Challenge 2: Positional Information
**Scenario**: Self-attention is permutation invariant. "Dog bites Man" = "Man bites Dog".
**Solution**:
*   **Sinusoidal Embeddings**: Add sine/cosine waves to input embeddings.
*   **RoPE (Rotary Positional Embeddings)**: The modern standard (Llama). Rotates the Q/K vectors. Allows better extrapolation to longer lengths.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the time complexity of Self-Attention?**
> **Answer**: $O(N^2 \cdot d)$.
> *   $N$: Sequence length.
> *   $d$: Embedding dimension.
> *   This quadratic cost is why early Transformers were limited to 512 tokens.

**Q2: Why do we need the scaling factor $\sqrt{d_k}$?**
> **Answer**: Without it, the dot products $QK^T$ can become very large. This pushes the Softmax function into regions where gradients are extremely small (saturation), leading to the vanishing gradient problem.

**Q3: Explain "Causal Masking".**
> **Answer**: In a Decoder-only model (GPT), when predicting the next word, the model must not "cheat" by seeing the future words. We apply a mask (upper triangular matrix of $-\infty$) to the attention scores so that position $i$ cannot attend to position $j$ if $j > i$.

---

## 5. Further Reading
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
