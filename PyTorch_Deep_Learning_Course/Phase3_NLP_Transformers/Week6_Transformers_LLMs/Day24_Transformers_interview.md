# Day 24: Transformers - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Architecture, Attention, and Optimization

### 1. Why is Self-Attention $O(N^2)$?
**Answer:**
*   We compute the similarity (dot product) between every pair of tokens.
*   Matrix size is $N \times N$.
*   This limits standard Transformers to sequences of ~4096 tokens.

### 2. What is the role of the "Scale" factor $\sqrt{d_k}$?
**Answer:**
*   Without scaling, for large $d_k$, the dot products can be very large in magnitude.
*   This pushes the Softmax function into regions where gradients are extremely small (saturation).
*   Scaling keeps the variance of the dot products to 1.

### 3. Why Multi-Head Attention instead of Single-Head?
**Answer:**
*   Allows the model to attend to information from different representation subspaces at different positions.
*   Example: One head focuses on "Who did it", another on "When it happened".
*   Averaging a single head would lose this distinct information.

### 4. Explain "Masked Self-Attention".
**Answer:**
*   Used in the Decoder (GPT).
*   Ensures the prediction for position $i$ depends only on positions $0$ to $i-1$.
*   We mask future positions ($i+1$ to $N$) by setting their attention scores to $-\infty$.

### 5. What is the difference between Encoder and Decoder in Transformer?
**Answer:**
*   **Encoder**: Bidirectional Self-Attention. Sees full context. (BERT).
*   **Decoder**: Masked (Causal) Self-Attention. Sees only past. Has Cross-Attention to Encoder. (GPT).

### 6. Why LayerNorm and not BatchNorm?
**Answer:**
*   In NLP, batch statistics are noisy (variable lengths) and irrelevant.
*   LayerNorm normalizes each token's features independently, which makes more sense for sequences.

### 7. What is "Pre-Norm" vs "Post-Norm"?
**Answer:**
*   **Post-Norm**: Norm after residual. Unstable gradients at init.
*   **Pre-Norm**: Norm before sublayer. Stable gradients. Standard in modern LLMs.

### 8. What are Query, Key, and Value?
**Answer:**
*   Abstractions from Information Retrieval.
*   **Query**: The vector representing the current token looking for context.
*   **Key**: The vector representing other tokens advertising their content.
*   **Value**: The content vector to be aggregated if Query matches Key.

### 9. How does Transformer handle variable length sequences?
**Answer:**
*   **Padding**: Pad to max length.
*   **Masking**: Use padding masks in Attention so the model ignores pad tokens.

### 10. What is the purpose of the Feed-Forward Network?
**Answer:**
*   Attention mixes information *between* tokens (Spatial mixing).
*   FFN processes information *within* each token (Channel mixing).
*   Adds non-linearity and capacity.

### 11. Can Transformers extrapolate to longer sequences?
**Answer:**
*   Poorly with learned embeddings.
*   Better with Sinusoidal or RoPE (Rotary Positional Embeddings).
*   Still a major research challenge (ALiBi, etc.).

### 12. What is "Cross-Attention"?
**Answer:**
*   Attention where Queries come from one sequence (Target) and Keys/Values come from another (Source).
*   Connects Encoder and Decoder.

### 13. Why do we add Positional Encodings?
**Answer:**
*   Self-Attention is permutation invariant. $Attention(A, B) = Attention(B, A)$.
*   Without PE, "Dog bites Man" and "Man bites Dog" look identical to the model.

### 14. What is the "Residual Connection" for?
**Answer:**
*   Solves Vanishing Gradient.
*   Allows gradients to flow through the network directly.
*   Makes the loss landscape smoother.

### 15. How many parameters in a Transformer Layer?
**Answer:**
*   Attention: $4 \times d^2$ (Q, K, V, O projections).
*   FFN: $8 \times d^2$ (assuming expansion 4d).
*   Total $\approx 12 d^2$.

### 16. What is "Label Smoothing"?
**Answer:**
*   Regularization technique used in the original paper.
*   Instead of one-hot target [0, 1, 0], use [0.1, 0.8, 0.1].
*   Prevents the model from becoming over-confident.

### 17. What is "Warmup" steps?
**Answer:**
*   Linearly increasing Learning Rate from 0 to max over first few thousand steps.
*   Crucial for Transformers (especially Post-Norm) to stabilize early training variance.

### 18. Why is the FFN expansion ratio usually 4?
**Answer:**
*   Empirical choice.
*   Provides enough capacity for the FFN to act as a Key-Value memory.
*   SwiGLU variants use different ratios (e.g., 8/3).

### 19. What is "GELU" activation?
**Answer:**
*   Gaussian Error Linear Unit. $x \Phi(x)$.
*   Smoother than ReLU. Used in BERT and GPT.
*   Probabilistic interpretation (Dropout-like).

### 20. Are Transformers universal approximators?
**Answer:**
*   Yes, they are Turing Complete (given enough depth/precision).
*   They can approximate any sequence-to-sequence function.
