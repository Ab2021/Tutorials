# Day 31: Transformers - Interview Questions

> **Topic**: LLM Architecture
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Explain the Self-Attention mechanism.
**Answer:**
*   Allows each token to look at every other token in the sequence.
*   Computes relevance scores ($Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$).
*   Captures long-range dependencies better than RNNs.

### 2. What are Query, Key, and Value vectors?
**Answer:**
*   **Query ($Q$)**: What I am looking for.
*   **Key ($K$)**: What I contain.
*   **Value ($V$)**: The actual content.
*   Analogy: Search engine. Query = Search term. Key = Document title. Value = Document content.

### 3. Why do we divide by $\sqrt{d_k}$ in Attention?
**Answer:**
*   Scaling factor.
*   Without it, dot products grow large with dimension $d_k$.
*   Large values push Softmax into regions with tiny gradients (Vanishing Gradient).

### 4. What is Multi-Head Attention?
**Answer:**
*   Running Self-Attention $h$ times in parallel with different weight matrices.
*   Allows model to focus on different aspects (e.g., Head 1: Syntax, Head 2: Semantics).
*   Outputs are concatenated.

### 5. Explain Positional Encoding. Why is it needed?
**Answer:**
*   Transformer has no recurrence/convolution. It sees a "bag of words".
*   Must inject order information.
*   Adds sine/cosine waves of different frequencies to embeddings.

### 6. What is the difference between Encoder and Decoder?
**Answer:**
*   **Encoder**: Bi-directional. Sees future tokens. Good for Understanding (BERT).
*   **Decoder**: Uni-directional (Causal). Sees only past tokens. Good for Generation (GPT).

### 7. What is Layer Normalization? Why use it instead of Batch Norm?
**Answer:**
*   Normalizes across the feature dimension for a single token.
*   Independent of Batch Size and Sequence Length.
*   Crucial for NLP where sequence lengths vary.

### 8. What is the complexity of Self-Attention?
**Answer:**
*   $O(N^2 \cdot d)$.
*   Quadratic with sequence length $N$.
*   Limiting factor for long contexts.

### 9. What is Feed-Forward Network (FFN) in Transformer?
**Answer:**
*   Applied to each position separately and identically.
*   Two linear layers with ReLU in between.
*   Acts as a "Key-Value Memory" storing facts.

### 10. What is Masked Self-Attention?
**Answer:**
*   Used in Decoder.
*   Prevents positions from attending to subsequent positions.
*   Ensures auto-regressive property (can't cheat by seeing future).

### 11. Explain "Pre-Norm" vs "Post-Norm".
**Answer:**
*   **Post-Norm**: Norm after Residual. Harder to train deep nets.
*   **Pre-Norm**: Norm before Sub-layer. More stable gradients. Standard in modern LLMs (GPT-3, Llama).

### 12. What is a Residual Connection?
**Answer:**
*   $x + Sublayer(x)$.
*   Allows gradients to flow through network without vanishing.

### 13. How does Transformer handle variable length input?
**Answer:**
*   Padding + Masking.
*   Attention mask sets score to $-\infty$ for pad tokens, so Softmax becomes 0.

### 14. What is the "KV Cache" in inference?
**Answer:**
*   In generation, we re-compute Attention for past tokens every step. Wasteful.
*   Cache $K$ and $V$ vectors of past tokens.
*   Only compute $Q, K, V$ for the *new* token.

### 15. Why are Transformers better than RNNs?
**Answer:**
*   **Parallelism**: Can process entire sequence at once (Training).
*   **Long-term dependency**: Path length is $O(1)$, not $O(N)$.

### 16. What is "Flash Attention"?
**Answer:**
*   IO-aware optimization.
*   Tiling to reduce memory reads/writes between HBM (GPU memory) and SRAM (Chip cache).
*   Speeds up attention and reduces memory usage from quadratic to linear (in practice).

### 17. What is Rotary Positional Embedding (RoPE)?
**Answer:**
*   Encodes position by rotating the vector in complex plane.
*   Better relative position generalization than absolute encoding.
*   Used in Llama, PaLM.

### 18. What is ALiBi (Attention with Linear Biases)?
**Answer:**
*   No positional embedding.
*   Adds a penalty to attention score based on distance.
*   Extrapolates to longer sequences than trained on.

### 19. What is the size of a Transformer model?
**Answer:**
*   Parameters $\approx 12 \cdot L \cdot d_{model}^2$.
*   Memory = Parameters + Gradients + Optimizer States + Activations.

### 20. Can Transformers process Images? (ViT).
**Answer:**
*   Yes. Split image into 16x16 patches.
*   Flatten patches into vectors. Treat as tokens.
