# Day 23: Seq2Seq & Attention - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Attention, Decoding, and Metrics

### 1. Why was the Attention Mechanism introduced?
**Answer:**
*   To solve the **Fixed-Length Context Vector Bottleneck** in standard Encoder-Decoder RNNs.
*   Standard Seq2Seq forgets early information in long sentences.
*   Attention allows the decoder to access the entire history of encoder states directly.

### 2. Explain the difference between Bahdanau and Luong Attention.
**Answer:**
*   **Bahdanau (Additive)**: Computes score using a feed-forward network: $v^T \tanh(W[s; h])$. Concatenates context with hidden state *before* prediction.
*   **Luong (Multiplicative)**: Computes score using dot product: $s^T W h$. Concatenates context with hidden state *after* prediction.

### 3. What is "Beam Search"?
**Answer:**
*   A heuristic search algorithm that explores a graph by expanding the most promising node in a limited set.
*   Keeps $k$ (beam width) best sequences at each time step.
*   Trade-off between Greedy (k=1) and Exhaustive Search.

### 4. What is "BLEU Score"?
**Answer:**
*   Precision-based metric for translation.
*   Measures n-gram overlap between candidate and reference.
*   Includes a Brevity Penalty.
*   Does not consider meaning or synonyms (only exact matches).

### 5. What is "Exposure Bias"?
**Answer:**
*   The discrepancy between Training (Teacher Forcing: seeing ground truth) and Inference (Autoregressive: seeing own predictions).
*   The model never learns to recover from its own errors during training.

### 6. Why do we mask the attention scores?
**Answer:**
*   To prevent the model from attending to **Padding Tokens**.
*   We set the score of padding tokens to $-\infty$ before Softmax, so their weight becomes 0.

### 7. What is "Hard Attention" vs "Soft Attention"?
**Answer:**
*   **Soft**: Deterministic. Weighted average of all states. Differentiable.
*   **Hard**: Stochastic. Samples one state to attend to. Non-differentiable (requires Reinforcement Learning/REINFORCE).

### 8. Can Attention be used without RNNs?
**Answer:**
*   Yes. That is the basis of the **Transformer**.
*   Self-Attention computes relationships between all tokens in a set simultaneously.

### 9. What is the computational complexity of Attention?
**Answer:**
*   $O(T_x \times T_y)$.
*   For every output word, we calculate scores for every input word.
*   Quadratic cost if sequence length is large.

### 10. What is "Copy Mechanism" (Pointer Generator)?
**Answer:**
*   A hybrid approach.
*   The model calculates a probability $p_{gen}$.
*   With prob $p_{gen}$, generate word from vocab.
*   With prob $1-p_{gen}$, copy word from input (using attention distribution).

### 11. Why is BLEU not a perfect metric?
**Answer:**
*   It relies on exact n-gram matching.
*   "The cat is on the mat" vs "There is a cat on the mat". Low n-gram overlap, but same meaning.
*   METEOR or BERTScore are better semantic metrics.

### 12. What is "Context Vector" in Attention?
**Answer:**
*   The weighted sum of encoder hidden states.
*   $c_t = \sum \alpha_{ti} h_i$.
*   Contains the relevant information for the current decoding step.

### 13. How does Attention provide interpretability?
**Answer:**
*   By plotting the Attention Weights ($\alpha$) as a heatmap (Alignment Matrix).
*   We can see which source words were responsible for generating a specific target word.

### 14. What is "Scheduled Sampling"?
**Answer:**
*   A curriculum learning strategy to fix Exposure Bias.
*   Flip a coin. Heads: Use Ground Truth. Tails: Use Model Prediction.
*   Increase probability of Tails over time.

### 15. What is "Length Normalization" in Beam Search?
**Answer:**
*   Beam search favors short sentences (probability is product of numbers < 1, so more terms = lower prob).
*   We divide the log-probability by the sequence length $L^\alpha$ to normalize.

### 16. What is "Coverage Mechanism"?
**Answer:**
*   Used to prevent repetition in Summarization.
*   Maintains a coverage vector (sum of past attention weights).
*   Penalizes attending to the same location multiple times.

### 17. Why use "Bidirectional Encoder" in Seq2Seq?
**Answer:**
*   To ensure the annotation $h_i$ for word $x_i$ contains context from both left and right.
*   Crucial for translation (adjective-noun order varies by language).

### 18. What is "Self-Attention"?
**Answer:**
*   Attention where Query, Key, and Value come from the *same* sequence.
*   Relates different positions of a single sequence to compute a representation of the sequence.

### 19. What is the dimension of the Attention Matrix?
**Answer:**
*   (Target Sequence Length, Source Sequence Length).

### 20. How does "Dot-Product Attention" differ from "Scaled Dot-Product"?
**Answer:**
*   Scaled divides by $\sqrt{d_k}$.
*   Prevents dot products from growing too large, which pushes Softmax into regions with small gradients.
