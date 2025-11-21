# Day 20: RNNs - Interview Questions

> **Topic**: Sequence Modeling
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Why use RNNs for sequence data?
**Answer:**
*   Handles variable length input.
*   Maintains internal state (memory) of history.
*   Share parameters across time steps.

### 2. Explain the Vanishing Gradient problem in RNNs.
**Answer:**
*   Gradients flow back through time (BPTT).
*   Repeated multiplication by weight matrix $W_h$.
*   If eigenvalues < 1, gradient vanishes. Long-term dependencies are lost.

### 3. What is LSTM (Long Short-Term Memory)? How does it solve vanishing gradients?
**Answer:**
*   Introduces **Cell State** (Highway for gradient).
*   Uses **Gates** (Input, Forget, Output) to control information flow.
*   Additive updates to cell state prevent gradient decay.

### 4. Explain the Forget Gate in LSTM.
**Answer:**
*   Decides what to throw away from cell state.
*   $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$.

### 5. What is GRU (Gated Recurrent Unit)? Difference from LSTM?
**Answer:**
*   Simplified LSTM.
*   2 Gates: **Update** and **Reset**.
*   No separate Cell State.
*   Faster to train, similar performance.

### 6. What is Bidirectional RNN?
**Answer:**
*   Two RNNs: One reads Forward, one Backward.
*   Concatenate states.
*   Captures context from both past and future.

### 7. What is Sequence-to-Sequence (Seq2Seq)?
**Answer:**
*   Encoder-Decoder architecture.
*   Encoder compresses input to context vector.
*   Decoder generates output sequence.
*   Used for Translation.

### 8. Explain the Attention Mechanism.
**Answer:**
*   Solves bottleneck of fixed context vector.
*   Decoder looks at **all** encoder states at every step.
*   Calculates weights (Attention scores) based on relevance.

### 9. What is Teacher Forcing?
**Answer:**
*   During training, feed the **Ground Truth** token as input to next step, instead of model's own prediction.
*   Stabilizes training.

### 10. What is Beam Search?
**Answer:**
*   Inference algorithm.
*   Instead of greedy decoding (pick max prob), keep top K sequences (Beams).
*   Expands K paths at each step.

### 11. What is BPTT (Backpropagation Through Time)?
**Answer:**
*   Unroll RNN for T steps.
*   Standard Backprop on the unrolled graph.

### 12. What is Gradient Clipping? Why is it needed in RNNs?
**Answer:**
*   Clips gradient norm.
*   Prevents **Exploding Gradients** (Eigenvalues > 1).

### 13. Compare RNNs vs Transformers.
**Answer:**
*   **RNN**: Sequential (Slow). $O(N)$ path length.
*   **Transformer**: Parallel (Fast). $O(1)$ path length. Attention mechanism.

### 14. What is Word Embedding (Word2Vec)?
**Answer:**
*   Dense vector representation of words.
*   Captures semantic meaning ($King - Man + Woman = Queen$).

### 15. Explain CBOW vs Skip-gram.
**Answer:**
*   **CBOW**: Predict center word from context.
*   **Skip-gram**: Predict context words from center word. (Better for rare words).

### 16. What is 1D Convolution for text?
**Answer:**
*   Slide filter over sequence of embeddings.
*   Detects N-gram patterns. Fast alternative to RNNs.

### 17. What is a Language Model?
**Answer:**
*   Predicts probability of next word given history. $P(w_t | w_{1:t-1})$.

### 18. How do you handle variable length sequences in batches?
**Answer:**
*   **Padding**: Pad with zeros to max length.
*   **Masking**: Tell loss function to ignore padded positions.

### 19. What is Perplexity?
**Answer:**
*   Metric for Language Models.
*   $2^{Entropy}$. Lower is better.

### 20. What is the "Bottleneck" in vanilla Seq2Seq?
**Answer:**
*   The fixed-size context vector must hold all information of the source sentence.
*   Fails for long sentences. Solved by Attention.
