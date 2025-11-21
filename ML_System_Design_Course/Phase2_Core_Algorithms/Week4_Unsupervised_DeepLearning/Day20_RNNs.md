# Day 20: RNNs, LSTMs & Sequence Modeling

> **Phase**: 2 - Core Algorithms
> **Week**: 4 - Unsupervised & Deep Learning
> **Focus**: Time & Sequences
> **Reading Time**: 50 mins

---

## 1. Recurrent Neural Networks (RNN)

Feedforward networks assume inputs are independent. RNNs have "memory".

### 1.1 The Loop
$$h_t = \tanh(W_h h_{t-1} + W_x x_t)$$
*   The hidden state $h_t$ depends on the current input $x_t$ and the previous state $h_{t-1}$.
*   **BPTT (Backpropagation Through Time)**: Unrolling the loop to train.

### 1.2 The Problem
*   **Vanishing Gradient**: Gradients must flow back through time. If $W_h < 1$, gradient vanishes exponentially. RNNs forget long-term dependencies.

---

## 2. LSTM & GRU

### 2.1 LSTM (Long Short-Term Memory)
Introduced the **Cell State** (Information Superhighway).
*   **Gates**: Sigmoid layers that decide what to keep/forget.
    *   **Forget Gate**: What to throw away from cell state?
    *   **Input Gate**: What new info to store?
    *   **Output Gate**: What to output to hidden state?
*   **Result**: Can remember dependencies over 100+ steps.

### 2.2 GRU (Gated Recurrent Unit)
Simplified LSTM. Merges Cell/Hidden state. 2 gates instead of 3. Faster, often equal performance.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Sequential Processing Speed
**Scenario**: RNNs process token by token. Cannot parallelize. Training is slow on GPUs.
**Solution**: **Transformers** (Attention is All You Need). They process the whole sequence in parallel. (Covered in Phase 4).

### Challenge 2: Variable Length Sequences
**Scenario**: Sentence A has 5 words. Sentence B has 50.
**Solution**:
*   **Padding**: Pad A with zeros to length 50.
*   **Masking**: Tell the loss function to ignore the padded zeros.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why do LSTMs solve the vanishing gradient problem?**
> **Answer**: The Cell State update is additive ($C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$). In standard RNNs, it's multiplicative (matrix multiplication). Additive gradients flow much better, acting like a "gradient superhighway" similar to ResNets.

**Q2: When would you use an RNN over a Transformer in 2025?**
> **Answer**:
> 1.  **Edge Devices**: RNNs have $O(1)$ inference memory (just store hidden state). Transformers have $O(N^2)$ or $O(N)$ attention cache.
> 2.  **Streaming**: RNNs naturally handle streaming data one step at a time.
> 3.  **Small Data**: Transformers are data-hungry.

**Q3: What is Bidirectional RNN?**
> **Answer**: Two RNNs. One reads forward, one backward. Concatenate states. Useful when context from the future is needed (e.g., filling in a missing word in a sentence). Not usable for real-time forecasting.

---

## 5. Further Reading
- [Understanding LSTMs (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
