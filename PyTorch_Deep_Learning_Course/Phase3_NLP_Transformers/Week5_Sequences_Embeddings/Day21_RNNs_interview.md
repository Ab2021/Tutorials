# Day 21: RNNs & LSTMs - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Sequence Modeling, Gradients, and Architecture

### 1. Why do RNNs suffer from Vanishing Gradients?
**Answer:**
*   During BPTT, the gradient at step $t$ is the product of gradients from $t+1$ to $T$.
*   This involves repeated multiplication by the recurrent weight matrix $W_{hh}$ and the derivative of the activation (tanh/sigmoid).
*   If spectral radius of $W_{hh} < 1$ or activation derivative < 1, the product approaches zero exponentially.

### 2. How does LSTM solve the Vanishing Gradient problem?
**Answer:**
*   Through the **Cell State** ($C_t$) and the **Forget Gate**.
*   $C_t = f_t C_{t-1} + ...$
*   If $f_t = 1$, the gradient flows back linearly ($1 \times 1 \times ...$) without decay.
*   It creates a "Gradient Superhighway".

### 3. What is the difference between LSTM and GRU?
**Answer:**
*   **LSTM**: 3 Gates (Input, Forget, Output). Separate Cell State ($C$) and Hidden State ($h$).
*   **GRU**: 2 Gates (Reset, Update). Merged State.
*   GRU is computationally cheaper and often performs similarly. LSTM is more expressive for very long dependencies.

### 4. Why do we need "Gradient Clipping"?
**Answer:**
*   To prevent **Exploding Gradients**.
*   In RNNs, gradients can grow exponentially ($W > 1$).
*   Clipping ensures the update step doesn't overshoot the minimum or cause numerical overflow (NaN).

### 5. What is "Bidirectional LSTM"?
**Answer:**
*   Two LSTMs trained simultaneously: one reads left-to-right, one reads right-to-left.
*   Outputs are concatenated.
*   Allows the network to use context from both past and future (e.g., for Named Entity Recognition).
*   Cannot be used for Autoregressive generation (predicting next word).

### 6. Explain "Teacher Forcing".
**Answer:**
*   During training, instead of feeding the model's *own* predicted output $y_{t-1}$ as input for step $t$, we feed the **ground truth** $y_{t-1}$.
*   Stabilizes training and converges faster.
*   Disadvantage: Exposure Bias (Model never learns to recover from its own mistakes).

### 7. What is "Truncated BPTT"?
**Answer:**
*   Approximation for infinite sequences.
*   Run forward/backward for $k$ steps. Update weights.
*   Pass the final hidden state to the next chunk, but *detach* it from the computation graph (stop gradients).

### 8. Why use `pack_padded_sequence` in PyTorch?
**Answer:**
*   Efficiency and Correctness.
*   Without it, the LSTM processes padding tokens (zeros) as valid inputs, updating the hidden state with garbage.
*   Packing tells the LSTM to skip padding and stop processing when the sequence ends.

### 9. Can RNNs process images?
**Answer:**
*   Yes. Pixel RNNs or treating image rows as a sequence.
*   However, CNNs or ViTs are usually better due to 2D locality and parallelization.

### 10. What is the activation function of the LSTM gates? Why?
**Answer:**
*   **Sigmoid**.
*   Output range $[0, 1]$.
*   Acts as a switch: 0 (Closed/Block), 1 (Open/Pass).
*   Tanh is used for the *content* update (range $[-1, 1]$).

### 11. What is "Seq2Seq"?
**Answer:**
*   Encoder-Decoder architecture using RNNs.
*   Encoder compresses input sequence to a context vector.
*   Decoder generates output sequence from context vector.
*   Used for Translation, Summarization.

### 12. Why are RNNs slow to train?
**Answer:**
*   **Sequential Dependency**: Step $t$ cannot be computed until $t-1$ is finished.
*   Prevents parallelization on GPUs (unlike CNNs/Transformers which process all tokens in parallel).

### 13. What is "Hidden State" vs "Cell State"?
**Answer:**
*   **Hidden State ($h$)**: The output of the LSTM block. Used for prediction and passed to next layer. Constrained by Tanh $[-1, 1]$.
*   **Cell State ($C$)**: The internal memory. Linearly updated. Can grow large. Not exposed directly.

### 14. What is "Stacked LSTM"?
**Answer:**
*   Multiple LSTM layers on top of each other.
*   Output of Layer 1 is input to Layer 2.
*   Learns hierarchical features (like deep CNNs).

### 15. How do you initialize RNN weights?
**Answer:**
*   **Orthogonal Initialization**: Helps maintain the norm of gradients during backprop (eigenvalues near 1).
*   Initialize Forget Gate bias to 1.0 (to encourage remembering by default).

### 16. What is "Attention" in the context of RNNs?
**Answer:**
*   Mechanism to solve the bottleneck of the fixed-size context vector.
*   Decoder looks at *all* Encoder hidden states, computes a weighted average (Context), and uses it for prediction.

### 17. What is "Perplexity"?
**Answer:**
*   Metric for Language Models. $PPL = \exp(\text{CrossEntropy})$.
*   Measures how confused the model is. Lower is better.

### 18. Why use "Dropout" in RNNs?
**Answer:**
*   Regularization.
*   Usually applied only to the non-recurrent connections (between layers), not the recurrent ones (time steps), to preserve memory.

### 19. What is "Many-to-One" vs "Many-to-Many"?
**Answer:**
*   **Many-to-One**: Sentiment Analysis (Sequence $\to$ Class).
*   **Many-to-Many**: Translation (Sequence $\to$ Sequence) or Tagging (Frame $\to$ Label).

### 20. What is "Echo State Network"?
**Answer:**
*   A type of Reservoir Computing.
*   Recurrent weights are fixed (random). Only the output weights are trained.
*   Fast training, but less powerful than fully trained RNNs.
