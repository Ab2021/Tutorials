# Day 20 (Part 1): Advanced Sequence Modeling

> **Phase**: 6 - Deep Dive
> **Topic**: Time Series & NLP
> **Focus**: BPTT, LSTM Internals, and Attention
> **Reading Time**: 60 mins

---

## 1. Backpropagation Through Time (BPTT)

RNNs are just very deep networks with shared weights.

### 1.1 The Gradient
*   $\frac{\partial L}{\partial W} = \sum_{t} \frac{\partial L_t}{\partial W}$.
*   Chain rule involves product of Jacobians: $\prod_{k=t}^{1} \frac{\partial h_k}{\partial h_{k-1}}$.
*   If Jacobian eigenvalues > 1: **Exploding Gradient**.
*   If Jacobian eigenvalues < 1: **Vanishing Gradient**.

---

## 2. LSTM Internals

Why does it solve vanishing gradient?

### 2.1 The Cell State ($C_t$)
*   $C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$.
*   **The Key**: The gradient flows through the $+$ operator.
*   $\frac{\partial C_t}{\partial C_{t-1}} = f_t$ (Forget Gate).
*   If $f_t \approx 1$, gradient flows unchanged. It acts like a ResNet highway for time.

---

## 3. Connectionist Temporal Classification (CTC)

How to train Speech-to-Text when Input (Audio) length $\neq$ Output (Text) length?

### 3.1 The Alignment Problem
*   Audio: "HHHEEELLLLLOOO". Text: "HELLO".
*   **CTC**: Introduces a "Blank" token $\epsilon$.
*   Predicts: "H \epsilon E \epsilon L L \epsilon O".
*   Collapses repeats and removes blanks.
*   **Loss**: Sum probability of *all valid alignments*.

---

## 4. Tricky Interview Questions

### Q1: Why use Tanh for cell state but Sigmoid for gates?
> **Answer**:
> *   **Sigmoid (0 to 1)**: Perfect for gating (Open/Close). 0 = Block, 1 = Pass.
> *   **Tanh (-1 to 1)**: Centers data around 0. Prevents internal state from drifting to infinity (Exploding) which happens with ReLU in RNNs.

### Q2: Explain Bahdanau Attention.
> **Answer**:
> *   **Context Vector**: Instead of a fixed context $C$ from the last hidden state, compute a weighted sum of *all* encoder states $h_i$.
> *   **Weights**: $\alpha_{ti} = \text{Softmax}(\text{Score}(s_{t-1}, h_i))$.
> *   Allows the decoder to "look back" at specific source words.

### Q3: Teacher Forcing?
> **Answer**:
> *   During training, feed the *Ground Truth* previous token as input, not the Model's *Predicted* previous token.
> *   **Pros**: Faster convergence.
> *   **Cons**: Exposure Bias. Model fails at inference time because it never learned to recover from its own mistakes.

---

## 5. Practical Edge Case: Variable Length Sequences
*   **Problem**: Batching sentences of length 10 and 100.
*   **Fix**: Padding + Masking.
*   **PyTorch**: `pack_padded_sequence`. Optimizes compute by ignoring pad tokens.

