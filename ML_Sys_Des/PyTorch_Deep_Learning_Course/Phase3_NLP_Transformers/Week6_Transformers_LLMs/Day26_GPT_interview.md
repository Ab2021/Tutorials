# Day 26: GPT & Decoders - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Generative Models, Sampling, and Optimization

### 1. What is the difference between "Causal" and "Masked" Language Modeling?
**Answer:**
*   **Causal (GPT)**: Predict next token based on *past* tokens only. Unidirectional.
*   **Masked (BERT)**: Predict masked token based on *past and future* tokens. Bidirectional.

### 2. Why is GPT called "Autoregressive"?
**Answer:**
*   Because it consumes its own output as input for the next step.
*   $y_t = f(y_{t-1}, ...)$.

### 3. What is "KV Cache"? Why is it important?
**Answer:**
*   Storing the Key and Value matrices of previous tokens during generation.
*   Avoids recomputing attention for the entire history at every step.
*   Reduces complexity from $O(N^3)$ to $O(N^2)$ (or $O(N)$ per step).

### 4. Explain "Top-P" (Nucleus) Sampling.
**Answer:**
*   Instead of sampling from the top $K$ words, we sample from the smallest set of words whose cumulative probability exceeds $P$ (e.g., 0.9).
*   Allows for dynamic vocabulary size. If the model is unsure (flat distribution), it considers many words. If sure (peaked), it considers few.

### 5. What is "Temperature" in sampling?
**Answer:**
*   A scaling factor for logits before Softmax.
*   $T < 1$: Exaggerates differences (makes high probs higher). Reduces diversity.
*   $T > 1$: Dampens differences (makes distribution flatter). Increases diversity.

### 6. What is "In-Context Learning"?
**Answer:**
*   The ability of LLMs to perform a task just by seeing examples in the prompt (Few-Shot), without weight updates.
*   "Translate English to French: Cat -> Chat, Dog -> Chien, Apple -> ?"

### 7. What is "RoPE" (Rotary Positional Embedding)?
**Answer:**
*   Encodes position by rotating the query/key vectors.
*   Preserves relative position information via the rotation angle difference.
*   Allows better extrapolation to longer sequences.

### 8. Why do we use "Pre-Norm" in GPT-3?
**Answer:**
*   Putting LayerNorm *inside* the residual block (before attention/FFN).
*   Prevents gradient explosion in deep networks.
*   Allows training without Warmup (sometimes).

### 9. What is "Flash Attention"?
**Answer:**
*   An algorithm to compute Exact Attention faster and with less memory.
*   Uses tiling to keep data in GPU SRAM (fast cache), minimizing reads/writes to HBM (slow memory).

### 10. What is "Chinchilla Scaling Law"?
**Answer:**
*   Paper by DeepMind.
*   States that most models are undertrained.
*   Optimal compute allocation: Scale parameters and training tokens equally.
*   For a 10B model, you need ~200B tokens (20x ratio).

### 11. What is "Hallucination"?
**Answer:**
*   When the LLM generates factually incorrect or nonsensical information confidently.
*   Caused by the probabilistic nature of next-token prediction (it prioritizes plausibility over truth).

### 12. What is "Beam Search" in GPT?
**Answer:**
*   Keeping top $K$ sequences.
*   Rarely used in open-ended generation (chatbots) because it leads to repetitive/boring text.
*   Used in Translation/Summarization.

### 13. What is "SwiGLU"?
**Answer:**
*   Activation function used in LLaMA.
*   $\text{Swish}(xW) \cdot (xV)$.
*   Adds a gating mechanism to the FFN.

### 14. Why is the context window limited?
**Answer:**
*   $O(N^2)$ memory and compute complexity of Attention.
*   KV Cache also grows linearly with $N$, consuming VRAM.

### 15. What is "Perplexity"?
**Answer:**
*   $2^{Entropy}$.
*   The weighted average branching factor.
*   If PPL = 10, the model is as confused as if it had to choose uniformly from 10 words.

### 16. What is "Instruction Tuning"?
**Answer:**
*   Fine-tuning a base LLM (which just completes text) on a dataset of (Instruction, Output) pairs.
*   Makes the model helpful and capable of following commands (ChatGPT vs GPT-3).

### 17. What is "RLHF" (Reinforcement Learning from Human Feedback)?
**Answer:**
*   Training a Reward Model to predict human preference.
*   Optimizing the LLM to maximize this reward using PPO (Proximal Policy Optimization).
*   Aligns model with human values (Helpful, Honest, Harmless).

### 18. What is "Grouped Query Attention" (GQA)?
**Answer:**
*   Used in LLaMA-2/3.
*   Interpolation between Multi-Head (MHA) and Multi-Query (MQA).
*   Multiple query heads share a single key/value head.
*   Reduces KV Cache size and memory bandwidth usage.

### 19. What is "Zero-Shot" vs "Few-Shot"?
**Answer:**
*   **Zero-Shot**: No examples in prompt. "Translate this: ..."
*   **Few-Shot**: Examples in prompt. "En: Cat, Fr: Chat. En: Dog, Fr: Chien. En: Apple, Fr: ..."

### 20. Why does GPT use BPE?
**Answer:**
*   To handle open vocabulary.
*   Ensures every string can be tokenized (fallback to bytes).
*   Compresses common words into single tokens.
