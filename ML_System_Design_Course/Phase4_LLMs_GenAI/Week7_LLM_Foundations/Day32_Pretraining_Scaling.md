# Day 32: Pretraining & Scaling Laws

> **Phase**: 4 - LLMs & GenAI
> **Week**: 7 - LLM Foundations
> **Focus**: How LLMs are Made
> **Reading Time**: 50 mins

---

## 1. The Recipe for Intelligence

Pretraining is compressing the internet into a neural network.

### 1.1 The Objective
**Next Token Prediction**: $P(w_t | w_{1 \dots t-1})$.
*   Simple objective, but forces the model to learn grammar, facts, reasoning, and coding to minimize loss.

### 1.2 Tokenization (BPE)
*   LLMs don't read words; they read tokens.
*   **Byte Pair Encoding (BPE)**: Iteratively merge frequent pairs of characters.
*   **Efficiency**: "The" is 1 token. "Antidisestablishmentarianism" might be 3 tokens.
*   **Issues**: LLMs struggle with spelling and arithmetic because tokenization obscures the individual characters/digits.

---

## 2. Scaling Laws

### 2.1 Kaplan vs. Chinchilla
*   **Kaplan (2020)**: Said "Scale model size faster than data." Resulted in huge, undertrained models (GPT-3).
*   **Chinchilla (DeepMind 2022)**: Proved that for a fixed compute budget, you should scale model size and data size **equally**.
    *   **Rule of Thumb**: Train on 20 tokens for every 1 parameter.
    *   **Llama 3**: Trained on 15 Trillion tokens (far beyond Chinchilla optimal). This is "overtraining" to make the model smaller and faster at inference time (Llama-3-8B).

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Data Quality
**Scenario**: Training on "The Pile" (Common Crawl). Contains spam, PII, and toxic content.
**Solution**:
*   **Deduplication**: Exact and Fuzzy dedup. Removing duplicates improves performance significantly.
*   **Filtering**: Heuristics (e.g., "If > 50% non-alphanumeric, drop").
*   **Synthetic Data**: Using GPT-4 to generate high-quality reasoning traces (Textbooks is All You Need).

### Challenge 2: Training Instability
**Scenario**: Loss spikes. Model diverges.
**Solution**:
*   **Gradient Clipping**.
*   **Weight Decay**.
*   **Restart**: Go back to the last checkpoint, skip the bad batch, and resume.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What does "Compute Optimal" mean?**
> **Answer**: It refers to the allocation of a fixed compute budget (FLOPs) between Model Size ($N$) and Dataset Size ($D$) to achieve the lowest possible Loss. Chinchilla states $N \propto D$.

**Q2: Why do we "overtrain" models like Llama 3 (8B params on 15T tokens)?**
> **Answer**: Chinchilla optimizes for *Training Cost*. But in production, *Inference Cost* dominates. A smaller model (8B) is much cheaper to run than a larger model (70B). By overtraining the small model, we maximize its performance for its size, saving money during deployment.

**Q3: How does BPE handle unknown words (OOV)?**
> **Answer**: BPE falls back to byte-level tokens. It can represent *any* string as a sequence of bytes. There is no "Unknown Token" ($<UNK>$) in modern LLMs.

---

## 5. Further Reading
- [Training Compute-Optimal Large Language Models (Chinchilla Paper)](https://arxiv.org/abs/2203.15556)
- [Llama 3 Technical Report](https://ai.meta.com/blog/meta-llama-3/)
