# Day 32 (Part 1): Advanced Pretraining

> **Phase**: 6 - Deep Dive
> **Topic**: Data Engineering for LLMs
> **Focus**: Deduplication, Tokenization, and Curriculum
> **Reading Time**: 60 mins

---

## 1. Deduplication at Scale

### 1.1 MinHash LSH
*   **Goal**: Find near-duplicate documents in 10TB text.
*   **Algo**:
    1.  Shingle docs (n-grams).
    2.  Hash shingles. Keep min hash.
    3.  Jaccard Similarity $\approx$ Probability of collision.

### 1.2 Impact
*   Deduplication prevents the model from memorizing verbatim text (Privacy/Regurgitation).
*   Improves Zero-shot performance.

---

## 2. Tokenization Nightmares

### 2.1 Glitch Tokens
*   Tokens that appear rarely in training but have huge weights.
*   Prompting "SolidGoldMagikarp" caused GPT-3 to crash/hallucinate.
*   **Cause**: Reddit usernames in Common Crawl that were tokenized as single unique tokens but rarely seen.

### 2.2 Byte Fallback
*   BPE usually merges characters.
*   If a character is unseen (Emoji), fallback to UTF-8 bytes.

---

## 3. Tricky Interview Questions

### Q1: Why "Curriculum Learning"?
> **Answer**:
> *   Train on easy/high-quality data (Wikipedia) first.
> *   Train on noisy/hard data (Common Crawl) later.
> *   **Reality**: Reverse is often better. Train on volume first, fine-tune on quality (Annealing).

### Q2: Explain "Grokking".
> **Answer**:
> *   Model achieves 100% train accuracy but random test accuracy.
> *   Keep training for long time.
> *   Suddenly test accuracy jumps.
> *   **Theory**: Model switches from memorization to generalization.

### Q3: 3D Parallelism?
> **Answer**:
> *   **Data Parallel**: Split Batch.
> *   **Pipeline Parallel**: Split Layers.
> *   **Tensor Parallel**: Split Matrices (Megatron-LM).

---

## 4. Practical Edge Case: Multi-Lingual Imbalance
*   **Problem**: English is 90% of data. Other languages starve.
*   **Fix**: Up-sample low-resource languages.

