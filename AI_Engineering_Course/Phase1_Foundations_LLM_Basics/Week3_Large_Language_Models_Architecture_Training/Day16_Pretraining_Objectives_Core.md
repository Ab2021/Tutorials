# Day 16: Pre-training Objectives
## Core Concepts & Theory

### The Self-Supervised Revolution

Before 2018, NLP relied on supervised learning (labeled data).
The breakthrough of modern LLMs is **Self-Supervised Learning (SSL)**: learning from raw, unlabeled text by creating a "pretext task" where the data itself provides the labels.

### 1. Causal Language Modeling (CLM)

**Used by:** GPT-1/2/3, LLaMA, Mistral, PaLM.
**Task:** Predict the next token given the previous tokens.
$$ P(x) = \prod_{i=1}^n P(x_i | x_{<i}) $$

**Mechanism:**
- Input: "The cat sat on"
- Target: "cat sat on the"
- Loss: Cross Entropy at each position.

**Pros:**
- **Generative:** Naturally suited for text generation.
- **Scalable:** Infinite training data (just scrape the web).
- **Simple:** Decoder-only architecture.

**Cons:**
- **Unidirectional:** Token at position $i$ cannot see tokens at $i+1$. Limits understanding of bidirectional context (e.g., "The bank of the river" vs "The bank of America").

### 2. Masked Language Modeling (MLM)

**Used by:** BERT, RoBERTa, DeBERTa.
**Task:** Mask a percentage of tokens (usually 15%) and predict them based on surrounding context (left and right).

**Mechanism:**
- Input: "The [MASK] sat on the [MASK]."
- Target: "cat", "mat".
- Loss: Cross Entropy only on masked positions.

**Pros:**
- **Bidirectional:** Model sees full context. Better for understanding tasks (Classification, NER, QA).

**Cons:**
- **Sample Inefficient:** Only learns from 15% of tokens per batch (vs 100% for CLM).
- **Train-Test Mismatch:** `[MASK]` token doesn't appear during inference.

### 3. Permutation Language Modeling (PLM)

**Used by:** XLNet.
**Task:** Predict tokens in a random order, but with access to "previous" tokens in the permutation.
**Goal:** Combine benefits of CLM (autoregressive) and MLM (bidirectional).
**Status:** Largely superseded by standard CLM and T5-style objectives.

### 4. Span Corruption (Denoising)

**Used by:** T5, UL2.
**Task:** Mask contiguous spans of text and generate the missing spans.
- Input: "The cat [Sentinel_0] the mat."
- Target: "[Sentinel_0] sat on [Sentinel_1]"

**Pros:**
- **Efficient:** Predicts multiple tokens at once.
- **Flexible:** Can be tuned for different corruption rates.

### 5. Next Sentence Prediction (NSP)

**Used by:** Original BERT.
**Task:** Given Sentence A and Sentence B, predict if B follows A.
- Input: `[CLS] The man went to the store [SEP] He bought milk [SEP]` -> Label: IsNext
- Input: `[CLS] The man went to the store [SEP] Penguins are birds [SEP]` -> Label: NotNext

**Status:** **Deprecated.**
Later research (RoBERTa) showed NSP is not helpful and might even hurt performance. It was replaced by training on long contiguous sequences (Full-Sentences).

### Summary of Objectives

| Objective | Type | Example Models | Best For |
| :--- | :--- | :--- | :--- |
| **CLM** | Autoregressive | GPT, LLaMA | Generation, Zero-shot |
| **MLM** | Autoencoding | BERT, RoBERTa | Classification, Embedding |
| **Span** | Seq2Seq | T5, BART | Translation, Summarization |

### Next Steps
In the Deep Dive, we will analyze why CLM has won the race for general-purpose LLMs despite the theoretical advantages of bidirectional models.
