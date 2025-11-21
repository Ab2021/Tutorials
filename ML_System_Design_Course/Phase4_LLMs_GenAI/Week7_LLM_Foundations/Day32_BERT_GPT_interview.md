# Day 32: BERT & GPT - Interview Questions

> **Topic**: Pre-training Paradigms
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the difference between BERT and GPT?
**Answer:**
*   **BERT**: Encoder-only. Masked LM objective. Bi-directional. Good for Classification/NER.
*   **GPT**: Decoder-only. Causal LM objective. Uni-directional. Good for Generation.

### 2. Explain Masked Language Modeling (MLM).
**Answer:**
*   Randomly mask 15% of tokens.
*   Model predicts masked tokens based on context.
*   Forces model to understand context from both sides.

### 3. Explain Causal Language Modeling (CLM).
**Answer:**
*   Predict next token $w_t$ given $w_{1:t-1}$.
*   Standard objective for Generative AI.

### 4. What is Next Sentence Prediction (NSP) in BERT?
**Answer:**
*   Input: `[CLS] Sent A [SEP] Sent B`.
*   Binary classification: Is B the actual next sentence after A?
*   Helps with relationship understanding (QA, NLI). (Dropped in RoBERTa).

### 5. Why does BERT use WordPiece tokenization?
**Answer:**
*   Subword tokenization.
*   Balances vocabulary size and OOV (Out of Vocabulary) problem.
*   Common words are whole, rare words are split (`playing`, `play`, `##ing`).

### 6. What is RoBERTa? How is it different from BERT?
**Answer:**
*   "Robustly Optimized BERT".
*   Removed NSP task.
*   Dynamic Masking (mask changes every epoch).
*   More data, larger batch size.

### 7. What is T5 (Text-to-Text Transfer Transformer)?
**Answer:**
*   Encoder-Decoder.
*   Treats every task as text-to-text.
*   "Translate English to German: ..." -> "..."
*   "Summarize: ..." -> "..."

### 8. What is "Zero-Shot" Learning in GPT-3?
**Answer:**
*   Give model a task description without any examples.
*   "Translate this to French: Hello" -> Model generates "Bonjour".
*   Relies on massive pre-training.

### 9. What is "Few-Shot" Learning (In-Context Learning)?
**Answer:**
*   Give model a few examples in the prompt.
*   Input: "En: Dog, Fr: Chien \n En: Cat, Fr: Chat \n En: Mouse, Fr:"
*   Model completes the pattern. No weight updates.

### 10. Why is GPT-3 so large (175B)?
**Answer:**
*   Scaling Laws.
*   Performance scales as power law with Parameters, Data, and Compute.
*   Larger models are more sample-efficient and exhibit emergent abilities.

### 11. What is the "CLS" token in BERT?
**Answer:**
*   Special token added at start.
*   Its final hidden state is used as the aggregate representation of the sequence for classification tasks.

### 12. Can BERT generate text?
**Answer:**
*   Not naturally. It's bi-directional.
*   Can be used for "Gibbs Sampling" generation (fill in blanks iteratively), but slow and poor quality compared to GPT.

### 13. What is DistilBERT?
**Answer:**
*   Knowledge Distillation of BERT.
*   40% smaller, 60% faster, 97% performance.
*   Removes token-type embeddings and pooler.

### 14. What is the limit of context length in BERT/GPT?
**Answer:**
*   Usually 512 (BERT) or 2048/4096 (GPT).
*   Due to $O(N^2)$ attention complexity.

### 15. What is "Scaling Law"?
**Answer:**
*   Kaplan et al. (2020).
*   Loss decreases predictably as $N$ (params) and $D$ (data) increase.
*   Compute is the bottleneck.

### 16. What is "Chinchilla Scaling"?
**Answer:**
*   Hoffmann et al. (2022).
*   Optimal scaling: Double model size $\leftrightarrow$ Double data size.
*   Most models (GPT-3) were under-trained (too big, not enough data).

### 17. What is "Emergent Behavior"?
**Answer:**
*   Capabilities that appear suddenly at certain scale (e.g., Arithmetic, Code generation).
*   Not present in smaller models.

### 18. How do you handle long documents with BERT?
**Answer:**
*   Sliding Window.
*   Hierarchical BERT (Process chunks, then aggregate).
*   Longformer / BigBird (Sparse attention).

### 19. What is BART?
**Answer:**
*   Denoising Autoencoder (Encoder-Decoder).
*   Corrupt text (mask, delete, shuffle) -> Reconstruct original.
*   Good for Summarization.

### 20. Why is Decoder-only architecture dominant for LLMs?
**Answer:**
*   Efficiency in generation (KV Cache).
*   Strong zero-shot performance.
*   Simpler pre-training objective (just predict next token).
