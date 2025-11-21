# Day 27: Tokenization - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Subword Algorithms, Unicode, and Efficiency

### 1. Why do we need Subword Tokenization?
**Answer:**
*   To solve the OOV (Out of Vocabulary) problem of word-level models.
*   To reduce sequence length compared to character-level models.
*   To capture morphological meaning (e.g., "walking" $\to$ "walk" + "ing").

### 2. Explain the BPE algorithm.
**Answer:**
*   Initialize vocab with characters.
*   Count frequency of adjacent pairs.
*   Merge most frequent pair.
*   Add merged token to vocab.
*   Repeat until target vocab size is reached.

### 3. What is the difference between BPE and WordPiece?
**Answer:**
*   **BPE**: Merges based on frequency.
*   **WordPiece**: Merges based on likelihood improvement (Language Model probability).
*   WordPiece chooses the merge that maximizes the probability of the training data.

### 4. What is "Byte-Level BPE"?
**Answer:**
*   Applying BPE on bytes (UTF-8) instead of Unicode characters.
*   Base vocab is 256.
*   Guarantees no `<UNK>` tokens (any string can be represented as bytes).
*   Used in GPT-2/3/4.

### 5. What is "SentencePiece"?
**Answer:**
*   A library/method that treats input as a raw stream of characters, including spaces.
*   Does not rely on pre-tokenization (like splitting by whitespace).
*   Language agnostic (works for Japanese/Chinese where no spaces exist).

### 6. Why does LLaMA use a larger vocabulary (32k/128k) than BERT (30k)?
**Answer:**
*   Larger vocab = Better compression.
*   "The quick brown fox" might be 4 tokens instead of 6.
*   Shorter sequences = Faster inference (Attention is $O(N^2)$) and larger context window.

### 7. What is the `##` in BERT tokens?
**Answer:**
*   It denotes a suffix.
*   "playing" $\to$ "play", "##ing".
*   "##ing" means "ing" attached to the previous word, not the word "ing".

### 8. How does Tokenization affect model performance?
**Answer:**
*   Poor tokenization (e.g., splitting numbers digit-by-digit) makes arithmetic hard for LLMs.
*   Splitting code keywords affects coding performance.
*   Consistent tokenization is crucial for multilingual models.

### 9. What is "Unigram" tokenization?
**Answer:**
*   A probabilistic method.
*   Starts with a huge vocab and prunes tokens that don't help likelihood.
*   Allows sampling multiple segmentations for regularization (Subword Regularization).

### 10. What is "NFKC" normalization?
**Answer:**
*   Unicode normalization form.
*   Standardizes characters (e.g., converting "ﬁ" ligature to "f" + "i", or Full-width "Ａ" to "A").
*   Crucial for consistent input processing.

### 11. Why is `<UNK>` bad?
**Answer:**
*   It destroys information. The model has no idea what the word was.
*   Byte-level BPE eliminates this.

### 12. What is "Pre-tokenization"?
**Answer:**
*   The step before the main algorithm (BPE).
*   Usually splitting by whitespace or punctuation.
*   GPT-2 uses a regex to split contractions ("don't" $\to$ "don", "'t").

### 13. How do you handle multiple languages?
**Answer:**
*   Train a shared tokenizer on a multilingual corpus (Wikipedia).
*   The tokenizer learns common subwords across languages (e.g., "tion", "ing", roots).
*   Increases vocab size (e.g., XLM-R uses 250k).

### 14. What is "Tiktoken"?
**Answer:**
*   OpenAI's fast BPE implementation.
*   Optimized for speed.
*   Includes special tokens for FIM (Fill-In-the-Middle) and end-of-text.

### 15. Why do we add special tokens like `[CLS]` and `[SEP]`?
**Answer:**
*   To mark structural boundaries.
*   `[CLS]`: Aggregate representation.
*   `[SEP]`: Sentence boundary.
*   The model learns to attend to these tokens for specific tasks.

### 16. What is the cost of a large vocabulary?
**Answer:**
*   **Parameters**: Embedding matrix is $V \times D$. Output layer is $D \times V$.
*   **Compute**: Softmax over $V$ is expensive.

### 17. Can we change the tokenizer after pre-training?
**Answer:**
*   No. The model weights are tied to specific token indices.
*   Changing tokenizer invalidates the embedding matrix.

### 18. How does SentencePiece handle spaces?
**Answer:**
*   It replaces space with a meta-symbol (e.g., `_` U+2581).
*   "Hello World" $\to$ "Hello", "_World".
*   Allows lossless reconstruction (detokenization).

### 19. What is "Greedy" vs "Optimal" tokenization?
**Answer:**
*   BPE is greedy (merge most frequent).
*   Unigram finds the Viterbi path (most probable segmentation).

### 20. Why do math/code models need specific tokenizers?
**Answer:**
*   Standard text tokenizers might split numbers "123" into "1", "23".
*   Code tokenizers need to preserve indentation (tabs/spaces) which are usually stripped in text.
