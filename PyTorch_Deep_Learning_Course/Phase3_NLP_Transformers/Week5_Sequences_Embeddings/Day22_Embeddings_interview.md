# Day 22: Word Embeddings - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Word2Vec, GloVe, and Vector Space

### 1. What is the "Distributional Hypothesis"?
**Answer:**
*   "You shall know a word by the company it keeps" (Firth, 1957).
*   Words appearing in similar contexts have similar meanings.
*   This is the basis for all modern embedding methods (Word2Vec, BERT).

### 2. Difference between CBOW and Skip-Gram?
**Answer:**
*   **CBOW**: Predicts Center word from Sum/Avg of Context words. Faster. Better for frequent words.
*   **Skip-Gram**: Predicts Context words from Center word. Slower. Better for rare words and semantic nuances.

### 3. Why do we use "Negative Sampling"?
**Answer:**
*   Computing the full Softmax over a large vocabulary (e.g., 100k words) is computationally expensive (denominator sum).
*   Negative Sampling approximates the gradient by updating the correct class and a few random incorrect classes (noise).

### 4. How does GloVe differ from Word2Vec?
**Answer:**
*   **Word2Vec**: Predictive model (Neural Network). Learns online.
*   **GloVe**: Count-based model (Matrix Factorization). Learns from the global co-occurrence matrix.
*   GloVe explicitly optimizes the dot product to equal the log-probability of co-occurrence.

### 5. What is the problem with One-Hot Encoding?
**Answer:**
*   **High Dimensionality**: Vector size = Vocab size.
*   **Sparsity**: Mostly zeros.
*   **No Semantics**: All vectors are orthogonal. Distance between "Cat" and "Dog" is same as "Cat" and "Car".

### 6. How does FastText handle OOV words?
**Answer:**
*   It represents words as bags of character n-grams (e.g., "apple" = ap, app, ppl, ple, le).
*   For an unknown word, it sums the vectors of its constituent n-grams.
*   Allows capturing morphological similarity (prefixes/suffixes).

### 7. What is "Polysemy" and how does Word2Vec handle it?
**Answer:**
*   Polysemy: A word having multiple meanings (e.g., "Bank").
*   Word2Vec **fails** to handle it. It assigns a single static vector that is a weighted average of all meanings.
*   Contextual embeddings (ELMo, BERT) solve this.

### 8. Why Cosine Similarity instead of Euclidean Distance?
**Answer:**
*   In high dimensions, Euclidean distance is affected by the magnitude (length) of the vector.
*   Magnitude often correlates with word frequency, not meaning.
*   Cosine similarity focuses on the orientation (angle), which captures semantic relationship.

### 9. What is "t-SNE"?
**Answer:**
*   t-Distributed Stochastic Neighbor Embedding.
*   Dimensionality reduction technique for visualization.
*   Preserves local structure (neighbors stay neighbors).
*   Non-linear.

### 10. Can we initialize embeddings randomly?
**Answer:**
*   Yes. If you have a large dataset, you can learn task-specific embeddings from scratch (`nn.Embedding`).
*   For small datasets, it's better to use pre-trained vectors (Transfer Learning).

### 11. What is "Stop Word Removal"? Is it necessary?
**Answer:**
*   Removing common words (the, a, is).
*   Necessary for Count-based methods (TF-IDF) to reduce noise.
*   **Not** recommended for Deep Learning (BERT/LSTM) because these words provide grammatical structure and context.

### 12. What is the dimension of the embedding vector?
**Answer:**
*   Hyperparameter (usually 50, 100, 300, 768).
*   Larger dim = More capacity, risk of overfitting.
*   Smaller dim = Compression, loss of nuance.

### 13. Explain "Hierarchical Softmax".
**Answer:**
*   An efficient way to compute Softmax using a binary tree (Huffman Tree).
*   Reduces complexity from $O(V)$ to $O(\log V)$.
*   Used in original Word2Vec paper.

### 14. What happens if you add two word vectors?
**Answer:**
*   You get a vector representing the combined meaning (e.g., "Vietnam" + "Capital" $\approx$ "Hanoi").
*   This linearity is a surprising emergent property of Word2Vec.

### 15. What is "Byte Pair Encoding" (BPE)?
**Answer:**
*   A subword tokenization algorithm.
*   Iteratively merges the most frequent pair of characters/tokens.
*   Balances between Character-level (too granular) and Word-level (too sparse) representations.
*   Used in GPT, RoBERTa.

### 16. How do you handle "Padding" in embeddings?
**Answer:**
*   Use a special `<PAD>` token (index 0).
*   Set its embedding vector to all zeros.
*   Tell the layer to ignore it: `nn.Embedding(..., padding_idx=0)`.

### 17. What is "TF-IDF"?
**Answer:**
*   Term Frequency - Inverse Document Frequency.
*   Classical weighting scheme.
*   High weight for words that are frequent in *this* document but rare in *all* documents (Discriminative words).

### 18. Why is the embedding layer often the largest part of the model?
**Answer:**
*   Parameters = $V \times D$.
*   If $V=50,000$ and $D=1024$, that's 50M parameters just for the lookup table.
*   ALBERT solves this by factorizing the embedding matrix.

### 19. What is "Rotary Positional Embedding" (RoPE)?
**Answer:**
*   Encodes position by rotating the embedding vector in the complex plane.
*   Used in LLaMA/PaLM.
*   Allows better generalization to longer sequence lengths than absolute sinusoidal encodings.

### 20. What is "Cross-Lingual Embedding"?
**Answer:**
*   Aligning embedding spaces of two languages (English and French) so that "Cat" and "Chat" are close.
*   Enables Zero-Shot Cross-Lingual Transfer.
