# Day 22: Word Embeddings - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Positional Encodings, Contextual Embeddings (ELMo), and Cosine Similarity

## 1. The Limit of Static Embeddings

Word2Vec/GloVe are **Static**.
*   "Bank" (River) and "Bank" (Money) have the *same* vector.
*   This polysemy limits performance.

**Contextualized Embeddings (ELMo)**:
*   Use a Bi-LSTM to compute embeddings *on the fly* based on the sentence.
*   $Emb(Bank) = f(sentence)$.
*   Precursor to BERT.

## 2. Positional Encoding

Transformers process words in parallel (Set), not Sequence. They don't know order.
We must inject position info.
**Sinusoidal Encoding (Vaswani)**:
$$ PE(pos, 2i) = \sin(pos / 10000^{2i/d}) $$
$$ PE(pos, 2i+1) = \cos(pos / 10000^{2i/d}) $$
*   Added to the word embedding: $x = Embed(w) + PE(pos)$.
*   Allows the model to learn relative positions ($PE_{pos+k}$ is a linear function of $PE_{pos}$).

## 3. Cosine Similarity

Why not Euclidean Distance?
*   High-dimensional spaces are weird. Euclidean distance is sensitive to vector magnitude (frequency).
*   Cosine Similarity measures the **Angle**.
$$ \text{Sim}(A, B) = \frac{A \cdot B}{||A|| ||B||} $$
*   Range: [-1, 1].

## 4. Embedding Projector

Visualizing 300D vectors in 2D/3D.
*   **PCA**: Preserves global variance. Linear.
*   **t-SNE**: Preserves local neighborhood. Non-linear. Great for clusters.
*   **UMAP**: Faster and better global structure than t-SNE.

## 5. Hierarchical Softmax

Alternative to Negative Sampling for Word2Vec.
*   Organize vocab in a Huffman Tree.
*   Probability of a word = Product of probabilities along the path in the tree.
*   Reduces complexity from $O(V)$ to $O(\log V)$.
