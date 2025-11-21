# Day 22: Word Embeddings - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Word2Vec, GloVe, and Embedding Layers

## 1. Theoretical Foundation: From One-Hot to Dense

**One-Hot Encoding**:
*   Vector size = Vocab size (e.g., 50,000).
*   Sparse. Orthogonal.
*   Problem: $dot(king, queen) = 0$. No semantic similarity.

**Distributed Representation (Embeddings)**:
*   Dense vector (e.g., 300 dim).
*   Learned from context (Distributional Hypothesis: "Words that appear in similar contexts have similar meanings").
*   $king - man + woman \approx queen$.

## 2. Word2Vec (Mikolov et al., 2013)

Two architectures:
1.  **CBOW (Continuous Bag of Words)**: Predict Center word from Context words.
2.  **Skip-Gram**: Predict Context words from Center word.

**Training Trick: Negative Sampling**
Instead of calculating Softmax over 50k words (expensive denominator), we train a binary classifier:
*   Real Pair: (fox, jumps) $\to$ 1.
*   Noise Pair: (fox, broccoli) $\to$ 0.

## 3. GloVe (Global Vectors)

Word2Vec relies on local windows. GloVe uses global statistics.
Constructs a Co-occurrence Matrix $X$.
Factorizes $X$ such that $w_i^T w_j \approx \log P(j|i)$.

## 4. Implementation: PyTorch Embedding Layer

`nn.Embedding` is essentially a Lookup Table.

```python
import torch
import torch.nn as nn

# 1. Initialize from scratch
vocab_size = 10000
embed_dim = 300
embedding = nn.Embedding(vocab_size, embed_dim)

# Input: Indices (Batch, Seq)
input_indices = torch.tensor([[1, 5, 9], [2, 4, 8]])
output = embedding(input_indices) # (2, 3, 300)

# 2. Load Pre-trained Vectors (GloVe)
# Assume we have a matrix 'glove_weights' (numpy)
glove_weights = torch.from_numpy(load_glove_vectors())
embedding.weight.data.copy_(glove_weights)
embedding.weight.requires_grad = False # Freeze if needed
```

## 5. Subword Embeddings (FastText)

Word2Vec fails for OOV (Out of Vocabulary) words.
**FastText**: Represents a word as a bag of character n-grams.
*   "apple" $\to$ <ap, app, ppl, ple, le>
*   Vector("apple") = Sum(Vector(n-grams)).
*   Can generate vectors for unseen words!
