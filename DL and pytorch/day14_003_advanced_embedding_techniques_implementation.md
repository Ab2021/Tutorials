# Day 14.3: Advanced Embedding Techniques - A Practical Guide

## Introduction: Beyond Word2Vec

Word2Vec was revolutionary, but it has a significant limitation: it generates a **single, static vector** for each word. This is a problem for words with multiple meanings, known as **polysemy**. For example, the word "bank" has a very different meaning in "river bank" versus "investment bank," but Word2Vec would represent both with the exact same vector.

To address this and other limitations, a new generation of embedding techniques emerged. These models create **contextualized embeddings**, where the vector representation for a word changes depending on the sentence it appears in.

This guide provides a practical overview of two of the most influential advanced embedding techniques: **GloVe** (which combines the best of count-based and prediction-based methods) and **ELMo** (a pioneer in contextualized embeddings).

**Today's Learning Objectives:**

1.  **Understand the Limitations of Static Embeddings:** Grasp the problem of polysemy with Word2Vec.
2.  **Learn the Core Idea of GloVe:** Understand how GloVe (Global Vectors) combines the global statistics of count-based methods with the local context window of prediction-based methods.
3.  **Grasp the Concept of Contextualized Embeddings (ELMo):** Learn how ELMo (Embeddings from Language Models) uses a deep, bidirectional LSTM to generate word vectors that are a function of the entire input sentence.
4.  **Use Pre-trained GloVe Embeddings:** Learn how to load and use pre-trained GloVe vectors in a PyTorch `nn.Embedding` layer.
5.  **Appreciate the Evolution to Transformers:** Understand how these models paved the way for fully contextualized models like BERT and GPT.

---

## Part 1: GloVe - Global Vectors for Word Representation

**The Idea:** The creators of GloVe argued that both count-based methods (like co-occurrence matrices) and prediction-based methods (like Word2Vec) have their own strengths.
*   **Count-based methods** are good at capturing global statistical information but are poor at capturing complex patterns like analogy.
*   **Prediction-based methods** are good at capturing complex patterns but fail to use the global statistical information efficiently.

GloVe combines the best of both worlds. It is trained on **global word-word co-occurrence statistics**, but its objective function is designed in a way that the resulting vectors exhibit the linear structures (like the king-queen analogy) that made Word2Vec famous.

**How it Works (High-Level):**
1.  It first builds a large, global co-occurrence matrix from a corpus.
2.  It then defines a specific loss function that tries to make the dot product of two word vectors equal to the logarithm of their probability of co-occurrence.
3.  It uses an optimization procedure (like Adagrad) to learn the embedding vectors that satisfy this loss function.

**The Takeaway:** You can think of GloVe as a more principled way to perform dimensionality reduction on a co-occurrence matrix, specifically designed to produce a vector space with meaningful linear substructures.

### 1.1. Using Pre-trained GloVe Embeddings

Training GloVe from scratch is a major undertaking. The common practice is to use the pre-trained vectors released by the Stanford NLP group. We can easily load these into a PyTorch `nn.Embedding` layer.

```python
import torch
import torch.nn as nn
import torchtext.vocab as vocab

print("--- Part 1: Using Pre-trained GloVe Embeddings ---")

# --- 1. Load the Pre-trained GloVe vectors ---
# torchtext provides a convenient way to download and load these.
# This might take a few minutes on the first run.
# '6B' refers to the version trained on 6 billion tokens.
# `dim=100` means we are loading the 100-dimensional vectors.
glove = vocab.GloVe(name='6B', dim=100)

print(f"Loaded {len(glove.itos)} words from GloVe.")

# --- 2. Get the vector for a specific word ---
cat_vector = glove['cat']
kitten_vector = glove['kitten']
car_vector = glove['car']

print(f"Shape of the 'cat' vector: {cat_vector.shape}")

# --- 3. Measure Cosine Similarity ---
cos = nn.CosineSimilarity(dim=0)
sim_cat_kitten = cos(cat_vector, kitten_vector)
sim_cat_car = cos(cat_vector, car_vector)

print(f"\nCosine similarity between 'cat' and 'kitten': {sim_cat_kitten:.4f}")
print(f"Cosine similarity between 'cat' and 'car': {sim_cat_car:.4f}")

# --- 4. Load GloVe weights into an nn.Embedding layer ---
# This is a common pattern for initializing your model's embedding layer.

# Assume we have our own vocabulary from our specific dataset
my_vocab = {'the': 0, 'cat': 1, 'car': 2, '<unk>': 3}
my_vocab_size = len(my_vocab)
embedding_dim = 100

# Create the embedding layer
my_embedding_layer = nn.Embedding(my_vocab_size, embedding_dim)

# Create a weight matrix to hold the GloVe vectors
glove_weights = torch.zeros(my_vocab_size, embedding_dim)
for word, index in my_vocab.items():
    try:
        glove_weights[index] = glove[word]
    except KeyError:
        # For words in our vocab but not in GloVe, we leave them as zeros
        # or initialize them randomly.
        glove_weights[index] = torch.randn(embedding_dim)

# Load the pre-trained weights into our embedding layer
my_embedding_layer.weight.data.copy_(glove_weights)

# Now, this embedding layer is initialized with GloVe vectors!
# We can choose to freeze it or fine-tune it during training.
my_embedding_layer.weight.requires_grad = False # Freeze the layer

print("\nSuccessfully loaded GloVe weights into a custom nn.Embedding layer.")
```

---

## Part 2: ELMo - Context Matters

**The Problem:** GloVe and Word2Vec are still **static**. The vector for "bank" is always the same.

**The Solution: ELMo (Embeddings from Language Models)**

ELMo was a major breakthrough that introduced truly **contextualized embeddings**.

**How it Works (High-Level):**
1.  **Deep Bidirectional LSTM:** ELMo is not just an embedding matrix; it's a large, pre-trained, multi-layer bidirectional LSTM.
2.  **Character-based Input:** It starts with character-level convolutions to handle out-of-vocabulary words.
3.  **Function of the Entire Sentence:** To get the embedding for a word, you feed the **entire sentence** into the pre-trained biLSTM. The representation for that word is then computed as a **function of the internal states of the LSTM**. Specifically, it's a weighted average of the hidden states from all layers of the biLSTM for that word.

**The Key Insight:** The vector for the word "bank" in "river bank" will be different from the vector for "bank" in "investment bank," because the hidden states of the LSTM will be different due to the different contexts.

![ELMo](https://i.imgur.com/3gQ5z0A.png)

**The Takeaway:** ELMo marked the beginning of the end for static embeddings. It showed that pre-training a deep neural network on a language modeling task and then using its internal states as embeddings was a powerful way to capture context.

---

## Part 3: The Evolution to Transformers

ELMo was a pivotal step, but it still relied on LSTMs, which are sequential and can be slow to train.

The final step in this evolution was to replace the LSTM in the language model with the more powerful and parallelizable **Transformer** architecture. This led directly to the models that dominate modern NLP:

*   **BERT (Bidirectional Encoder Representations from Transformers):** Can be seen as the successor to ELMo's bidirectional philosophy. It uses a deep Transformer encoder and is pre-trained on a masked language modeling task, allowing it to learn rich, bidirectional contextual embeddings.

*   **GPT (Generative Pre-trained Transformer):** Uses a deep Transformer decoder and is pre-trained on a standard left-to-right language modeling task, making it exceptionally good at text generation.

These Transformer-based models don't just produce embeddings; they are powerful models in their own right that can be directly fine-tuned for downstream tasks, as we saw in Day 8.

## Conclusion

The journey from static to contextualized embeddings represents a major leap in NLP. While static embeddings like Word2Vec and GloVe are still useful and lightweight, the ability of contextual models to handle polysemy and capture the nuances of language has made them the standard for nearly all state-of-the-art systems.

**Key Takeaways:**

1.  **Static vs. Contextual:** Word2Vec and GloVe produce a single, static vector for each word type. ELMo, BERT, and GPT produce a different vector for each word token, depending on its context.
2.  **GloVe Combines Paradigms:** It leverages global co-occurrence statistics (like count-based methods) but is optimized in a way that produces powerful linear analogies (like prediction-based methods).
3.  **ELMo Introduced Deep Context:** It was a pioneer in using the internal states of a deep, pre-trained bidirectional language model as the word embedding.
4.  **Transformers are the Successors:** Models like BERT took the contextual idea from ELMo and supercharged it with the more powerful and parallelizable Transformer architecture, leading to the current state of the art.
5.  **Practical Usage:** For most modern applications, you will get the best performance by using a pre-trained Transformer model (like BERT) and fine-tuning it, rather than using static GloVe or Word2Vec embeddings as an input layer.

## Self-Assessment Questions

1.  **Polysemy:** What is the main limitation of static embedding models like Word2Vec and GloVe?
2.  **GloVe's Core Idea:** What two different approaches to creating embeddings does GloVe try to combine?
3.  **ELMo's Core Idea:** How does ELMo generate a different vector for the same word in different sentences?
4.  **Using GloVe:** What is the standard way to incorporate pre-trained GloVe vectors into your own PyTorch model?
5.  **Evolution:** How did the architecture of ELMo lead to the development of models like BERT?
