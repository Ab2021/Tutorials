# Day 14.2: Word2Vec & Skip-gram Models - A Practical Guide

## Introduction: Learning Embeddings from Context

In the previous guide, we saw that count-based methods like TF-IDF can create useful document vectors, but they fail to capture the deeper semantic meaning or word order. The breakthrough in modern NLP came with **prediction-based models** that learn word embeddings by training a neural network on a simple, self-supervised task.

**Word2Vec** is the name for a collection of such models, with the two most famous being the **Continuous Bag-of-Words (CBOW)** model and the **Skip-gram** model. Instead of just counting co-occurrences, these models *learn* to embed words into a vector space where vectors capture rich semantic relationships. This is the model that famously learned that `vector('King') - vector('Man') + vector('Woman')` results in a vector very close to `vector('Queen')`.

This guide will provide a practical deep dive into the **Skip-gram** architecture, the more popular of the two, showing how it works and how to implement it from scratch in PyTorch.

**Today's Learning Objectives:**

1.  **Understand the Prediction-based Approach:** Grasp the core idea of learning embeddings by predicting context words.
2.  **Learn the Skip-gram Architecture:** Understand its objective: given a target word, predict the words that appear in its context window.
3.  **Implement a Skip-gram Dataset:** Write the code to generate `(target, context)` word pairs from a corpus.
4.  **Build and Train a Skip-gram Model in PyTorch:** Implement the simple neural network that forms the basis of the model.
5.  **Extract and Visualize the Learned Embeddings:** See how to get the final word vectors from the model's weight matrix and visualize them to see the learned relationships.

---

## Part 1: The Skip-gram Model

**The Task:** The Skip-gram model flips the problem on its head. Instead of predicting a word from its context (which is what CBOW does), Skip-gram **predicts the context from a single target word**.

**The Process:**
1.  We take a large corpus of text.
2.  We slide a window over the text. The word in the center is the **target word**. The words on either side of it within the window are the **context words**.
3.  This creates a set of `(target, context)` pairs. For example, in the sentence "the quick brown fox jumps over the lazy dog" with a window size of 2, one of the training samples would be:
    *   Input (Target Word): `fox`
    *   Outputs (Context Words): `quick`, `brown`, `jumps`, `over`
4.  We then train a simple neural network to take the target word as input and predict the probability of each context word appearing.

**The Key Insight:** The neural network itself is not the goal. The goal is to learn the weights of the **embedding layer** (the hidden layer) of this network. After training, this weight matrix becomes our lookup table of word embeddings. The network forces words with similar contexts to have similar embedding vectors in order to make better predictions.

![Skip-gram Model](https://i.imgur.com/J4g9Y7L.png)

---

## Part 2: Implementing Skip-gram from Scratch

Let's build and train a Skip-gram model on a simple corpus.

### 2.1. Data Preparation

First, we need to tokenize our text, build a vocabulary, and create our `(target, context)` pairs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

print("--- Part 2.1: Data Preparation for Skip-gram ---")

# --- 1. Sample Corpus and Tokenization ---
corpus = "king queen prince princess man woman boy girl strong weak fast slow run walk"
raw_tokens = corpus.split()

# --- 2. Build Vocabulary ---
vocab = {word: i for i, word in enumerate(set(raw_tokens))}
idx_to_word = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)

# --- 3. Generate (target, context) pairs ---
def create_skipgram_dataset(tokens, vocab, window_size=2):
    data = []
    # Convert tokens to indices
    indices = [vocab[word] for word in tokens]
    for i in range(window_size, len(indices) - window_size):
        target_word_idx = indices[i]
        # Get the context words before and after the target
        context_indices = indices[i - window_size : i] + indices[i + 1 : i + window_size + 1]
        for context_word_idx in context_indices:
            data.append([target_word_idx, context_word_idx])
    return data

skipgram_data = create_skipgram_dataset(raw_tokens, vocab, window_size=2)

print(f"Vocabulary Size: {vocab_size}")
print(f"Number of (target, context) pairs: {len(skipgram_data)}")
print(f"Example pair (indices): {skipgram_data[0]}")
print(f"Example pair (words): ['{idx_to_word[skipgram_data[0][0]]}', '{idx_to_word[skipgram_data[0][1]]}']")

# Convert to tensors
inputs = torch.LongTensor([pair[0] for pair in skipgram_data])
labels = torch.LongTensor([pair[1] for pair in skipgram_data])
```

### 2.2. The Skip-gram Model Architecture

The model is surprisingly simple: an embedding layer and a linear layer.

```python
print("\n--- Part 2.2: The Skip-gram Model ---")

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        # The input embedding layer. This is the matrix we want to learn.
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        # The output linear layer.
        self.out_linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word_idx):
        # Get the embedding vector for the input target word
        # Shape: (batch_size, embedding_dim)
        embedded = self.in_embedding(target_word_idx)
        
        # Pass through the linear layer to get scores for all vocab words
        # Shape: (batch_size, vocab_size)
        scores = self.out_linear(embedded)
        return scores

# --- Parameters ---
EMBEDDING_DIM = 10

model = SkipGramModel(vocab_size, EMBEDDING_DIM)
```

### 2.3. The Training Loop

We train the model using `CrossEntropyLoss` to predict the correct context word from the scores.

```python
print("\n--- Part 2.3: Training the Model ---")

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop ---
for epoch in range(100):
    optimizer.zero_grad()
    
    # Get scores for all input words
    scores = model(inputs)
    
    # Calculate the loss
    loss = loss_function(scores, labels)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
```

---

## Part 3: Extracting and Visualizing the Embeddings

After training, the actual output of the model is discarded. The valuable part is the **weight matrix of the input embedding layer**. This matrix contains our learned word vectors.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("\n--- Part 3: Extracting and Visualizing Embeddings ---")

# --- 1. Extract the Embedding Matrix ---
# The learned embeddings are the weights of the `in_embedding` layer.
learned_embeddings = model.in_embedding.weight.detach().cpu().numpy()

print(f"Shape of the learned embedding matrix: {learned_embeddings.shape}")

# --- 2. Visualize with t-SNE ---
# t-SNE is a dimensionality reduction technique that is great for visualizing
# high-dimensional data in 2D or 3D.

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(learned_embeddings)

# --- 3. Plot the results ---
plt.figure(figsize=(15, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Annotate each point with its corresponding word
for i, word in enumerate(idx_to_word.values()):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title('Word2Vec Embeddings Visualized with t-SNE')
plt.grid(True)
plt.show()
```

**Interpreting the Plot:**
In the resulting plot, you should see semantically related words clustered together. For example:
*   `king`, `queen`, `prince`, `princess` should be in one cluster.
*   `man`, `woman`, `boy`, `girl` should be in another.
*   The vector relationship between `king` and `queen` should be similar to the one between `man` and `woman` (representing the gender vector).
*   `strong` and `fast` might be close, and `weak` and `slow` might be close.

## Conclusion: Learning Meaning from Prediction

Word2Vec and the Skip-gram model represent a fundamental shift from count-based vector models to prediction-based ones. By training a simple neural network on the auxiliary task of predicting context words, we can learn dense, meaningful word embeddings that capture rich semantic relationships.

**Key Takeaways:**

1.  **Prediction over Counting:** Word2Vec learns embeddings not by counting, but by training a model to predict context, which forces the embeddings to capture semantic meaning.
2.  **The Model is a Means to an End:** The Skip-gram neural network is just a tool. The final product is the weight matrix of its input embedding layer.
3.  **Vector Arithmetic:** The resulting vector space exhibits impressive linear structures, allowing for analogical reasoning like `king - man + woman = queen`.
4.  **Foundation for Modern NLP:** These pre-trained embeddings became the foundational input layer for almost all downstream NLP tasks for many years, before being integrated into even larger models like BERT.

**Optimization Note (Negative Sampling):**
In practice, calculating the softmax over the entire vocabulary (which can be huge) at every step is very inefficient. The original Word2Vec paper introduced an optimization called **Negative Sampling**, where instead of predicting the context word out of the whole vocabulary, the task is changed to a binary classification: "Is this word a true context word, or a random 'negative' sample?" This is much faster to train and produces equally good embeddings.

## Self-Assessment Questions

1.  **Skip-gram vs. CBOW:** What is the core difference between the prediction task in the Skip-gram model and the CBOW model?
2.  **Training Data:** For the sentence "the cat sat on the mat" and a window size of 1, what are all the `(target, context)` pairs that would be generated?
3.  **The Final Product:** After training a Skip-gram model, what part of the model do we actually keep and use?
4.  **Vector Relationships:** What does the famous `king - man + woman = queen` example demonstrate about the vector space learned by Word2Vec?
5.  **Negative Sampling:** What is the main motivation for using Negative Sampling instead of the full softmax in the output layer?
