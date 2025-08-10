# Day 14.4: PyTorch Implementation & Usage - A Practical Guide

## Introduction: From Theory to Application

We have explored the theory behind various word embedding techniques, from classic count-based models to modern contextual representations. Now, it's time to focus on the practical application within a PyTorch workflow. How do we efficiently load these embeddings, integrate them into our models, and use them effectively during training?

This guide provides a hands-on, practical walkthrough of using embedding layers in PyTorch, with a focus on loading pre-trained embeddings and the common patterns for using them in downstream tasks.

**Today's Learning Objectives:**

1.  **Master the `nn.Embedding` Layer:** Understand its parameters and how it functions as a simple lookup table.
2.  **Learn to Initialize with Pre-trained Embeddings:** Implement the full workflow for loading pre-trained GloVe or Word2Vec vectors into your `nn.Embedding` layer's weight matrix.
3.  **Understand the "Freezing" vs. "Fine-tuning" Trade-off:** Learn how to control whether the embedding layer is updated during training (`requires_grad`) and understand the implications of each choice.
4.  **Build a Complete Text Classification Model:** Integrate a pre-trained embedding layer into a full sentiment analysis model and train it.

---

## Part 1: The `nn.Embedding` Layer - A Deep Dive

The `nn.Embedding` layer is the heart of text processing in PyTorch. It is, at its core, a simple lookup table.

*   **Functionality:** It stores a weight matrix of shape `(num_embeddings, embedding_dim)`.
*   **Input:** It takes a tensor of integer indices as input (e.g., `[batch_size, seq_len]`).
*   **Output:** For each input index, it looks up the corresponding row in the weight matrix. The output is a dense tensor of shape `(batch_size, seq_len, embedding_dim)`.

### 1.1. A Simple Example

```python
import torch
import torch.nn as nn

print("--- Part 1: The nn.Embedding Layer ---")

# --- Parameters ---
vocab_size = 10 # Our vocabulary has 10 unique words
embedding_dim = 4 # We want to represent each word with a 4-dimensional vector

# --- Create the layer ---
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

print(f"Embedding layer weight matrix shape: {embedding_layer.weight.shape}")

# --- Create some input data ---
# A batch of 2 sentences, with 5 word indices each.
# The indices must be less than vocab_size.
input_indices = torch.LongTensor([[1, 5, 2, 8, 9], [4, 3, 2, 1, 0]])

# --- Get the embeddings ---
embedded_output = embedding_layer(input_indices)

print(f"\nInput indices shape: {input_indices.shape}")
print(f"Output embeddings shape: {embedded_output.shape}")

# --- Let's verify the lookup ---
# The embedding for the first word of the first sentence (index 1)
first_word_embedding = embedded_output[0, 0, :]
# The corresponding row in the weight matrix
first_word_from_weight = embedding_layer.weight[1]

print(f"\nEmbedding for index 1 from output: {first_word_embedding}")
print(f"Row 1 from weight matrix:      {first_word_from_weight}")
print(f"Are they the same? {torch.allclose(first_word_embedding, first_word_from_weight)}")
```

### 1.2. The `padding_idx` Argument

This is a very useful argument. When you provide a `padding_idx`, the embedding vector at that index will always be a vector of zeros, and crucially, **it will not be updated during training** (its gradient will always be zero). This is the correct way to handle padded sequences.

```python
# Create an embedding layer where index 0 is for padding
pad_idx = 0
embedding_layer_with_pad = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

# The vector at the padding index is initialized to zeros
print(f"\nVector at padding index {pad_idx}: {embedding_layer_with_pad.weight[pad_idx]}")
```

---

## Part 2: Loading Pre-trained GloVe Embeddings

This is the most common and effective way to use embeddings. Instead of starting with a random weight matrix, we initialize it with powerful, pre-trained vectors like GloVe.

**The Workflow:**
1.  Build the vocabulary for **your specific dataset**.
2.  Create an `nn.Embedding` layer with the correct dimensions.
3.  Create a new weight matrix of the same shape, initialized with zeros or random values.
4.  Load the pre-trained GloVe vectors (e.g., using `torchtext.vocab`).
5.  Iterate through your own vocabulary. For each word, find its corresponding vector in the GloVe vocabulary and copy it into your new weight matrix.
6.  Use `.load_state_dict()` or `.copy_()` to load your custom weight matrix into the `nn.Embedding` layer.

```python
import torchtext.vocab as vocab

print("\n--- Part 2: Loading Pre-trained Embeddings ---")

# --- 1. Load GloVe ---
glove = vocab.GloVe(name='6B', dim=50) # Using 50-dim for speed

# --- 2. Our Custom Vocabulary and Embedding Layer ---
# This vocab would be built from our training data.
my_vocab = {'<unk>': 0, '<pad>': 1, 'the': 2, 'cat': 3, 'sat': 4, 'on': 5, 'a': 6, 'mat': 7, 'pytorch': 8}
my_vocab_size = len(my_vocab)
embedding_dim = 50
pad_idx = my_vocab['<pad>']

my_embedding_layer = nn.Embedding(my_vocab_size, embedding_dim, padding_idx=pad_idx)

# --- 3. Build the Weight Matrix ---
# Create a matrix to hold the vectors
pretrained_weights = torch.zeros(my_vocab_size, embedding_dim)

# Keep track of how many words we found
words_found = 0

for word, idx in my_vocab.items():
    try:
        # Get the vector from GloVe
        pretrained_weights[idx] = glove[word]
        words_found += 1
    except KeyError:
        # For words not in GloVe (like <unk>, <pad>, or rare words),
        # we leave them as random initialization.
        pretrained_weights[idx] = torch.randn(embedding_dim)

print(f"Found {words_found}/{my_vocab_size} words in the GloVe vocabulary.")

# --- 4. Load the weights into the layer ---
my_embedding_layer.weight.data.copy_(pretrained_weights)

print("Successfully loaded pre-trained weights.")

# --- Let's check the vector for 'cat' ---
cat_idx = torch.LongTensor([my_vocab['cat']])
cat_vec_from_layer = my_embedding_layer(cat_idx)
cat_vec_from_glove = glove['cat']

print(f"Are the 'cat' vectors from our layer and GloVe the same? {torch.allclose(cat_vec_from_layer.squeeze(), cat_vec_from_glove)}")
```

---

## Part 3: To Freeze or To Fine-Tune?

Once you have loaded the pre-trained embeddings, you have a critical choice to make: do you update these embeddings during training?

This is controlled by the `requires_grad` attribute of the embedding layer's weight tensor.

### 3.1. Strategy 1: Freezing the Embeddings

*   **How:** `embedding_layer.weight.requires_grad = False`
*   **What it does:** The embedding vectors will **not** be updated during backpropagation. They are treated as fixed features.
*   **When to use it:**
    *   When your downstream dataset is **very small**. Fine-tuning on a small dataset risks distorting the powerful pre-trained embeddings and overfitting to your specific data.
    *   When you want to train faster.

### 3.2. Strategy 2: Fine-Tuning the Embeddings

*   **How:** `embedding_layer.weight.requires_grad = True` (This is the default).
*   **What it does:** The embedding vectors will be updated by the optimizer just like any other parameter in your model. They will be slowly adjusted to become more specialized for your specific task and domain.
*   **When to use it:**
    *   When your downstream dataset is **large enough**.
    *   When the domain of your task is very different from the domain the embeddings were trained on (e.g., using news-based GloVe vectors for a medical text task).

**Best Practice:** A common and effective strategy is to start with frozen embeddings and train the rest of your model. Then, as a second step, you can "unfreeze" the embedding layer and fine-tune the entire model with a very low learning rate.

--- 

## Part 4: Full Text Classification Example

Let's put it all together in a sentiment analysis model that uses a pre-trained, frozen embedding layer.

```python
print("\n--- Part 4: Full Classification Example ---")

# We will reuse the model and data loading from Day 12.2, but modify the model
# to accept a pre-trained weight matrix.

class SentimentClassifierWithPretrained(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx, pretrained_weights):
        super().__init__()
        # --- The Key Difference ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Load the pre-trained weights
        self.embedding.weight.data.copy_(pretrained_weights)
        # Freeze the embedding layer
        self.embedding.weight.requires_grad = False
        # ------------------------
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# --- Dummy Usage ---
# Assume `my_vocab_size` and `pretrained_weights` are defined as in Part 2.
model = SentimentClassifierWithPretrained(
    vocab_size=my_vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=256,
    output_dim=1,
    n_layers=2,
    bidirectional=True,
    dropout=0.5,
    pad_idx=pad_idx,
    pretrained_weights=pretrained_weights
)

print("Model with frozen pre-trained embeddings created successfully.")

# Check that the embedding layer is indeed frozen
for name, param in model.named_parameters():
    if 'embedding' in name:
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
```

## Conclusion

The `nn.Embedding` layer is your gateway to processing text in PyTorch. While you can train it from scratch, initializing it with pre-trained embeddings like GloVe or Word2Vec provides your model with a massive head start, injecting a rich understanding of language semantics before it has even seen a single sample from your dataset.

**Key Takeaways:**

1.  **`nn.Embedding` is a Lookup Table:** It maps integer indices to dense vectors.
2.  **Loading Pre-trained Weights is Standard Practice:** The workflow involves building a weight matrix that matches your custom vocabulary and then loading it into the embedding layer.
3.  **Freeze or Fine-tune:** The decision to freeze or fine-tune the embeddings is a trade-off between preserving the powerful pre-trained knowledge and specializing the embeddings for your task. Freezing is safer for small datasets.
4.  **Use `padding_idx`:** This is the correct way to handle padding, as it ensures the padding token's vector is always zero and does not get updated during training.

Mastering these practical techniques is essential for building high-performing NLP models in PyTorch.

## Self-Assessment Questions

1.  **`nn.Embedding` I/O:** If you pass a tensor of shape `[16, 20]` to an `nn.Embedding` layer with `embedding_dim=100`, what will be the shape of the output tensor?
2.  **Initialization:** What is the main advantage of initializing an `nn.Embedding` layer with GloVe vectors instead of random values?
3.  **Out-of-Vocabulary (OOV):** In our workflow for loading pre-trained weights, what happens if a word from our custom vocabulary does not exist in the GloVe vocabulary?
4.  **Freezing:** What is the single line of code used to "freeze" an embedding layer?
5.  **Use Case:** You are building a sentiment classifier for Twitter data. You have a very large dataset of 1 million tweets. Would you choose to freeze or fine-tune your GloVe embedding layer? Why?

