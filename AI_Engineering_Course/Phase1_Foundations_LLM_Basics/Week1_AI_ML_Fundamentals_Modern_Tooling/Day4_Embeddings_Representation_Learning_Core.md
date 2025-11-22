# Day 4: Embeddings & Representation Learning
## Core Concepts & Theory

### From Discrete Tokens to Continuous Vectors

Neural networks operate on continuous vectors, not discrete symbols. Embeddings are the bridge:

**Token → Embedding → Neural Processing**

```
"hello" → token_id=100 → embedding[100] = [0.2, -0.5, 0.1, ...] → Model
```

### Why Embeddings?

**1. Dimensionality Reduction:**
- Vocabulary: 50,000 tokens
- One-hot encoding: 50,000-dimensional sparse vectors
- Embedding: 512-dimensional dense vectors
- **Compression**: 100× smaller, captures semantics

**2. Semantic Similarity:**
- One-hot: No notion of similarity (all words equidistant)
- Embeddings: Similar words have similar vectors
  - king ≈ queen (both royalty)
  - cat ≈ dog (both animals)

**3. Learned Representations:**
- Not hand-crafted features
- Automatically learned from data
- Capture linguistic properties

### Static Embeddings: Word2Vec

**Skip-Gram Model:**

Predict context words given center word.

```
Sentence: "The quick brown fox jumps"
Center word: "brown"
Context: ["quick", "fox"] (window size = 1)

Objective: Maximize P(quick|brown) × P(fox|brown)
```

**Architecture:**

```python
# Simplified Word2Vec
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, center_word):
        # center_word: (batch,)
        embed = self.embedding(center_word)  # (batch, embed_dim)
        logits = self.output(embed)  # (batch, vocab_size)
        return logits

# Training
model = SkipGram(vocab_size=50000, embed_dim=300)

# For each (center, context) pair:
center = torch.tensor([word_to_id["brown"]])
context = torch.tensor([word_to_id["quick"]])

logits = model(center)
loss = F.cross_entropy(logits, context)
loss.backward()
```

**Negative Sampling:**

Computing softmax over 50K vocabulary is expensive!

**Solution:** Instead of predicting exact context word, distinguish real context from random words:

```python
# Positive sample: (brown, quick) - real context
# Negative samples: (brown, elephant), (brown, computer) - random words

# Train binary classifier: real context vs fake
def negative_sampling_loss(center, positive, negatives):
    center_embed = embedding(center)  # (embed_dim,)
    
    # Positive: should have high similarity
    pos_embed = embedding(positive)
    pos_score = torch.dot(center_embed, pos_embed)
    pos_loss = -F.logsigmoid(pos_score)
    
    # Negatives: should have low similarity
    neg_embeds = embedding(negatives)  # (num_neg, embed_dim)
    neg_scores = torch.matmul(neg_embeds, center_embed)  # (num_neg,)
    neg_loss = -F.logsigmoid(-neg_scores).sum()
    
    return pos_loss + neg_loss
```

**Why This Works:**

Words appearing in similar contexts get similar embeddings:
- "king" and "queen" both appear near "royal", "palace", "crown"
- Their embeddings become similar through gradient descent

**Famous Property: Analogies**

```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
```

This emerges from training, not explicitly programmed!

### CBOW (Continuous Bag of Words)

Inverse of Skip-Gram: Predict center word from context.

```
Context: ["the", "quick", "fox"]
Predict: "brown"
```

**Architecture:**

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, context_words):
        # context_words: (batch, context_size) e.g., (32, 4)
        context_embeds = self.embedding(context_words)  # (batch, context_size, embed_dim)
        # Average context embeddings
        avg_context = context_embeds.mean(dim=1)  # (batch, embed_dim)
        logits = self.output(avg_context)  # (batch, vocab_size)
        return logits
```

**Skip-Gram vs CBOW:**
- **Skip-Gram**: Better for rare words (each word gets multiple training examples)
- **CBOW**: Faster training, better for frequent words

### GloVe (Global Vectors)

Combines matrix factorization with local context:

**Key Insight:** Co-occurrence statistics contain semantic information.

**Co-occurrence Matrix:**

```
        king  queen  man   woman  royal
king    0     30     10    5      25
queen   30    0      5     12     28
man     10    5      0     15     2
woman   5     12     15    0      3
royal   25    28     2     3      0
```

**Objective:**

For words i and j, learn embeddings such that:

```
w_i · w_j + b_i + b_j = log(X_ij)
```

Where X_ij = number of times word j appears in context of word i.

**Loss Function:**

```python
def glove_loss(embeddings, biases, cooccurrence_matrix):
    loss = 0
    for i, j in cooccurrence_matrix:
        x_ij = cooccurrence_matrix[i, j]
        if x_ij == 0:
            continue
        
        # Predicted log co-occurrence
        pred = torch.dot(embeddings[i], embeddings[j]) + biases[i] + biases[j]
        # Actual log co-occurrence
        target = math.log(x_ij)
        
        # Weighted squared error (weight frequent pairs more)
        weight = min(1.0, (x_ij / x_max) ** 0.75)
        loss += weight * (pred - target) ** 2
    
    return loss
```

**Advantages:**
- Faster training (one pass over co-occurrence matrix, not entire corpus)
- Directly optimizes global statistics
- Good performance on analogies and similarity tasks

### FastText: Subword Embeddings

**Problem with Word2Vec/GloVe:**
- Each word is atomic
- "dogs" and "dog" have completely different embeddings
- Can't handle OOV words

**FastText Solution:**

Represent each word as bag of character n-grams.

```
"where" → 
  trigrams: <wh, whe, her, ere, re>
  word itself: <where>

Embedding("where") = avg(emb(<wh), emb(whe), ..., emb(<where>))
```

**Implementation:**

```python
def get_ngrams(word, n=3):
    word = f"<{word}>"  # Add boundaries
    return [word[i:i+n] for i in range(len(word)-n+1)]

class FastText(nn.Module):
    def __init__(self, vocab_size, ngram_vocab_size, embed_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.ngram_embeddings = nn.Embedding(ngram_vocab_size, embed_dim)
    
    def forward(self, word_ids, ngram_ids_list):
        # word_ids: (batch,)
        # ngram_ids_list: list of (batch, variable_length)
        
        word_embeds = self.word_embeddings(word_ids)  # (batch, embed_dim)
        
        # Average n-gram embeddings
        ngram_embeds = []
        for ngrams in ngram_ids_list:
            ngram_embed = self.ngram_embeddings(ngrams).mean(dim=1)
            ngram_embeds.append(ngram_embed)
        
        # Combine
        final_embed = word_embeds + sum(ngram_embeds) / len(ngram_embeds)
        return final_embed
```

**Advantages:**
- Handles morphology: "running" ≈ "run" (share n-grams)
- OOV words: Can embed unseen words from n-grams
- Rare words: Better representations (share subwords with common words)

### From Static to Contextual Embeddings

**Problem with Static Embeddings:**

"bank" has same embedding in:
- "river bank" (geological feature)
- "savings bank" (financial institution)

**Solution: Contextual Embeddings**

Different embedding for same word depending on context!

### ELMo (Embeddings from Language Models)

**Key Idea:** Use bi-directional LSTM language model, extract hidden states as embeddings.

**Architecture:**

```python
class ELMo(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Forward LSTM (predict next word)
        self.forward_lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True
        )
        
        # Backward LSTM (predict previous word)
        self.backward_lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True
        )
    
    def forward(self, tokens):
        # tokens: (batch, seq_len)
        embeds = self.embedding(tokens)
        
        # Forward pass
        forward_out, _ = self.forward_lstm(embeds)
        
        # Backward pass (reverse sequence)
        backward_out, _ = self.backward_lstm(embeds.flip(dims=[1]))
        backward_out = backward_out.flip(dims=[1])
        
        # Concatenate forward and backward
        contextualized = torch.cat([forward_out, backward_out], dim=-1)
        
        return contextualized  # (batch, seq_len, 2*hidden_dim)
```

**Usage:**

```python
# Pre-train ELMo on large corpus
elmo = ELMo(vocab_size=50000, embed_dim=512, hidden_dim=1024)
# Train to predict next/previous words

# For downstream task, extract contextual embeddings:
sentence = "I went to the bank"
tokens = tokenizer(sentence)
contextual_embeds = elmo(tokens)  # Different for each occurrence of "bank"

# Use as features for downstream task
outputs = task_model(contextual_embeds)
```

**Advantages:**
- Context-dependent
- Captures syntax and semantics
- Transfer learning: Pre-train once, use for many tasks

### Modern Contextual Embeddings: BERT

BERT uses Transformer encoder (attention mechanism) instead of LSTM.

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "I went to the bank"
inputs = tokenizer(sentence, return_tensors='pt')

# Get contextualized embeddings
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # (1, seq_len, 768)

# Each token has 768-dim contextualized embedding
# "bank" has different embedding than in "savings bank" sentence
```

### Embedding Dimensions: Trade-offs

**Common Sizes:**
- Word2Vec/GloVe: 100-300 dimensions
- FastText: 300 dimensions
- ELMo: 512-1024 dimensions
- BERT-base: 768 dimensions
- BERT-large: 1024 dimensions
- GPT-3: 12,288 dimensions (!)

**Trade-offs:**

**Smaller (100-300):**
- ✓ Faster computation
- ✓ Less memory
- ✓ Less overfitting on small data
- ✗ Less capacity (can't capture complex semantics)

**Larger (768-12,288):**
- ✓ More capacity (captures subtle semantics)
- ✓ Better performance on complex tasks
- ✗ Slower, more memory
- ✗ Requires more data to train

**Rule of Thumb:**
- Task complexity ∝ Embedding dimension
- More training data → Can support larger embeddings
- Modern LLMs: 4096-12,288 dimensions

### Embedding Initialization

**Random Initialization:**
```python
embedding = nn.Embedding(vocab_size, embed_dim)
# Default: Uniform(-1/sqrt(embed_dim), 1/sqrt(embed_dim))
```

**Pre-trained Initialization:**
```python
# Load Word2Vec embeddings
pretrained_embeddings = load_word2vec()  # (vocab_size, 300)

embedding = nn.Embedding(vocab_size, 300)
embedding.weight.data = torch.tensor(pretrained_embeddings)

# Option 1: Freeze (don't update during training)
embedding.weight.requires_grad = False

# Option 2: Fine-tune (update during training)
embedding.weight.requires_grad = True
```

**When to use pre-trained:**
- Small training dataset
- Domain similar to pre-training corpus
- Faster convergence

**When to train from scratch:**
- Large training dataset
- Domain-specific vocabulary (medical, legal, code)
- Different language

### Positional Embeddings

For Transformers, position information is crucial (no inherent order).

**Learned Positional Embeddings:**
```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, tokens):
        # tokens: (batch, seq_len)
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(tokens)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine
        return token_embeds + pos_embeds
```

**Sinusoidal Positional Encodings (Original Transformer):**
```python
def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

**Advantages of sinusoidal:**
- Deterministic (no learned parameters)
- Extrapolates to longer sequences (not limited to max_len)

### Summary

Embeddings transform discrete tokens into continuous vectors:

**Static Embeddings:**
- **Word2Vec (Skip-Gram/CBOW)**: Local context, analogies, 2013
- **GloVe**: Global co-occurrence, efficient, 2014
- **FastText**: Subword-aware, handles OOV, 2016

**Contextual Embeddings:**
- **ELMo**: Bi-LSTM, context-dependent, 2018
- **BERT**: Transformer encoder, state-of-the-art, 2018+

**Key Concepts:**
- Dimensionality reduction (sparse one-hot → dense embedding)
- Semantic similarity (similar words ≈ similar vectors)
- Transfer learning (pre-train, fine-tune)
- Context-dependence (modern embeddings adapt to context)

Modern LLMs build on these foundations, using massive embedding matrices (50K vocab × 4096 dim = 200M parameters just for embeddings!).
