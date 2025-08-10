# Day 14.4: PyTorch Implementation and Usage - Practical Word Embeddings with PyTorch

## Overview

PyTorch implementation of word embeddings represents the practical application of distributional semantics and neural language modeling theories through efficient, scalable, and flexible deep learning frameworks that enable researchers and practitioners to build sophisticated natural language processing systems. PyTorch's embedding layers, optimization utilities, and tensor operations provide comprehensive tools for implementing classical models like Word2Vec and GloVe as well as modern contextualized embeddings, while supporting advanced techniques including custom tokenization, subword modeling, transfer learning, and fine-tuning for downstream tasks. This comprehensive exploration covers the complete pipeline from data preprocessing and vocabulary construction through model implementation, training strategies, evaluation methodologies, and deployment considerations, providing practical knowledge for building production-ready embedding systems that can handle large-scale text corpora and support diverse NLP applications.

## PyTorch Embedding Fundamentals

### nn.Embedding Layer Architecture

**Basic Embedding Layer**
```python
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
    
    def forward(self, input_ids):
        return self.embedding(input_ids)
```

**Mathematical Operation**
$$\mathbf{E} \in \mathbb{R}^{|V| \times d}$$
$$\mathbf{x}_i = \mathbf{E}[w_i] \quad \text{(lookup operation)}$$

**Embedding Parameters**
- **num_embeddings**: Vocabulary size $|V|$
- **embedding_dim**: Embedding dimension $d$
- **padding_idx**: Index for padding tokens (weights frozen at zero)
- **max_norm**: Maximum L2 norm for embedding vectors
- **norm_type**: Norm type for renormalization
- **scale_grad_by_freq**: Scale gradients by word frequency

**Advanced Initialization**
```python
def initialize_embeddings(embedding_layer, method='xavier_uniform'):
    if method == 'xavier_uniform':
        nn.init.xavier_uniform_(embedding_layer.weight)
    elif method == 'normal':
        nn.init.normal_(embedding_layer.weight, mean=0, std=0.1)
    elif method == 'uniform':
        nn.init.uniform_(embedding_layer.weight, -0.1, 0.1)
    elif method == 'kaiming':
        nn.init.kaiming_normal_(embedding_layer.weight)
    
    # Zero out padding token
    if embedding_layer.padding_idx is not None:
        embedding_layer.weight.data[embedding_layer.padding_idx].fill_(0)
```

### Embedding Bag for Efficient Aggregation

**EmbeddingBag Architecture**
```python
class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, mode='mean'):
        super(EmbeddingBagModel, self).__init__()
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            mode=mode  # 'sum', 'mean', or 'max'
        )
    
    def forward(self, input_ids, offsets=None):
        return self.embedding_bag(input_ids, offsets)
```

**Aggregation Operations**
- **Sum**: $\mathbf{h} = \sum_{i=1}^{n} \mathbf{E}[w_i]$
- **Mean**: $\mathbf{h} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{E}[w_i]$
- **Max**: $\mathbf{h} = \max_{i=1}^{n} \mathbf{E}[w_i]$

**Offset-based Batching**
```python
# Example: Two documents with different lengths
input_ids = torch.tensor([1, 2, 3, 4, 5, 6])  # Concatenated word IDs
offsets = torch.tensor([0, 3])  # Document boundaries
# Doc 1: [1, 2, 3], Doc 2: [4, 5, 6]

embeddings = embedding_bag(input_ids, offsets)
# Returns: [embedding_doc1, embedding_doc2]
```

## Skip-gram Implementation

### Complete Skip-gram Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, Counter

class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=5, negative_samples=5):
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.window_size = window_size
        self.negative_samples = negative_samples
        
        # Prepare training pairs
        self.prepare_data(corpus)
        self.prepare_negative_sampling()
    
    def prepare_data(self, corpus):
        self.training_pairs = []
        
        for sentence in corpus:
            sentence_ids = [self.word_to_idx[word] for word in sentence 
                          if word in self.word_to_idx]
            
            for i, center_word in enumerate(sentence_ids):
                # Dynamic window size
                window = np.random.randint(1, self.window_size + 1)
                
                for j in range(max(0, i - window), 
                              min(len(sentence_ids), i + window + 1)):
                    if i != j:
                        self.training_pairs.append((center_word, sentence_ids[j]))
    
    def prepare_negative_sampling(self):
        # Unigram distribution with 3/4 power
        word_counts = Counter()
        for pair in self.training_pairs:
            word_counts[pair[0]] += 1
        
        # Create sampling table
        vocab_size = len(self.vocab)
        sampling_table = np.zeros(int(1e8), dtype=np.int32)
        
        total_count = sum(word_counts.values())
        cumulative_prob = 0
        word_idx = 0
        
        for i in range(len(sampling_table)):
            sampling_table[i] = word_idx
            if i / len(sampling_table) > cumulative_prob:
                word_idx += 1
                if word_idx < vocab_size:
                    prob = (word_counts.get(word_idx, 0) ** 0.75) / total_count
                    cumulative_prob += prob
        
        self.sampling_table = sampling_table
    
    def get_negative_samples(self, positive_word, k):
        negative_samples = []
        while len(negative_samples) < k:
            neg_word = self.sampling_table[np.random.randint(0, len(self.sampling_table))]
            if neg_word != positive_word:
                negative_samples.append(neg_word)
        return negative_samples
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        center_word, context_word = self.training_pairs[idx]
        negative_samples = self.get_negative_samples(context_word, self.negative_samples)
        
        return {
            'center': torch.tensor(center_word, dtype=torch.long),
            'positive': torch.tensor(context_word, dtype=torch.long),
            'negatives': torch.tensor(negative_samples, dtype=torch.long)
        }

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        
        # Two embedding matrices as in Word2Vec
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.init_embeddings()
    
    def init_embeddings(self):
        init_range = 0.5 / self.center_embeddings.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, center_words, positive_words, negative_words):
        # Get embeddings
        center_embeds = self.center_embeddings(center_words)  # [batch, embed_dim]
        positive_embeds = self.context_embeddings(positive_words)  # [batch, embed_dim]
        negative_embeds = self.context_embeddings(negative_words)  # [batch, neg_samples, embed_dim]
        
        # Positive score: center · positive
        positive_score = torch.sum(center_embeds * positive_embeds, dim=1)  # [batch]
        
        # Negative scores: center · negatives
        center_expanded = center_embeds.unsqueeze(1)  # [batch, 1, embed_dim]
        negative_scores = torch.bmm(negative_embeds, center_expanded.transpose(1, 2))
        negative_scores = negative_scores.squeeze(2)  # [batch, neg_samples]
        
        return positive_score, negative_scores
    
    def get_embeddings(self):
        """Return final word embeddings (average of center and context)"""
        center_weights = self.center_embeddings.weight.data
        context_weights = self.context_embeddings.weight.data
        return (center_weights + context_weights) / 2
```

### Skip-gram Loss Function

```python
class SkipGramLoss(nn.Module):
    def __init__(self):
        super(SkipGramLoss, self).__init__()
    
    def forward(self, positive_scores, negative_scores):
        # Positive loss: -log(sigmoid(positive_score))
        positive_loss = -F.logsigmoid(positive_scores).mean()
        
        # Negative loss: -log(sigmoid(-negative_scores))
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=1).mean()
        
        return positive_loss + negative_loss

def train_skipgram(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            center = batch['center'].to(device)
            positive = batch['positive'].to(device)
            negatives = batch['negatives'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pos_scores, neg_scores = model(center, positive, negatives)
            
            # Calculate loss
            loss = criterion(pos_scores, neg_scores)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 1000 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
```

## CBOW Implementation

### CBOW Model Architecture

```python
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOWModel, self).__init__()
        
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range = 0.5 / self.embeddings.embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.fill_(0)
    
    def forward(self, context_words):
        # context_words: [batch_size, context_size]
        embeds = self.embeddings(context_words)  # [batch_size, context_size, embed_dim]
        
        # Average context embeddings
        context_embed = torch.mean(embeds, dim=1)  # [batch_size, embed_dim]
        
        # Predict center word
        output = self.linear(context_embed)  # [batch_size, vocab_size]
        
        return output

class CBOWDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.context_size = context_size
        self.training_data = self.prepare_data(corpus)
    
    def prepare_data(self, corpus):
        training_data = []
        
        for sentence in corpus:
            sentence_ids = [self.word_to_idx[word] for word in sentence 
                          if word in self.word_to_idx]
            
            for i in range(self.context_size, len(sentence_ids) - self.context_size):
                context = (sentence_ids[i-self.context_size:i] + 
                          sentence_ids[i+1:i+self.context_size+1])
                target = sentence_ids[i]
                training_data.append((context, target))
        
        return training_data
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        context, target = self.training_data[idx]
        return {
            'context': torch.tensor(context, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

def train_cbow(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(context)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
```

## GloVe Implementation

### GloVe Co-occurrence Matrix Construction

```python
import scipy.sparse as sp
from collections import defaultdict
import pickle

class GloVeCooccurrence:
    def __init__(self, vocab, window_size=10, min_count=5):
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.window_size = window_size
        self.min_count = min_count
    
    def build_cooccurrence_matrix(self, corpus):
        """Build co-occurrence matrix from corpus"""
        cooccurrence = defaultdict(float)
        
        for sentence in corpus:
            sentence_ids = [self.word_to_idx[word] for word in sentence 
                          if word in self.word_to_idx]
            
            for i, center_word in enumerate(sentence_ids):
                for j in range(max(0, i - self.window_size),
                              min(len(sentence_ids), i + self.window_size + 1)):
                    if i != j:
                        context_word = sentence_ids[j]
                        distance = abs(i - j)
                        weight = 1.0 / distance  # Distance weighting
                        
                        # Symmetric co-occurrence
                        cooccurrence[(center_word, context_word)] += weight
                        cooccurrence[(context_word, center_word)] += weight
        
        # Filter by minimum count
        filtered_cooccurrence = {pair: count for pair, count in cooccurrence.items() 
                               if count >= self.min_count}
        
        return self.create_sparse_matrix(filtered_cooccurrence)
    
    def create_sparse_matrix(self, cooccurrence):
        """Convert to sparse matrix format"""
        vocab_size = len(self.vocab)
        row_indices = []
        col_indices = []
        values = []
        
        for (i, j), value in cooccurrence.items():
            row_indices.append(i)
            col_indices.append(j)
            values.append(value)
        
        matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                              shape=(vocab_size, vocab_size))
        return matrix.tocsr()

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, x_max=100, alpha=0.75):
        super(GloVeModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        
        # Word and context embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Bias terms
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize embeddings uniformly
        init_range = 0.5 / self.embedding_dim
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        
        # Initialize biases to zero
        self.word_biases.weight.data.fill_(0)
        self.context_biases.weight.data.fill_(0)
    
    def weighting_function(self, x):
        """GloVe weighting function"""
        return torch.where(x < self.x_max, 
                          (x / self.x_max) ** self.alpha, 
                          torch.ones_like(x))
    
    def forward(self, word_indices, context_indices, cooccurrence_values):
        # Get embeddings and biases
        word_embeds = self.word_embeddings(word_indices)
        context_embeds = self.context_embeddings(context_indices)
        word_bias = self.word_biases(word_indices).squeeze()
        context_bias = self.context_biases(context_indices).squeeze()
        
        # Compute dot products
        dot_product = torch.sum(word_embeds * context_embeds, dim=1)
        
        # Compute predictions
        predictions = dot_product + word_bias + context_bias
        
        # Compute weights
        weights = self.weighting_function(cooccurrence_values)
        
        # Compute loss
        loss = weights * (predictions - torch.log(cooccurrence_values)) ** 2
        
        return loss.mean()
    
    def get_embeddings(self):
        """Return final embeddings (average of word and context)"""
        word_weights = self.word_embeddings.weight.data
        context_weights = self.context_embeddings.weight.data
        return (word_weights + context_weights) / 2

def train_glove(model, cooccurrence_matrix, optimizer, device, epochs=50):
    model.train()
    
    # Convert sparse matrix to coordinate format
    coo_matrix = cooccurrence_matrix.tocoo()
    word_indices = torch.tensor(coo_matrix.row, dtype=torch.long, device=device)
    context_indices = torch.tensor(coo_matrix.col, dtype=torch.long, device=device)
    cooccurrence_values = torch.tensor(coo_matrix.data, dtype=torch.float, device=device)
    
    # Create batches
    batch_size = 10000
    num_batches = (len(word_indices) + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(word_indices))
        word_indices = word_indices[perm]
        context_indices = context_indices[perm]
        cooccurrence_values = cooccurrence_values[perm]
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(word_indices))
            
            batch_words = word_indices[start_idx:end_idx]
            batch_contexts = context_indices[start_idx:end_idx]
            batch_values = cooccurrence_values[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(batch_words, batch_contexts, batch_values)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
```

## FastText Implementation

### Subword-aware FastText Model

```python
import re
from typing import List, Set

class FastTextTokenizer:
    def __init__(self, min_n=3, max_n=6, bucket_size=2000000):
        self.min_n = min_n
        self.max_n = max_n
        self.bucket_size = bucket_size
    
    def get_word_ngrams(self, word: str) -> List[str]:
        """Extract character n-grams from word"""
        word = f"<{word}>"  # Add boundary markers
        ngrams = []
        
        for n in range(self.min_n, min(len(word), self.max_n) + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        
        return ngrams
    
    def hash_ngram(self, ngram: str) -> int:
        """Hash n-gram to bucket index"""
        h = 0
        for char in ngram:
            h = h * 31 + ord(char)
        return h % self.bucket_size

class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, bucket_size=2000000, 
                 min_n=3, max_n=6):
        super(FastTextModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bucket_size = bucket_size
        self.tokenizer = FastTextTokenizer(min_n, max_n, bucket_size)
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Subword embeddings
        self.subword_embeddings = nn.Embedding(bucket_size, embedding_dim)
        
        self.init_weights()
    
    def init_weights(self):
        init_range = 0.5 / self.embedding_dim
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.subword_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def get_word_vector(self, word_idx, word_text):
        """Get representation for a word (word + subword features)"""
        # Word embedding
        word_embed = self.word_embeddings(word_idx)
        
        # Subword embeddings
        ngrams = self.tokenizer.get_word_ngrams(word_text)
        subword_indices = [self.tokenizer.hash_ngram(ngram) for ngram in ngrams]
        
        if subword_indices:
            subword_indices_tensor = torch.tensor(subword_indices, dtype=torch.long, 
                                                device=word_embed.device)
            subword_embeds = self.subword_embeddings(subword_indices_tensor)
            subword_embed = torch.mean(subword_embeds, dim=0)
            
            # Combine word and subword embeddings
            return word_embed + subword_embed
        else:
            return word_embed
    
    def forward(self, center_words, context_words, center_texts, negative_words=None):
        """Forward pass for Skip-gram with subwords"""
        batch_size = center_words.size(0)
        
        # Get center word representations
        center_embeds = []
        for i in range(batch_size):
            center_embed = self.get_word_vector(center_words[i], center_texts[i])
            center_embeds.append(center_embed)
        center_embeds = torch.stack(center_embeds)
        
        # Context embeddings (using standard embedding lookup)
        context_embeds = self.word_embeddings(context_words)
        
        # Positive scores
        positive_scores = torch.sum(center_embeds * context_embeds, dim=1)
        
        if negative_words is not None:
            # Negative embeddings
            negative_embeds = self.word_embeddings(negative_words)
            center_expanded = center_embeds.unsqueeze(1)
            negative_scores = torch.bmm(negative_embeds, center_expanded.transpose(1, 2))
            negative_scores = negative_scores.squeeze(2)
            return positive_scores, negative_scores
        
        return positive_scores

def create_fasttext_vocabulary(corpus, min_count=5):
    """Create vocabulary with word frequencies"""
    word_freq = Counter()
    
    for sentence in corpus:
        for word in sentence:
            word_freq[word] += 1
    
    # Filter by minimum count
    vocab = [word for word, freq in word_freq.items() if freq >= min_count]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    return vocab, word_to_idx, word_freq
```

## Advanced PyTorch Features

### Custom Embedding Layers

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length=5000):
        super(PositionalEmbedding, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_length, embedding_dim)
        
        self.LayerNorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, 
                                      device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = word_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class ConditionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_conditions):
        super(ConditionalEmbedding, self).__init__()
        
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.condition_embeddings = nn.Embedding(num_conditions, embedding_dim)
        
        # Learned combination weights
        self.combine_layer = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, input_ids, condition_ids):
        base_embeds = self.base_embeddings(input_ids)
        condition_embeds = self.condition_embeddings(condition_ids)
        
        # Expand condition embeddings to match input shape
        if len(condition_embeds.shape) < len(base_embeds.shape):
            condition_embeds = condition_embeds.unsqueeze(1).expand_as(base_embeds)
        
        # Combine embeddings
        combined = torch.cat([base_embeds, condition_embeds], dim=-1)
        output = torch.tanh(self.combine_layer(combined))
        
        return output
```

### Embedding Regularization

```python
class RegularizedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0.1, 
                 weight_dropout=0.1, alpha=1e-6):
        super(RegularizedEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = weight_dropout
        self.alpha = alpha  # L2 regularization
        
    def forward(self, input_ids):
        # Apply weight dropout during training
        if self.training and self.weight_dropout > 0:
            mask = torch.rand_like(self.embedding.weight) > self.weight_dropout
            masked_weight = self.embedding.weight * mask.float()
            embeds = F.embedding(input_ids, masked_weight)
        else:
            embeds = self.embedding(input_ids)
        
        embeds = self.dropout(embeds)
        return embeds
    
    def regularization_loss(self):
        """L2 regularization loss"""
        return self.alpha * torch.norm(self.embedding.weight, p=2)

class VariationalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim=None):
        super(VariationalEmbedding, self).__init__()
        
        if latent_dim is None:
            latent_dim = embedding_dim // 2
        
        self.latent_dim = latent_dim
        
        # Encoder for mean and log-variance
        self.mu_embedding = nn.Embedding(vocab_size, latent_dim)
        self.logvar_embedding = nn.Embedding(vocab_size, latent_dim)
        
        # Decoder
        self.decoder = nn.Linear(latent_dim, embedding_dim)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, input_ids):
        mu = self.mu_embedding(input_ids)
        logvar = self.logvar_embedding(input_ids)
        
        z = self.reparameterize(mu, logvar)
        embeds = self.decoder(z)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return embeds, kl_loss
```

## Pre-trained Embeddings Integration

### Loading and Using Pre-trained Embeddings

```python
import numpy as np
import torch

class PretrainedEmbeddingLoader:
    def __init__(self, embedding_path, vocab=None, embedding_dim=300):
        self.embedding_path = embedding_path
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.word_to_idx = None
        self.embeddings = None
    
    def load_word2vec_binary(self, limit=None):
        """Load Word2Vec binary format"""
        from gensim.models import KeyedVectors
        
        model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
        
        if limit:
            vocab = list(model.vocab.keys())[:limit]
        else:
            vocab = list(model.vocab.keys())
        
        embeddings = np.array([model[word] for word in vocab])
        
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.embeddings = torch.FloatTensor(embeddings)
        
        return self.embeddings
    
    def load_glove_text(self, limit=None):
        """Load GloVe text format"""
        embeddings = []
        vocab = []
        
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                parts = line.rstrip().split(' ')
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                
                vocab.append(word)
                embeddings.append(vector)
        
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.embeddings = torch.FloatTensor(embeddings)
        
        return self.embeddings
    
    def create_embedding_layer(self, vocab_size=None, freeze=False):
        """Create PyTorch embedding layer with pre-trained weights"""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_* method first.")
        
        if vocab_size is None:
            vocab_size = len(self.vocab)
        
        embedding_layer = nn.Embedding(vocab_size, self.embedding_dim)
        
        # Initialize with pre-trained weights
        with torch.no_grad():
            embedding_layer.weight[:len(self.vocab)] = self.embeddings
            
            # Initialize OOV words with random vectors
            if vocab_size > len(self.vocab):
                nn.init.normal_(embedding_layer.weight[len(self.vocab):], 
                              mean=0, std=0.1)
        
        if freeze:
            embedding_layer.weight.requires_grad = False
        
        return embedding_layer

class AdaptiveEmbedding(nn.Module):
    def __init__(self, pretrained_embeddings, vocab_size, embedding_dim,
                 adaptation_dim=50):
        super(AdaptiveEmbedding, self).__init__()
        
        # Frozen pre-trained embeddings
        self.pretrained = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=True
        )
        
        # Learnable adaptation layer
        self.adaptation = nn.Embedding(vocab_size, adaptation_dim)
        self.projection = nn.Linear(embedding_dim + adaptation_dim, embedding_dim)
        
    def forward(self, input_ids):
        pretrained_embeds = self.pretrained(input_ids)
        adaptive_embeds = self.adaptation(input_ids)
        
        combined = torch.cat([pretrained_embeds, adaptive_embeds], dim=-1)
        output = self.projection(combined)
        
        return output
```

## Evaluation and Analysis Tools

### Embedding Similarity and Analogy

```python
class EmbeddingEvaluator:
    def __init__(self, embeddings, vocab, word_to_idx):
        self.embeddings = embeddings  # [vocab_size, embed_dim]
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    def get_word_vector(self, word):
        """Get vector for a word"""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        else:
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    def most_similar(self, word, top_k=10):
        """Find most similar words"""
        word_vec = self.get_word_vector(word)
        if word_vec is None:
            return []
        
        # Compute similarities with all words
        similarities = F.cosine_similarity(word_vec.unsqueeze(0), self.embeddings)
        
        # Get top-k most similar (excluding the word itself)
        _, indices = similarities.topk(top_k + 1)
        
        results = []
        for idx in indices:
            if idx.item() != self.word_to_idx[word]:
                sim_word = self.idx_to_word[idx.item()]
                similarity = similarities[idx].item()
                results.append((sim_word, similarity))
                
                if len(results) == top_k:
                    break
        
        return results
    
    def word_analogy(self, a, b, c, top_k=1):
        """Solve word analogy: a is to b as c is to ?"""
        vec_a = self.get_word_vector(a)
        vec_b = self.get_word_vector(b)
        vec_c = self.get_word_vector(c)
        
        if any(vec is None for vec in [vec_a, vec_b, vec_c]):
            return []
        
        # Compute target vector: b - a + c
        target_vec = vec_b - vec_a + vec_c
        
        # Find most similar words to target
        similarities = F.cosine_similarity(target_vec.unsqueeze(0), self.embeddings)
        
        # Exclude input words
        exclude_indices = [self.word_to_idx[word] for word in [a, b, c] 
                          if word in self.word_to_idx]
        for idx in exclude_indices:
            similarities[idx] = -float('inf')
        
        _, indices = similarities.topk(top_k)
        
        results = []
        for idx in indices:
            word = self.idx_to_word[idx.item()]
            similarity = similarities[idx].item()
            results.append((word, similarity))
        
        return results
    
    def evaluate_similarity_dataset(self, dataset_path):
        """Evaluate on word similarity dataset"""
        human_scores = []
        model_scores = []
        
        with open(dataset_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                word1, word2, human_score = parts[0], parts[1], float(parts[2])
                
                vec1 = self.get_word_vector(word1)
                vec2 = self.get_word_vector(word2)
                
                if vec1 is not None and vec2 is not None:
                    model_score = self.cosine_similarity(vec1, vec2)
                    human_scores.append(human_score)
                    model_scores.append(model_score)
        
        # Compute correlation
        correlation = np.corrcoef(human_scores, model_scores)[0, 1]
        return correlation, len(human_scores)

def visualize_embeddings_2d(embeddings, vocab, words_to_plot=None, method='tsne'):
    """Visualize embeddings in 2D using t-SNE or PCA"""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Select words to plot
    if words_to_plot is None:
        words_to_plot = vocab[:100]  # Plot first 100 words
    
    word_indices = [i for i, word in enumerate(vocab) if word in words_to_plot]
    selected_embeddings = embeddings[word_indices].cpu().numpy()
    selected_words = [vocab[i] for i in word_indices]
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    for i, word in enumerate(selected_words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title(f'Word Embeddings Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
```

## Complete Training Pipeline

### End-to-End Training Example

```python
def main():
    # Configuration
    config = {
        'vocab_size': 10000,
        'embedding_dim': 300,
        'window_size': 5,
        'negative_samples': 5,
        'min_count': 5,
        'batch_size': 1024,
        'learning_rate': 0.001,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Load and preprocess data
    print("Loading corpus...")
    corpus = load_corpus("path/to/corpus.txt")  # Implement based on your data
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab, word_to_idx, word_freq = build_vocabulary(corpus, config['min_count'])
    config['vocab_size'] = len(vocab)
    
    print(f"Vocabulary size: {config['vocab_size']}")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = SkipGramDataset(corpus, vocab, config['window_size'], 
                             config['negative_samples'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                          shuffle=True, num_workers=4)
    
    # Initialize model
    print("Initializing model...")
    model = SkipGramModel(config['vocab_size'], config['embedding_dim'])
    model.to(config['device'])
    
    # Loss and optimizer
    criterion = SkipGramLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training
    print("Starting training...")
    train_skipgram(model, dataloader, optimizer, criterion, 
                   config['device'], config['epochs'])
    
    # Extract final embeddings
    print("Extracting embeddings...")
    final_embeddings = model.get_embeddings()
    
    # Save embeddings
    torch.save({
        'embeddings': final_embeddings,
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'config': config
    }, 'embeddings.pt')
    
    # Evaluation
    print("Evaluating embeddings...")
    evaluator = EmbeddingEvaluator(final_embeddings, vocab, word_to_idx)
    
    # Test similarity
    test_words = ['king', 'queen', 'man', 'woman']
    for word in test_words:
        if word in word_to_idx:
            similar = evaluator.most_similar(word, 5)
            print(f"Most similar to '{word}': {similar}")
    
    # Test analogy
    analogies = [('king', 'queen', 'man'), ('france', 'paris', 'germany')]
    for a, b, c in analogies:
        result = evaluator.word_analogy(a, b, c, 1)
        if result:
            print(f"{a} : {b} :: {c} : {result[0][0]} ({result[0][1]:.3f})")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
```

## Key Questions for Review

### Implementation Fundamentals
1. **Embedding Layer Usage**: How do PyTorch embedding layers differ from manual matrix multiplication approaches?

2. **Memory Efficiency**: What are the trade-offs between different embedding storage and lookup strategies?

3. **Gradient Flow**: How does gradient flow work through embedding layers during backpropagation?

### Model Architecture
4. **Skip-gram vs CBOW**: How do the PyTorch implementations of Skip-gram and CBOW differ in computational complexity?

5. **Negative Sampling**: What are the implementation considerations for efficient negative sampling in PyTorch?

6. **Subword Models**: How does FastText's subword approach affect memory usage and training time?

### Training Optimization
7. **Batch Processing**: How should batching be handled for variable-length sequences in embedding training?

8. **Learning Rate Scheduling**: What learning rate schedules work best for different embedding models?

9. **Regularization**: When and how should dropout and weight decay be applied to embedding layers?

### Advanced Features
10. **Custom Embeddings**: How can specialized embedding architectures be implemented for domain-specific tasks?

11. **Pre-trained Integration**: What are the best practices for fine-tuning pre-trained embeddings?

12. **Multi-GPU Training**: How can embedding training be scaled across multiple GPUs effectively?

## Conclusion

PyTorch implementation of word embeddings provides comprehensive tools and frameworks for building sophisticated natural language processing systems that can learn distributed semantic representations from large text corpora and support diverse downstream applications. This comprehensive exploration has established:

**Implementation Mastery**: Deep understanding of PyTorch's embedding layers, optimization utilities, and tensor operations demonstrates how modern deep learning frameworks enable efficient implementation of classical and advanced embedding methods with high-performance computational backends.

**Architecture Flexibility**: Systematic coverage of Skip-gram, CBOW, GloVe, and FastText implementations shows how different theoretical approaches to distributional semantics can be translated into practical PyTorch modules with appropriate architectural choices and training procedures.

**Advanced Techniques**: Integration of subword modeling, pre-trained embedding loading, custom embedding layers, and regularization methods provides tools for handling complex linguistic phenomena and improving embedding quality for specific applications and domains.

**Training Optimization**: Comprehensive treatment of negative sampling, batch processing, learning rate scheduling, and parallel training demonstrates how to efficiently train embedding models on large-scale corpora while maintaining numerical stability and convergence properties.

**Evaluation Frameworks**: Implementation of similarity testing, analogy solving, and visualization tools provides practical methods for assessing embedding quality and understanding the geometric properties of learned representations.

**Production Deployment**: Coverage of model serialization, inference optimization, and integration strategies shows how embedding systems can be deployed in production environments and integrated with larger NLP pipelines.

PyTorch implementation of word embeddings is crucial for modern NLP because:
- **Framework Integration**: Seamless integration with other PyTorch-based NLP models and training pipelines
- **Performance Optimization**: Access to optimized CUDA kernels and automatic differentiation for efficient training
- **Research Flexibility**: Easy prototyping and experimentation with novel embedding architectures and training procedures  
- **Production Readiness**: Robust deployment tools and optimization features for real-world applications
- **Community Support**: Extensive ecosystem of pre-trained models, datasets, and implementation examples

The implementation techniques and practical knowledge covered provide essential skills for building embedding-based NLP systems using PyTorch. Understanding these approaches is fundamental for developing modern language models, implementing transfer learning systems, and creating specialized embeddings for domain-specific applications that require sophisticated semantic understanding capabilities.