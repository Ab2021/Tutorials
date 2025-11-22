# Day 4: Embeddings & Representation Learning
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Word2Vec: Mathematical Framework

**Skip-Gram Objective (Formal):**

Given corpus of T words w₁, w₂, ..., w_T, maximize log-likelihood:

```
L = (1/T) Σ_t Σ_{-c≤j≤c, j≠0} log P(w_{t+j} | w_t)
```

Where:
- w_t = center word at position t
- w_{t+j} = context word at offset j
- c = context window size

**Softmax Formulation:**

```
P(w_o | w_i) = exp(v'_{w_o} · v_{w_i}) / Σ_{w=1}^V exp(v'_w · v_{w_i})
```

Where:
- v_{w_i} = input embedding for word w_i
- v'_{w_o} = output embedding for word w_o  
- V = vocabulary size

**Why Two Embedding Matrices?**

Word2Vec maintains two matrices:
- Input embeddings W_in: (vocab_size,embed_dim)
- Output embeddings W_out: (vocab_size, embed_dim)

**Reasoning:**

1. **Asymmetric Roles:** Words play different roles as center vs context
2. **Optimization:** Easier to learn (separate parameters)
3. **Final Embeddings:** Usually take W_in or (W_in + W_out)/2

**Negative Sampling: Detailed Derivation**

Softmax over V=50K words requires computing 50K exponentials per training example!

**Negative Sampling transforms the problem:**

Instead of multi-class classification (predict which context word):
→ Binary classification (is this a real context word?)

**Objective:**

```
log σ(v'_{w_o} · v_{w_i}) + Σ_{i=1}^k E_{w_i ~ P_n(w)} [log σ(-v'_{w_i} · v_{w_i})]
```

Where:
- σ(x) = sigmoid  = 1 / (1 + exp(-x))
- k = number of negative samples (typically 5-20)
- P_n(w) = noise distribution (usually P(w)^{3/4})

**Why P(w)^{3/4}?**

Raises probability of rare words:

```
If P("the") = 0.1, P("zephyr") = 0.0001

Uniform sampling: "the" sampled 1000× more
P(w)^{3/4}:
  P("the")^{3/4} = 0.1^{0.75} ≈ 0.178
  P("zephyr")^{3/4} = 0.0001^{0.75} ≈ 0.0018
  
Ratio: only 100× difference (better balance)
```

**Implementation:**

```python
class Word2VecNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize with small values
        self.input_embeddings.weight.data.uniform_(-0.5/embed_dim, 0.5/embed_dim)
        self.output_embeddings.weight.data.zero_()
    
    def forward(self, center_words, context_words, negative_samples):
        # center_words: (batch,)
        # context_words: (batch,)
        # negative_samples: (batch, num_neg)
        
        center_embeds = self.input_embeddings(center_words)  # (batch, embed_dim)
        context_embeds = self.output_embeddings(context_words)  # (batch, embed_dim)
        neg_embeds = self.output_embeddings(negative_samples)  # (batch, num_neg, embed_dim)
        
        # Positive score
        pos_score = (center_embeds * context_embeds).sum(dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score).mean()
        
        # Negative scores
        neg_scores = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze()  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_scores).sum(dim=1).mean()
        
        return pos_loss + neg_loss
```

**Subsampling Frequent Words:**

Very frequent words ("the", "a") provide less information.

**Solution:** Subsample frequent words with probability:

```
P(discard w_i) = 1 - sqrt(t / f(w_i))
```

Where:
- f(w_i) = frequency of word w_i
- t = threshold (typically 10^{-5})

Example:
```
f("the") = 0.1 → P(discard) = 1 - sqrt(10^{-5}/0.1) = 0.99 (discard 99%)
f("zephyr") = 0.0001 → P(discard) = 1 - sqrt(10^{-5}/0.0001) = 0.68 (discard 68%)
```

Balances training signal!

### GloVe: Matrix Factorization View

**Co-occurrence Matrix X:**

X_ij = number of times word j appears in context of word i

**Insight:** X contains all information for embeddings!

**GloVe Objective:**

```
J = Σ_{i,j} f(X_ij) (w_i^T w_j + b_i + b_j - log X_ij)^2
```

Where f(x) is weighting function:

```python
def weighting_function(x, x_max=100, alpha=0.75):
    if x < x_max:
        return (x / x_max) ** alpha
    else:
        return 1.0
```

**Why This Weighting?**

1. **Rare pairs (X_ij small):** Low weight (don't let noise dominate)
2. **Very frequent pairs (X_ij large):** Cap weight (diminishing returns)
3. **alpha=0.75:** Empirically optimal

**Connection to Word2Vec:**

GloVe can be seen as factorizing log(X):

```
log(X) ≈ W · W^T
```

Where W = embeddings matrix.

Both Word2Vec and GloVe are implicitly factorizing PMI (Pointwise Mutual Information) matrix!

**PMI:**

```
PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) × P(w_j)))
```

High PMI = words co-occur more than expected by chance.

### FastText: Subword Embeddings Deep Dive

**N-gram Hashing:**

With trigrams for vocab of V words:
- Potential trigrams: 26^3 = 17,576 (just lowercase English)
- With punctuation, capitals, Unicode: millions!

**Solution:** Hash n-grams to fixed size (e.g., 2M buckets)

```python
import hashlib

def hash_ngram(ngram, num_buckets=2000000):
    hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
    return hash_val % num_buckets

# Example
hash_ngram("<wh") → 184753
hash_ngram("whe") → 1847261
```

**Full FastText Embedding:**

```python
class FastTextEmbedding(nn.Module):
    def __init__(self, vocab_size, ngram_buckets, embed_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.ngram_embeddings = nn.Embedding(ngram_buckets, embed_dim)
        self.ngram_buckets = ngram_buckets
    
    def get_ngram_ids(self, word, n=3):
        word = f"<{word}>"
        ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
        ngram_ids = [hash_ngram(ng, self.ngram_buckets) for ng in ngrams]
        return ngram_ids
    
    def forward(self, word, word_id):
        # Word embedding
        word_embed = self.word_embeddings(torch.tensor([word_id]))
        
        # N-gram embeddings
        ngram_ids = self.get_ngram_ids(word)
        ngram_ids_tensor = torch.tensor(ngram_ids)
        ngram_embeds = self.ngram_embeddings(ngram_ids_tensor)
        ngram_avg = ngram_embeds.mean(dim=0)
        
        # Combine
        final_embed = word_embed + ngram_avg
        return final_embed
```

**OOV Word Embedding:**

For unseen word "chatgpt":
```
Ngrams: <ch, cha, hat, atg, tgp, gpt, pt>
Embedding = avg(emb(<ch), emb(cha), ..., emb(pt))
```

Even though "chatgpt" wasn't in training, can embed it!

**Trade-off:**
- More parameters (vocab + ngram buckets)
- Slower inference (need to compute ngrams)
- But: handles morphology and OOV

### Contextual Embeddings: The Paradigm Shift

**Static Embedding Problem:**

One vector per word type (regardless of context).

```python
embed("bank") = [0.2, -0.1, ...] # Same everywhere!

# Leads to averaging of senses:
# embed("bank") ≈ 0.5 * embed("river bank") + 0.5 * embed("savings bank")
# Neither semantic!
```

**Contextual Solution:**

```python
embed("bank", context="I went to the bank to deposit money") = [0.5, 0.3, ...]
embed("bank", context="We sat by the river bank") = [-0.2, 0.1, ...]
# Different vectors!
```

**ELMo Architecture Deep Dive:**

2-layer bi-directional LSTM.

**Forward LSTM:**

```
h_t^f = LSTM_f(h_{t-1}^f, x_t)
```

Predicts next word: P(w_{t+1} | w_1, ..., w_t)

**Backward LSTM:**

```
h_t^b = LSTM_b(h_{t+1}^b, x_t)
```

Predicts previous word: P(w_{t-1} | w_t, ..., w_T)

**Contextualized Embedding:**

```
ELMo_k^{task} = γ^{task} Σ_{j=0}^L s_j^{task} h_{k,j}
```

Where:
- h_{k,j} = hidden state at layer j, position k
- s_j = learned softmax-normalized weights (task-specific)
- γ = scaling factor

**Why Weighted Sum of All Layers?**

Different layers capture different information:
- Layer 0 (embedding): Syntactic (POS tagging)
- Layer 1: Semantic+syntax (NER)
- Layer 2: High-level semantics (QA, entailment)

Task-specific weighting lets model choose relevant layers!

**Training Objective:**

Joint forward + backward language modeling:

```
L = Σ_t [log P_f(w_t | w_1, ..., w_{t-1}) + log P_b(w_t | w_{t+1}, ..., w_T)]
```

**Problem with ELMo:**

LSTMs are sequential → can't parallelize across sequence.

**Solution:** Transformers (BERT) use attention → fully parallelizable!

### BERT Embeddings: Layer-wise Representations

BERT has 12 layers (base) or 24 layers (large).

**Each layer produces different representations:**

```python
from transformers import BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
inputs = tokenizer("The bank is closed", return_tensors='pt')

outputs = model(**inputs)
hidden_states = outputs.hidden_states  # Tuple of 13 tensors (embedding + 12 layers)

# Extract layer-specific embeddings
layer_0 = hidden_states[0]  # Input embeddings
layer_6 = hidden_states[6]  # Middle layer
layer_12 = hidden_states[12]  # Final layer

# Different layers for different tasks:
# - Layers 1-4: Surface, syntax (POS tagging)
# - Layers 5-8: Syntax+semantics (parsing, NER)
# - Layers 9-12: Semantics (entailment, QA)
```

**Probing Studies Show:**

BERT layers are hierarchical:
1. **Lower layers**: Encode syntactic information (phrase structure)
2. **Middle layers**: Encode semantic roles (subject, object)
3. **Upper layers**: Task-specific information

**Anisotropy Problem:**

BERT embeddings suffer from anisotropy: occupy narrow cone in embedding space (not uniformly distributed).

**Consequence:** All embeddings are somewhat similar (high cosine similarity even for unrelated words).

**Solution: BERT-flow, SimCSE**
- Post-processing to spread embeddings uniformly
- Contrastive learning objectives

### Embedding Arithmetic: Why It Works

**Famous Example:**

```
king - man + woman ≈ queen
```

**Explanation:**

Embeddings capture relationships as vector offsets:

```
v(king) - v(man) ≈ "royal male" - "male" = "royal"
v(woman) + "royal" ≈ v(queen)
```

**Formalization:**

If training objective encourages:
- Similar contexts → similar embeddings

Then:
- "king" and "queen" both appear near "royal", "throne"
- "man" and "king" both appear in male contexts
- "woman" and "queen" both appear in female contexts

**Vector algebra emerges:**

```
v(king) = v(royal) + v(male) + noise
v(queen) = v(royal) + v(female) + noise
v(man) = v(male) + noise'
v(woman) = v(female) + noise'

v(king) - v(man) + v(woman) ≈ v(royal) + v(female) ≈ v(queen)
```

**Limitations:**

Works for prototypical relationships, fails for:
- Multiple senses: "left" (direction) vs "left" (departed)
- Cultural bias: "doctor" + "female" doesn't reliably give "doctor" (bias in data)
- Rare words: Noisy embeddings

### Summary: Evolution of Embeddings

**2013:** Word2Vec - Revolutionary (local context, negative sampling)

**2014:** GloVe - Global statistics, competitive with Word2Vec

**2016:** FastText - Subword n-grams, handles OOV

**2018:** ELMo - Contextualized with bi-LSTM, task-specific weighting

**2018:** BERT - Contextualized with Transformers, pre-train + fine-tune paradigm

**2020+:** Modern LLMs - Massive embeddings (GPT-3: 12,288-dim), frozen or LoRA-adapted

**Key Insight:**

Embeddings went from:
- One vector per word → One vector per word-in-context
- Static → Dynamic
- Isolated training → Transfer learning

Modern LLMs have 200M-1B parameters just in embeddings (vocab × dim)!
