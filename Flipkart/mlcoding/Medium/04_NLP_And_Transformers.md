# 🟡 Medium: NLP, Transformers & Text Processing from Scratch
> **Platform:** TensorTonic + Deep-ML | **Difficulty:** Medium | **🔥 Flipkart GenAI Critical**

---

## 🟡 PROBLEM 1: TF-IDF from Scratch

### Theory
**TF-IDF = Term Frequency × Inverse Document Frequency**

$$\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}$$

$$\text{IDF}(t) = \log\left(\frac{N+1}{|\{d: t \in d\}|+1}\right) + 1$$

$$\text{TF-IDF}(t, d) = \text{TF}(t,d) \times \text{IDF}(t)$$

**Intuition:** Words that appear frequently in a document (high TF) but rarely across corpus (high IDF) are more discriminative. Common words like "the", "a" have low IDF → low TF-IDF.

**Smoothing:** The +1 in numerator and denominator prevents log(0) and zero-division.

### Things to Focus On
- ✅ IDF smoothing: `log((N+1)/(df+1)) + 1` (sklearn default)
- ✅ L2 normalize TF-IDF vectors for cosine similarity comparison
- ✅ TF-IDF assumes bag of words — word order doesn't matter
- ✅ For entity matching, character n-gram TF-IDF handles misspellings
- ✅ BM25 (used in Elasticsearch) is a better probabilistic TF-IDF variant

### Implementation
```python
import numpy as np
from collections import Counter
from typing import List, Dict

class TFIDFVectorizer:
    """
    TF-IDF Vectorizer from scratch.
    Matches sklearn's TfidfVectorizer(smooth_idf=True, sublinear_tf=False) behavior.
    """
    
    def __init__(self, max_features: int = None, smooth_idf: bool = True,
                 sublinear_tf: bool = False, ngram_range: tuple = (1, 1)):
        self.max_features = max_features
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.ngram_range = ngram_range
        self.vocabulary_ = None  # Word → index mapping
        self.idf_ = None         # IDF scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split()
    
    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Extract n-grams for ngram_range=(min_n, max_n)."""
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams
    
    def fit(self, corpus: List[str]) -> 'TFIDFVectorizer':
        """Learn vocabulary and IDF weights from corpus."""
        N = len(corpus)
        
        # Build vocabulary (all unique terms in corpus)
        all_terms = set()
        doc_term_sets = []
        for doc in corpus:
            terms = self._get_ngrams(self._tokenize(doc))
            doc_term_sets.append(set(terms))
            all_terms.update(terms)
        
        # Sort vocabulary (for reproducibility)
        vocab = sorted(all_terms)
        
        # Optionally limit to max_features most frequent terms
        if self.max_features is not None:
            # Count document frequency for each term
            df_counter = Counter()
            for term_set in doc_term_sets:
                df_counter.update(term_set)
            vocab = [t for t, _ in df_counter.most_common(self.max_features)]
            vocab = sorted(vocab)
        
        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab)}
        
        # Compute IDF for each term
        df = np.zeros(len(vocab))  # Document frequency
        for term_set in doc_term_sets:
            for term, idx in self.vocabulary_.items():
                if term in term_set:
                    df[idx] += 1
        
        if self.smooth_idf:
            # sklearn formula: log((N+1)/(df+1)) + 1
            self.idf_ = np.log((N + 1) / (df + 1)) + 1
        else:
            # Standard formula: log(N/df) + 1
            self.idf_ = np.log(N / np.maximum(df, 1)) + 1
        
        return self
    
    def transform(self, corpus: List[str]) -> np.ndarray:
        """Transform corpus to TF-IDF matrix. Shape: (n_docs, vocab_size)"""
        V = len(self.vocabulary_)
        matrix = np.zeros((len(corpus), V))
        
        for doc_idx, doc in enumerate(corpus):
            tokens = self._get_ngrams(self._tokenize(doc))
            tf_counts = Counter(tokens)
            total_terms = len(tokens) if len(tokens) > 0 else 1
            
            for term, count in tf_counts.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    tf = count / total_terms  # Term frequency
                    
                    if self.sublinear_tf:
                        tf = 1 + np.log(tf)  # Sublinear TF dampening
                    
                    matrix[doc_idx, term_idx] = tf * self.idf_[term_idx]
        
        # L2 normalize each document vector
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-8)
        
        return matrix
    
    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)

# Test
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats are better than dogs",
    "machine learning and deep learning",
]

tfidf = TFIDFVectorizer()
matrix = tfidf.fit_transform(corpus)
print(f"TF-IDF matrix shape: {matrix.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Find similar documents using cosine similarity
sim = matrix @ matrix.T  # (n_docs, n_docs) — cosine sim after L2 norm
print(f"\nDoc 0 vs Doc 1 similarity (cat/dog): {sim[0,1]:.4f}")
print(f"Doc 0 vs Doc 3 similarity (different topic): {sim[0,3]:.4f}")
```

---

## 🟡 PROBLEM 2: Bag of Words

### Theory
Simplest text representation: count occurrences of each vocabulary word.

Each document → sparse vector of word counts, ignoring order.

$$\text{BoW}[t] = \text{count of term } t \text{ in document}$$

**Limitations:** No word order, no semantics, no handling of synonyms.

### Implementation
```python
def bag_of_words(corpus: List[str]) -> tuple:
    """
    Build BoW matrix from corpus.
    Returns: (matrix, vocabulary)
    """
    # Build vocabulary
    vocab = sorted(set(word.lower() for doc in corpus for word in doc.split()))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # Build matrix
    matrix = np.zeros((len(corpus), len(vocab)), dtype=int)
    for doc_idx, doc in enumerate(corpus):
        for word in doc.lower().split():
            if word in word_to_idx:
                matrix[doc_idx, word_to_idx[word]] += 1
    
    return matrix, vocab

# Test
corpus = ["I love cats", "I love dogs", "dogs are great"]
matrix, vocab = bag_of_words(corpus)
print(f"Vocabulary: {vocab}")
print(f"Matrix:\n{matrix}")
```

---

## 🟡 PROBLEM 3: Positional Encoding

### Theory
Transformers process tokens in parallel (no sequential order). Positional encoding adds position information to embeddings.

**Sinusoidal (Vaswani et al. 2017):**
$$PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$
$$PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

**Properties:**
- Unique encoding for each position
- Relative positions encode relative relationships (PE(pos+k) is linear function of PE(pos))
- Generalizes to sequences longer than training set

**Learned vs Sinusoidal:** BERT uses learned; original transformer uses sinusoidal.

### Things to Focus On
- ✅ Why different frequencies? Low-i dimensions change fast; high-i change slowly → like a clock with multiple hands
- ✅ Division by 10000^(2i/d) creates a geometric progression of wavelengths
- ✅ Added to embeddings (not concatenated): PE ∈ ℝ^d, embedding ∈ ℝ^d → sum ∈ ℝ^d
- ✅ Modern models (RoPE, ALiBi) use rotary/relative position encodings

### Implementation
```python
def positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (Vaswani et al., "Attention is All You Need").
    
    Args:
        max_seq_len: maximum sequence length
        d_model: embedding dimension (must be even)
    Returns:
        PE: shape (max_seq_len, d_model)
    """
    assert d_model % 2 == 0, "d_model must be even for sin/cos pairs"
    
    PE = np.zeros((max_seq_len, d_model))
    
    positions = np.arange(max_seq_len)[:, np.newaxis]   # (seq_len, 1)
    dims      = np.arange(0, d_model, 2)[np.newaxis, :] # (1, d_model/2)
    
    # Compute the angle: pos / 10000^(2i/d)
    angles = positions / (10000 ** (dims / d_model))    # (seq_len, d_model/2)
    
    # Sin for even indices, cos for odd
    PE[:, 0::2] = np.sin(angles)  # Even dimensions
    PE[:, 1::2] = np.cos(angles)  # Odd dimensions
    
    return PE

def add_positional_encoding(embeddings: np.ndarray) -> np.ndarray:
    """Add positional encoding to token embeddings."""
    seq_len, d_model = embeddings.shape
    pe = positional_encoding(seq_len, d_model)
    return embeddings + pe

# Test
PE = positional_encoding(max_seq_len=100, d_model=64)
print(f"PE shape: {PE.shape}")  # (100, 64)
print(f"PE[0, :4] = {PE[0, :4]}")   # Position 0
print(f"PE[1, :4] = {PE[1, :4]}")   # Position 1
print(f"All values in [-1, 1]: {np.all(np.abs(PE) <= 1.0)}")  # True ✓
```

---

## 🟡 PROBLEM 4: Causal Masking (Autoregressive LM)

### Theory
In autoregressive language models (GPT), the model can only attend to previous tokens, not future ones. This prevents "information leakage" during training.

**Causal mask:** Upper triangular matrix of -infinity.

At position i: can attend to positions 0, 1, ..., i (inclusive). Cannot attend to i+1, ..., n-1.

$$\text{Mask}[i][j] = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

When added to attention scores before softmax: softmax(-∞) → 0 → effectively zeroes out future attention.

### Things to Focus On
- ✅ BERT is NOT causal (bidirectional) — uses different masking (random token masking for MLM)
- ✅ GPT-style models ARE causal — attend only to past
- ✅ Implementation: `np.triu(np.ones(...) * -1e9, k=1)` or `torch.triu`
- ✅ k=1 in triu means diagonal is unmasked (tokens attend to themselves)

### Implementation
```python
def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal attention mask for autoregressive models.
    
    Returns: (seq_len, seq_len) mask where:
      - 0 means "can attend"
      - -1e9 means "cannot attend (future position)"
    """
    # Create upper triangular mask (above diagonal)
    # k=1: diagonal is 0 (tokens attend to themselves)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return mask

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: np.ndarray = None) -> np.ndarray:
    """
    Scaled dot-product attention with optional causal mask.
    
    Q, K, V: (seq_len, d_k)
    mask: (seq_len, seq_len) — causal mask or padding mask
    Returns: (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Scaled similarity scores
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Apply mask (add -inf to masked positions)
    if mask is not None:
        scores = scores + mask
    
    # Softmax over key dimension
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # Attention weights
    
    # Weighted sum of values
    return weights @ V

# Test
seq_len, d_k = 5, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

mask = create_causal_mask(seq_len)
print(f"Causal mask:\n{mask.astype(int)}")
# [[  0, -inf, -inf, -inf, -inf],
#  [  0,   0,  -inf, -inf, -inf],
#  [  0,   0,   0,  -inf, -inf],
#  ...

output = scaled_dot_product_attention(Q, K, V, mask)
print(f"Attention output shape: {output.shape}")  # (5, 8)
```

---

## 🟡 PROBLEM 5: Edit Distance (Levenshtein)

### Theory
Minimum number of single-character edits (insertions, deletions, substitutions) to transform string a into string b.

**DP recurrence:**
$$dp[i][j] = \begin{cases}
j & \text{if } i = 0 \\
i & \text{if } j = 0 \\
dp[i-1][j-1] & \text{if } a[i-1] = b[j-1] \\
1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) & \text{otherwise}
\end{cases}$$

**Applications:** Spell checking, DNA alignment, fuzzy entity matching (your resume!), autocomplete.

### Things to Focus On
- ✅ Time: O(mn), Space: O(mn) — can reduce to O(min(m,n)) with rolling array
- ✅ Jaro-Winkler: better for names (honors common prefixes)
- ✅ For Flipkart entity matching: use edit distance + phonetic similarity
- ✅ Normalized edit distance: `edit_dist / max(len(a), len(b))` for comparison

### Implementation
```python
def edit_distance(a: str, b: str) -> int:
    """
    Levenshtein edit distance using dynamic programming.
    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(a), len(b)
    
    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: transform empty string to/from each other
    for i in range(m + 1):
        dp[i][0] = i  # Delete i characters from a
    for j in range(n + 1):
        dp[0][j] = j  # Insert j characters from b
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No edit needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete from a
                    dp[i][j-1],    # Insert into a
                    dp[i-1][j-1]   # Substitute
                )
    
    return dp[m][n]

def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized to [0, 1]."""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    return edit_distance(a, b) / max_len

def jaccard_similarity(a: str, b: str, n: int = 2) -> float:
    """
    Jaccard similarity using character n-grams.
    Better than exact match for fuzzy entity matching.
    """
    def ngrams(s: str, n: int) -> set:
        return {s[i:i+n] for i in range(len(s) - n + 1)}
    
    ngrams_a = ngrams(a.lower(), n)
    ngrams_b = ngrams(b.lower(), n)
    
    if not ngrams_a and not ngrams_b:
        return 1.0
    if not ngrams_a or not ngrams_b:
        return 0.0
    
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union

# Test — entity matching
pairs = [
    ("Flipkart", "Flipkart"),     # Exact
    ("Samsung Galaxy", "Samsung Galxy"),  # Typo
    ("Apple iPhone 14", "Apple iphone 14"),  # Case diff
    ("Sony TV", "LG TV"),         # Different brand
]

for a, b in pairs:
    ed = edit_distance(a, b)
    jacc = jaccard_similarity(a, b)
    print(f"'{a}' vs '{b}': edit_dist={ed}, jaccard={jacc:.3f}")
```

---

## 🟡 PROBLEM 6: Perplexity (LLM Evaluation)

### Theory
Perplexity measures how well a language model predicts a text sequence:

$$\text{PPL}(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$$

**Intuition:** If perplexity = 100, the model is as "confused" as if choosing uniformly among 100 words at each step.

- Low PPL → model assigns high probability to actual text → good model
- PPL = 1: perfect model (always predicts correctly with probability 1)
- PPL = vocab_size: random model

### Things to Focus On
- ✅ Perplexity is exp(cross-entropy loss) for LM evaluation
- ✅ Compare across models only on same test set with same tokenization
- ✅ PPL is not a complete metric: doesn't measure hallucination, factuality
- ✅ GPT-3: ~20 PPL on Penn Treebank; human: ~70 (humans are "less predictable")

### Implementation
```python
def perplexity(log_probs: np.ndarray) -> float:
    """
    Compute perplexity from log-probabilities of each token.
    
    log_probs: array of log P(w_i | w_{<i}) for each token in sequence
    """
    N = len(log_probs)
    if N == 0:
        raise ValueError("Empty log-probability sequence")
    
    avg_neg_log_prob = -np.mean(log_probs)
    return np.exp(avg_neg_log_prob)

def sentence_perplexity(sentence: str, model_log_probs: dict) -> float:
    """
    Compute sentence-level perplexity given token log-probabilities.
    
    sentence: space-separated words
    model_log_probs: {word: log_probability} from language model
    """
    words = sentence.split()
    log_probs = []
    
    for word in words:
        if word in model_log_probs:
            log_probs.append(model_log_probs[word])
        else:
            # OOV: use very low probability (or UNK token)
            log_probs.append(np.log(1e-10))
    
    return perplexity(np.array(log_probs))

# Test
# Simulate token log-probabilities (output from LM for test sequence)
np.random.seed(42)
good_model_log_probs = np.log(np.random.uniform(0.5, 1.0, 100))  # High probs
bad_model_log_probs  = np.log(np.random.uniform(0.01, 0.1, 100))   # Low probs

print(f"Good model PPL: {perplexity(good_model_log_probs):.2f}")   # Low (~2-4)
print(f"Bad model PPL:  {perplexity(bad_model_log_probs):.2f}")    # High (~30+)
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"How does BM25 differ from TF-IDF?"** → BM25 adds term frequency saturation (TF doesn't grow unboundedly) and document length normalization. Mathematically: TF(t,d) → TF(t,d)·(k1+1)/(TF(t,d)+k1·(1-b+b·dl/avgdl)) where k1,b are params.
2. **"Why sinusoidal over learned positional encodings?"** → Generalizes to sequences longer than training (learned can't extrapolate). Same quality in practice. Used when you need to generalize to variable length inputs.
3. **"Causal mask vs padding mask — when to use which?"** → Causal mask: GPT-style generation (block future tokens). Padding mask: block attending to PAD tokens in variable-length batches. LLM decoder uses BOTH.
4. **"Our model's perplexity improved but BLEU dropped. What happened?"** → Model became more fluent but less precise/accurate (generated different valid sentences). Perplexity measures fluency; BLEU measures n-gram overlap with reference. Use both together.
