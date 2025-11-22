# Day 3: Tokenization & Text Processing
## Deep Dive - Internal Mechanics & Advanced Reasoning

### BPE Algorithm: Step-by-Step Internals

**Initial Setup:**

Given corpus: "low low low lowest"

Step 1: Character-level split with word boundaries
```
Vocabulary: ['l', 'o', 'w', '_', 'e', 's', 't']
Corpus representation: ['l', 'o', 'w', '_', 'l', 'o', 'w', '_', 'l', 'o', 'w', '_', 'l', 'o', 'w', 'e', 's', 't']
```

Step 2: Count all adjacent pairs
```python
from collections import Counter

def count_pairs(tokens):
    pairs = Counter()
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i+1])] += 1
    return pairs

# Result:
# ('l', 'o'): 4
# ('o', 'w'): 4
# ('w', '_'): 3
# ('w', 'e'): 1
# ('e', 's'): 1
# ('s', 't'): 1
```

Step 3: Merge most frequent pair
```
Most frequent: ('l', 'o') with count 4
Create new token: 'lo'
Update vocabulary: ['l', 'o', 'w', '_', 'e', 's', 't', 'lo']
Update corpus: ['lo', 'w', '_', 'lo', 'w', '_', 'lo', 'w', '_', 'lo', 'w', 'e', 's', 't']
```

**Iteration continues:**
- Next merge: ('lo', 'w') → 'low'
- Then: ('low', '_') → 'low_'
- Eventually: entire frequent words become single tokens

**Implementation Details:**

```python
def byte_pair_encoding(corpus, vocab_size):
    # Start with character vocabulary
    vocab = set(char for word in corpus for char in word)
    vocab.add('</w>')  # End-of-word token
    
    # Add space to represent word boundaries
    words = [list(word) + ['</w>'] for word in corpus.split()]
    
    merges = []
    while len(vocab) < vocab_size:
        pairs = count_all_pairs(words)
        if not pairs:
            break
        
        best = max(pairs, key=pairs.get)
        merge_pair(words, best)
        vocab.add(''.join(best))
        merges.append(best)
    
    return vocab, merges

def apply_bpe(text, merges):
    \"\"\"Apply learned BPE merges to new text.\"\"\"
    tokens = list(text) + ['</w>']
    for merge in merges:
        tokens = apply_merge(tokens, merge)
    return tokens
```

**Why This Works:**

1. **Zipf's Law**: Word frequencies follow power law
   - A few words are very frequent → quickly merge into tokens
   - Rare words stay as subwords → compositional representation

2. **Greedy is Good Enough**: 
   - Optimal tokenization is NP-hard
   - Greedy (always merge most frequent) works well in practice
   - Order of merges matters! (low + er vs lo + wer)

### WordPiece: Likelihood-Based Merging

**Key Difference from BPE:**

Instead of raw frequency, use likelihood ratio:

```
score(pair) = freq(pair) / (freq(first) × freq(second))
```

**Intuition:**

- High score: pair occurs together MORE than expected by chance
- Measures "mutual information" between subwords

**Example:**

```
"un" appears 1000 times
"happy" appears 500 times
"unhappy" appears 400 times

BPE score: freq("un", "happy") = 400
WordPiece score: 400 / (1000 × 500) = 0.0008

"happy" appears 500 times
"ness" appears 300 times
"happiness" appears 250 times

WordPiece score: 250 / (500 × 300) = 0.00167 (higher!)
```

"happiness" gets merged before "unhappy" despite lower raw frequency.

**Why?**

- "un-" appears in many words (unlock, undo, unhappy) → lower score
- "happiness" is more "cohesive" → higher score

**Implementation:**

```python
def wordpiece_merge_score(pair_freq, first_freq, second_freq):
    return pair_freq / (first_freq * second_freq)

def wordpiece_training(corpus, vocab_size):
    vocab = init_char_vocab(corpus)
    
    while len(vocab) < vocab_size:
        pairs = count_pairs_with_context(corpus, vocab)
        scores = {}
        
        for pair, freq in pairs.items():
            first_freq = count_token_freq(pair[0], corpus, vocab)
            second_freq = count_token_freq(pair[1], corpus, vocab)
            scores[pair] = wordpiece_merge_score(freq, first_freq, second_freq)
        
        best_pair = max(scores, key=scores.get)
        merge(best_pair, corpus, vocab)
    
    return vocab
```

### SentencePiece: Unigram Language Model

**Fundamentally Different Approach:**

BPE/WordPiece: Bottom-up (start small, merge)
SentencePiece: Top-down (start large, prune)

**Unigram LM Tokenization:**

Given vocabulary V, tokenize text by finding segmentation that maximizes likelihood:

```
P(x₁, x₂, ..., x_n | V) = ∏ P(x_i | V)
```

Find segmentation maximizing this probability using Viterbi algorithm.

**Training Algorithm:**

1. Initialize with huge vocabulary (all substrings up to length N)
2. Iteratively:
   a. For each token in vocab, calculate loss if removed
   b. Remove X% of tokens with smallest loss increase
3. Repeat until vocab_size reached

**Why This Works:**

- Keeps tokens that reduce perplexity most
- Optimal tokenization given the Unigram assumption
- More theoretically grounded than greedy BPE

**Viterbi Decoding:**

```python
def viterbi_tokenize(text, vocab, probs):
    \"\"\"Find optimal tokenization using dynamic programming.\"\"\"
    n = len(text)
    # best_score[i] = best log-prob for text[:i]
    best_score = [-float('inf')] * (n + 1)
    best_score[0] = 0
    backpointer = [None] * (n + 1)
    
    for i in range(1, n + 1):
        for token in vocab:
            if text[i-len(token):i] == token:
                score = best_score[i-len(token)] + math.log(probs[token])
                if score > best_score[i]:
                    best_score[i] = score
                    backpointer[i] = token
    
    # Reconstruct path
    tokens = []
    i = n
    while i > 0:
        token = backpointer[i]
        tokens.append(token)
        i -= len(token)
    
    return list(reversed(tokens))
```

**Advantages over BPE:**

- No pre-tokenization (language-agnostic)
- Probabilistic (multiple segmentations possible)
- Reversible (can exactly reconstruct original text including spaces)

### Byte-Level BPE (GPT-2)

**Problem with Standard BPE:**

- Requires pre-defined character set
- Large Unicode range (150K+ characters) → huge base vocabulary
- OOV characters break tokenization

**GPT-2 Solution: Byte-Level BPE**

Operate on UTF-8 bytes (256 possible values) instead of characters.

```python
def bytes_to_unicode():
    \"\"\"Map bytes to printable Unicode characters.\"\"\"
    bs = list(range(ord(\"!\"), ord(\"~\")+1)) + \
         list(range(ord(\"¡\"), ord(\"¬\")+1)) + \
         list(range(ord(\"®\"), ord(\"ÿ\")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}
```

**Advantages:**

- Base vocabulary = 256 (small!)
- Can represent ANY text (any Unicode, any language, even binary)
- No UNK token needed (everything is representable)

**Trade-off:**

- Longer sequences (each non-ASCII char = multiple bytes)
- Example: "日本" (2 chars) → 6 bytes → 6 tokens minimum

### Tokenization and Model Performance

**Impact on Training:**

1. **Sequence Length:**
   - More tokens → longer sequences → more memory
   - LLaMA 32K vocab vs BERT 30K: Similar lengths despite different algorithms

2. **Embedding Matrix Size:**
   - vocab_size × d_model parameters
   - 50K vocab × 4096 dim = 200M parameters just for embeddings!
   - Trade-off: Larger vocab = larger embedding matrix

3. **Softmax Computation:**
   - Output layer: logits = hidden @ embedding.T
   - vocab_size = 50K: compute 50K logits per token
   - Expensive! (Mitigated by efficient matmul)

**Impact on Inference:**

```python
# Tokenization affects generation speed
# Longer tokenized sequences = more forward passes

# Example: Generating "Hello world"
# GPT-2 tokenizer: ["Hello", " world"] = 2 tokens
# Character-level: ["H","e","l","l","o"," ","w","o","r","l","d"] = 11 tokens

# Generation time ∝ number of tokens!
```

### Advanced Tokenization Challenges

**1. Tokenization-Model Co-Dependency:**

Model learns representations specific to tokenization.

Example:
- If "COVID" is one token → single embedding
- If "CO", "VID" two tokens → composition of two embeddings

Can't swap tokenizers without retraining!

**2. Subword Regularization:**

During training, randomly use different segmentations:

```
"unbelievable" could be tokenized as:
- ["un", "believable"]
- ["unbeliev", "able"]
- ["un", "be", "liev", "able"]
```

**Benefits:**
- Acts as data augmentation
- Makes model robust to tokenization variations
- Used in mT5, XLM-R

**3. Tokenization and Arithmetic:**

LLMs struggle with arithmetic partially due to tokenization:

```
"123 + 456 = 579"

Tokenized as: ["123", " +", " ", "456", " =", " ", "579"]

Problem: "123" is atomic token, model doesn't see "1", "2", "3" separately
Hard to learn place value, carrying, etc.
```

**Attempted Solutions:**
- Character-level for numbers
- Special number tokens
- Tool use (calculator)

**4. Tokenization Bias and Fairness:**

```python
# Analyze tokenization equity
def analyze_tokenization_bias(tokenizer, word_list):
    for word in word_list:
        tokens = tokenizer.tokenize(word)
        print(f"{word}: {len(tokens)} tokens - {tokens}")

# Example findings:
# English names: 1-2 tokens average
# African names: 3-5 tokens average
# Asian names: 4-8 tokens average

# More tokens → more parameters to learn → harder to learn → worse performance!
```

**Mitigation:**
- Multilingual training data
- Balanced vocabulary (ensure coverage of all groups)
- Post-hoc analysis and adjustment

### Summary

Tokenization internals reveal deep design trade-offs:

- **BPE**: Greedy frequency-based, simple, effective
- **WordPiece**: Likelihood-based, slightly better for rare words
- **SentencePiece/Unigram**: Probabilistic, language-agnostic, modern choice
- **Byte-level**: Universal coverage, no UNK, but longer sequences

Key insights:
- Tokenization is learned from data (affects all downstream behavior)
- No "correct" tokenization - trade-offs between sequence length, vocab size, fairness
- Modern LLMs use 32K-50K vocab, mostly SentencePiece
- Byte-level ensures universal coverage (GPT-2, GPT-3)
- Tokenization bias is a real fairness concern

Understanding tokenization internals is crucial for:
- Debugging model behavior
- Choosing/creating tokenizers for new domains
- Understanding model limitations (arithmetic, multilingual performance)
