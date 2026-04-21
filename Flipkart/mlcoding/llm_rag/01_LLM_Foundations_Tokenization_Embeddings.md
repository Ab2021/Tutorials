# 🤖 LLM & RAG Coding — Part 1: Foundations
> **Difficulty:** Easy → Medium | **Focus:** Tokens, Embeddings, Attention Math, Generation
> **Flipkart Relevance:** 🔥🔥🔥 — GenAI, Catalog Intelligence, Search, ChatBot

---

## HOW LLMs WORK — MENTAL MODEL (Learn this cold)

```
Input Text
    │
    ▼
[Tokenizer]         ← Text → Token IDs (BPE/WordPiece)
    │
    ▼
[Embedding Layer]   ← Token IDs → Dense vectors (vocab_size × d_model)
    │
    ▼
[+ Positional Enc.] ← Add position information to each token
    │
    ▼
[Transformer Block] × N layers
  ├─ Layer Norm
  ├─ Multi-Head Self-Attention
  ├─ Residual Connection
  ├─ Layer Norm
  ├─ Feed-Forward (FFN)
  └─ Residual Connection
    │
    ▼
[Language Model Head] ← Linear + Softmax → Probability over vocab
    │
    ▼
[Sampling/Decoding] ← Greedy / Top-K / Top-P / Beam Search
    │
    ▼
Output Token → Append → Repeat (autoregressive)
```

---

## 🟢 PROBLEM 1: Tokenization from Scratch — Character & Word Level

### Theory
**Tokenization** converts raw text to integer IDs that a model can process.

**Types:**
- **Character-level:** Each character is a token. Vocabulary ~100. Handles any word but slow (long sequences).
- **Word-level:** Each word is a token. Vocabulary ~50k+. Fast but fails on unseen words (OOV).
- **Subword (BPE/WordPiece):** Best of both — common words get own token; rare words split into subwords.

### Things to Focus On
- ✅ GPT uses BPE; BERT uses WordPiece; SentencePiece used in LLaMA, T5
- ✅ [CLS], [SEP], [PAD], [UNK], [MASK] are special tokens — know their purpose
- ✅ Vocabulary size tradeoff: large vocab → shorter sequences but OOM embeddings
- ✅ Tokenization is NOT splitting on spaces — "don't" → ["don", "'", "t"]

### Implementation
```python
import re
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

class CharacterTokenizer:
    """
    Character-level tokenizer: each character = one token.
    Vocabulary = all unique characters in corpus.
    """
    
    def __init__(self, special_tokens: list = None):
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def fit(self, corpus: List[str]) -> 'CharacterTokenizer':
        """Build vocabulary from corpus."""
        all_chars = set(ch for text in corpus for ch in text)
        
        # Add special tokens first
        for tok in self.special_tokens:
            self.char_to_id[tok] = len(self.char_to_id)
        
        # Add all characters
        for ch in sorted(all_chars):
            if ch not in self.char_to_id:
                self.char_to_id[ch] = len(self.char_to_id)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        return self
    
    def encode(self, text: str, max_len: int = None,
               add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert text to token IDs."""
        unk_id = self.char_to_id['<UNK>']
        ids = [self.char_to_id.get(ch, unk_id) for ch in text]
        
        if add_bos:
            ids = [self.char_to_id['<BOS>']] + ids
        if add_eos:
            ids = ids + [self.char_to_id['<EOS>']]
        
        if max_len is not None:
            if len(ids) < max_len:
                ids += [self.char_to_id['<PAD>']] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text."""
        chars = []
        for id_ in ids:
            ch = self.id_to_char.get(id_, '<UNK>')
            if skip_special and ch in self.special_tokens:
                continue
            chars.append(ch)
        return ''.join(chars)


class WordTokenizer:
    """
    Simple whitespace-based word tokenizer.
    Lowercase + split on punctuation.
    """
    
    def __init__(self, max_vocab: int = 10000):
        self.max_vocab = max_vocab
        self.word_to_id = {}
        self.id_to_word = {}
        self.PAD_ID = 0
        self.UNK_ID = 1
    
    def fit(self, corpus: List[str]) -> 'WordTokenizer':
        # Count all words
        word_counts = Counter()
        for text in corpus:
            tokens = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(tokens)
        
        # Build vocab: special tokens + most frequent words
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for word, _ in word_counts.most_common(self.max_vocab - 4):
            self.word_to_id[word] = len(self.word_to_id)
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        return self
    
    def encode(self, text: str) -> List[int]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.word_to_id.get(t, self.UNK_ID) for t in tokens]
    
    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.id_to_word.get(i, '<UNK>') 
                        for i in ids if i > 3)  # Skip special tokens

# Test
corpus = ["Hello world!", "The quick brown fox", "Machine learning is fun"]
tokenizer = CharacterTokenizer()
tokenizer.fit(corpus)
print(f"Vocab size: {tokenizer.vocab_size}")

text = "Hello"
ids = tokenizer.encode(text, add_bos=True, add_eos=True)
print(f"Encoded: {ids}")
print(f"Decoded: {tokenizer.decode(ids)}")
```

---

## 🟡 PROBLEM 2: Byte Pair Encoding (BPE) from Scratch

### Theory
BPE is the most widely used tokenization algorithm (GPT series, RoBERTa).

**Training algorithm:**
1. Start with character-level vocabulary
2. Count all adjacent pair frequencies
3. Merge the most frequent pair → create new subword token
4. Repeat until vocabulary reaches target size

**Inference:** Apply learned merge rules (in order) to new text.

**Why BPE?**
- Unseen words: "ChatGPT" → ["Chat", "G", "PT"] — graceful degradation
- Balances vocabulary size vs sequence length
- Language-agnostic (works for any script)

### Things to Focus On
- ✅ BPE merge rules are applied in the order they were learned
- ✅ GPT-2 uses byte-level BPE — never has OOV (worst case: individual bytes)
- ✅ Vocabulary size: GPT-4 uses ~100k tokens; GPT-2 used ~50k
- ✅ WordPiece (BERT): similar but uses likelihood-based merging instead of frequency

### Implementation
```python
class BytePairEncoder:
    """
    BPE tokenizer training and inference from scratch.
    Simplified ASCII version (byte-level BPE would encode bytes not chars).
    """
    
    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.merges = {}         # (pair) → merged_token, in order learned
        self.vocab = set()
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
    
    def _get_word_freqs(self, corpus: List[str]) -> Dict[tuple, int]:
        """Split corpus into character-level words with frequencies."""
        word_freq = Counter()
        for text in corpus:
            for word in text.lower().split():
                # Add end-of-word marker
                word_chars = tuple(list(word) + ['</w>'])
                word_freq[word_chars] += 1
        return word_freq
    
    def _get_pair_freqs(self, word_freqs: Dict[tuple, int]) -> Counter:
        """Count frequency of all adjacent symbol pairs."""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i+1])] += freq
        return pair_freqs
    
    def _merge_pair(self, word_freqs: Dict[tuple, int], 
                     pair: tuple) -> Dict[tuple, int]:
        """Apply one merge: replace all occurrences of pair with merged token."""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)  # Concatenate pair into new token
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def fit(self, corpus: List[str]) -> 'BytePairEncoder':
        """Learn BPE merge rules from corpus."""
        # Initialize with character vocabulary
        word_freqs = self._get_word_freqs(corpus)
        
        # Initial vocab: all unique characters
        self.vocab = set(ch for word in word_freqs for ch in word)
        
        # Learn merges until target vocab size
        n_merges = self.vocab_size - len(self.vocab)
        
        for i in range(n_merges):
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            
            # Most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged = ''.join(best_pair)
            
            # Record merge rule
            self.merges[best_pair] = merged
            self.vocab.add(merged)
            
            # Apply merge to all words
            word_freqs = self._merge_pair(word_freqs, best_pair)
        
        return self
    
    def encode(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges."""
        tokens = []
        for word in text.lower().split():
            # Start with character-level
            word_tokens = list(word) + ['</w>']
            
            # Apply merges in order
            for pair, merged in self.merges.items():
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        (word_tokens[i], word_tokens[i+1]) == pair):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)

# Test
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the cat sat on the mat",
    "machine learning is the future of technology",
    "natural language processing and machine learning",
]

bpe = BytePairEncoder(vocab_size=100)
bpe.fit(corpus)

print(f"Vocabulary size: {bpe.get_vocab_size()}")
print(f"Top 10 merges: {list(bpe.merges.items())[:10]}")
print(f"Encoded 'machine': {bpe.encode('machine')}")
print(f"Encoded 'learning': {bpe.encode('learning')}")
```

---

## 🟡 PROBLEM 3: Token Embedding Layer

### Theory
Converts token IDs to dense vectors. The embedding matrix E ∈ ℝ^{vocab_size × d_model} is lookup table.

**Forward pass:** `embedding = E[token_id]` — just a row selection.

**Training:** Gradients flow back to the embedding matrix, updating vectors for used tokens.

**Weight tying:** In language models, the input embedding matrix and the output projection matrix are often shared (same weights) — halves parameters and improves performance.

### Implementation
```python
class TokenEmbedding:
    """
    Simple lookup-table token embedding.
    E: (vocab_size, d_model)
    """
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Initialize with small random values
        self.weight = np.random.randn(vocab_size, d_model) * 0.02
        self.weight[padding_idx] = 0  # PAD token has zero embedding
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Lookup embedding for each token.
        token_ids: (batch, seq_len) integer array
        Returns: (batch, seq_len, d_model)
        """
        return self.weight[token_ids]
    
    def __call__(self, token_ids):
        return self.forward(token_ids)

class TransformerEmbedding:
    """
    Full embedding for transformer: token embedding + positional encoding + dropout.
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512,
                 dropout: float = 0.1, padding_idx: int = 0):
        self.d_model = d_model
        self.token_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_enc = self._sinusoidal_pe(max_seq_len, d_model)  # Pre-computed
        self.dropout_rate = dropout
        self.scale = np.sqrt(d_model)  # Scaling factor (Vaswani et al.)
    
    def _sinusoidal_pe(self, max_len: int, d_model: int) -> np.ndarray:
        """Pre-compute sinusoidal positional encodings."""
        PE = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
        PE[:, 0::2] = np.sin(pos / div_term)
        PE[:, 1::2] = np.cos(pos / div_term)
        return PE
    
    def forward(self, token_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        token_ids: (batch, seq_len)
        Returns: (batch, seq_len, d_model)
        """
        seq_len = token_ids.shape[1]
        
        # Scale token embeddings + add positional encoding
        x = self.token_emb(token_ids) * self.scale
        x = x + self.pos_enc[:seq_len]  # Add PE for positions 0..seq_len
        
        # Dropout during training
        if training and self.dropout_rate > 0:
            mask = np.random.rand(*x.shape) > self.dropout_rate
            x = x * mask / (1 - self.dropout_rate)
        
        return x

# Test
vocab_size, d_model, max_seq = 1000, 64, 128
emb = TransformerEmbedding(vocab_size, d_model, max_seq)

# Simulate batch of token IDs
batch_ids = np.array([[5, 23, 67, 12, 0], [14, 88, 3, 55, 99]])  # (2, 5) — 2 samples, 5 tokens
output = emb.forward(batch_ids, training=False)
print(f"Embedding output shape: {output.shape}")  # (2, 5, 64)
print(f"Scale factor: {emb.scale:.3f}")           # sqrt(64) = 8.0
```

---

## 🟡 PROBLEM 4: Feed-Forward Network (FFN) in Transformer

### Theory
Each Transformer layer has a position-wise FFN applied identically to each position:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with GELU: $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$

**Dimensions:** d_model → d_ff → d_model, where d_ff = 4 × d_model (typical).

**Why 4×?** Heuristic from original paper; modern variants use different ratios.

### Implementation
```python
class PositionwiseFFN:
    """
    Position-wise Feed-Forward Network in Transformer.
    Applied identically to each token position.
    """
    
    def __init__(self, d_model: int, d_ff: int = None, 
                 activation: str = 'relu', dropout: float = 0.1):
        d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        
        # He init for activation > 0 regions
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2 / d_ff)
        self.b2 = np.zeros(d_model)
        
        self.activation_fn = {
            'relu': lambda x: np.maximum(0, x),
            'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3))),
        }[activation]
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        x: (..., d_model) — any prefix shape (e.g., batch × seq_len)
        Returns: same shape as x
        """
        # Expand: d_model → d_ff
        hidden = self.activation_fn(x @ self.W1 + self.b1)
        
        # Dropout on hidden
        if training and self.dropout > 0:
            mask = (np.random.rand(*hidden.shape) > self.dropout) / (1 - self.dropout)
            hidden = hidden * mask
        
        # Contract: d_ff → d_model
        return hidden @ self.W2 + self.b2

# Test
ffn = PositionwiseFFN(d_model=64, activation='gelu')
x = np.random.randn(2, 10, 64)  # batch=2, seq=10, d_model=64
out = ffn.forward(x, training=False)
print(f"FFN input:  {x.shape}")   # (2, 10, 64)
print(f"FFN output: {out.shape}") # (2, 10, 64) — shape preserved ✓
```

---

## 🟢 PROBLEM 5: Text Chunking for RAG

### Theory
RAG (Retrieval-Augmented Generation) requires splitting long documents into chunks that are:
1. Small enough to fit in retrieval context
2. Large enough to carry semantic meaning
3. Overlapping to prevent information loss at boundaries

**Chunking strategies:**
- **Fixed-size with overlap:** Simple, fast, universal
- **Sentence-aware:** Split on sentence boundaries
- **Semantic:** Use embedding similarity to find natural break points (advanced)

### Things to Focus On
- ✅ chunk_size: typically 256-512 tokens for retrieval
- ✅ overlap: typically 10-20% of chunk_size to maintain context at boundaries
- ✅ Never split in the middle of a sentence (prefer sentence-boundary chunking)
- ✅ Metadata: always store chunk index, source doc, character offset with each chunk

### Implementation
```python
class TextChunker:
    """
    Multiple chunking strategies for RAG pipelines.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_characters(self, text: str) -> List[Dict]:
        """Fixed-size character-based chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            chunks.append({
                'text': chunk,
                'start_char': start,
                'end_char': min(end, len(text)),
                'chunk_id': len(chunks),
            })
            
            start += self.chunk_size - self.overlap  # Slide with overlap
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_by_sentences(self, text: str, max_chunk_size: int = None) -> List[Dict]:
        """
        Sentence-aware chunking: never split mid-sentence.
        Group sentences until chunk_size reached.
        """
        max_size = max_chunk_size or self.chunk_size
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > max_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({'text': chunk_text, 'chunk_id': len(chunks)})
                
                # Start new chunk with overlap (take last sentence for context)
                overlap_sents = current_chunk[-1:]  # Last sentence as context
                current_chunk = overlap_sents + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        
        if current_chunk:
            chunks.append({'text': ' '.join(current_chunk), 'chunk_id': len(chunks)})
        
        return chunks
    
    def chunk_by_tokens(self, text: str, tokenizer, 
                         chunk_size_tokens: int = 512, 
                         overlap_tokens: int = 50) -> List[Dict]:
        """Token-aware chunking (best for LLM context)."""
        tokens = tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'token_start': start,
                'token_end': min(end, len(tokens)),
                'n_tokens': len(chunk_tokens),
                'chunk_id': len(chunks),
            })
            
            start += chunk_size_tokens - overlap_tokens
        
        return chunks

# Test
long_text = """
Machine learning is a subset of artificial intelligence. It focuses on 
building systems that learn from data. Deep learning uses neural networks 
with many layers. These networks can learn complex patterns. Natural language 
processing allows computers to understand human language. Large language models
are trained on massive text corpora. They can generate coherent text responses.
Retrieval-augmented generation combines retrieval with generation. This helps
ground the model's responses in specific documents.
""".strip()

chunker = TextChunker(chunk_size=200, overlap=50)
char_chunks = chunker.chunk_by_characters(long_text)
sent_chunks = chunker.chunk_by_sentences(long_text, max_chunk_size=300)

print(f"Character chunks: {len(char_chunks)}")
for c in char_chunks:
    print(f"  Chunk {c['chunk_id']}: {len(c['text'])} chars | '{c['text'][:50]}...'")

print(f"\nSentence chunks: {len(sent_chunks)}")
for c in sent_chunks:
    print(f"  Chunk {c['chunk_id']}: '{c['text'][:80]}...'")
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"Why does GPT use BPE but BERT uses WordPiece?"** → Both are subword methods; WordPiece maximizes likelihood of training data; BPE maximizes compression. GPT is autoregressive so needs a tokenizer that works character-by-character; BERT's MLM works token-by-token. Practically, performance similar.
2. **"Why scale embeddings by √d_model?"** → Without scaling, dot products in attention grow with d_model magnitude, causing softmax to saturate. Scaling normalizes dot products to have similar variance regardless of embedding size.
3. **"What's the right chunk size for RAG?"** → Depends on retrieval model (embedding model context limit, usually 512 tokens) and generation model (GPT-4: 128k context). Rule of thumb: retrieve 3-5 chunks of 256-512 tokens each. Too small: loses context; too large: dilutes relevance.
4. **"When would you choose sentence-aware over fixed-size chunking?"** → Sentence-aware for coherent prose (articles, reports). Fixed-size for code (no natural sentence boundaries) or when latency matters. Token-aware is the gold standard when using known tokenizer.
