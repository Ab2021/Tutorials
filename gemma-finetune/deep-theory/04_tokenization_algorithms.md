# 4. Tokenization Algorithms — BPE, WordPiece, Unigram, SentencePiece

## Table of Contents
- [Why Subword Tokenization?](#why-subword-tokenization)
- [Byte-Pair Encoding (BPE) — Step by Step](#byte-pair-encoding-bpe--step-by-step)
- [WordPiece — BERT's Tokenizer](#wordpiece--berts-tokenizer)
- [Unigram LM — Statistical Approach](#unigram-lm--statistical-approach)
- [SentencePiece — Language-Independent Tokenization](#sentencepiece--language-independent-tokenization)
- [Byte-Level BPE — Handling Any Text](#byte-level-bpe--handling-any-text)
- [Vocabulary Size: How to Choose](#vocabulary-size-how-to-choose)
- [Tokenization Artifacts and Edge Cases](#tokenization-artifacts-and-edge-cases)
- [Practical: Analyzing Gemma's Tokenizer](#practical-analyzing-gemmas-tokenizer)

---

## Why Subword Tokenization?

### The Vocabulary Problem

```
Word-level tokenization:
  Vocabulary = all unique words: {"the", "cat", "sat", "unbelievable", "cryptocurrency", ...}
  
  Problems:
  1. Vocabulary size: English has ~170,000 words, plus names, slang, typos...
     Need vocabulary of 500K+ → embedding matrix of 500K × 2048 = 4 GB!
  
  2. Out-of-vocabulary (OOV): "Bitcoinification" → <UNK>
     Any new or rare word becomes meaningless <UNK> token.
  
  3. No morphological awareness:
     "run", "running", "runner", "runs" are 4 completely separate tokens.
     The model must learn each relationship independently.

Character-level tokenization:
  Vocabulary = {'a', 'b', 'c', ..., 'z', '0', ..., '9', ' ', '.', ...}
  Only ~100 tokens needed!
  
  Problems:
  1. Sequence length explosion: "Hello world" → 11 characters vs 2 words
     Attention is O(n²) → quadratically more expensive!
  
  2. Characters carry almost no semantic meaning individually.
     The model must learn to compose characters into meaningful units.

Subword tokenization (Goldilocks zone):
  Vocabulary = {"the", "cat", "un", "believ", "able", "crypto", "currency", ...}
  ~32K-256K tokens
  
  Benefits:
  ✅ Compact vocabulary (fits in reasonable embedding matrix)
  ✅ No OOV: any word decomposes into known subwords
  ✅ Morphologically aware: "un" + "believ" + "able" 
  ✅ Reasonable sequence length
```

---

## Byte-Pair Encoding (BPE) — Step by Step

BPE was originally a compression algorithm (1994). It was adapted for NLP by Sennrich et al. (2016), and is used by GPT-2, GPT-3, GPT-4, and LLaMA.

### Training the Tokenizer (Building the Vocabulary)

```
Training corpus: "low lower newest lowest widest"

Step 0: Start with character-level vocabulary
  Vocabulary: {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', ' ', '</w>'}
  
  Word frequencies (with end-of-word marker):
    "l o w </w>" : 5 times     (low, lower, lowest)
    "n e w e s t </w>" : 2
    "w i d e s t </w>" : 1
    "e r </w>" : 2              (lower)
    "e s t </w>" : 3            (newest, lowest, widest)

Step 1: Count all adjacent pairs
  ('e', 's'): 3     ← MOST FREQUENT
  ('s', 't'): 3
  ('l', 'o'): 5
  ('o', 'w'): 5
  ...

Step 2: Merge most frequent pair ('l', 'o') → 'lo'
  Updated words:
    "lo w </w>" : 5
    "n e w e s t </w>" : 2
    ...
  Vocabulary: {..., 'lo'}

Step 3: Count pairs again
  ('lo', 'w'): 5    ← MOST FREQUENT
  ('e', 's'): 3
  ...

Step 4: Merge ('lo', 'w') → 'low'
  Updated words:
    "low </w>" : 5
    ...
  Vocabulary: {..., 'lo', 'low'}

Step 5: Merge ('e', 's') → 'es'
Step 6: Merge ('es', 't') → 'est'
Step 7: Merge ('est', '</w>') → 'est</w>'
...

Continue until vocabulary reaches desired size (e.g., 256K for Gemma).

Final vocabulary includes:
  Single characters + merged subwords + common words
  {'l', 'o', 'w', ..., 'lo', 'low', 'est', 'newest', ...}
```

### Using the Tokenizer (Encoding New Text)

```
Input: "lowest"

Apply learned merge rules in order:
  Start: ['l', 'o', 'w', 'e', 's', 't']
  Rule 1 (l+o→lo):  ['lo', 'w', 'e', 's', 't']
  Rule 2 (lo+w→low): ['low', 'e', 's', 't']
  Rule 3 (e+s→es):   ['low', 'es', 't']
  Rule 4 (es+t→est): ['low', 'est']
  
Result: ["low", "est"]

Input: "newest"
  Start: ['n', 'e', 'w', 'e', 's', 't']
  → ['n', 'e', 'w', 'es', 't']
  → ['n', 'e', 'w', 'est']
  
Result: ["n", "e", "w", "est"]
  (rarer word → more tokens)
```

---

## WordPiece — BERT's Tokenizer

### How WordPiece Differs from BPE

```
BPE: Merge the most FREQUENT pair.
WordPiece: Merge the pair that maximizes LIKELIHOOD of the training data.

Likelihood-based merging:
  Score(a, b → ab) = freq(ab) / (freq(a) × freq(b))

  This favors merging pairs where:
  - "ab" appears often (high numerator)
  - "a" and "b" individually are rare (low denominator)
  
  Intuition: Merge pairs that are DISTINCTIVELY together,
  not just pairs where each part is common separately.
```

### Example

```
BPE might merge 'th' + 'e' → 'the' early (very frequent pair)
WordPiece might merge 'un' + 'do' → 'undo' first (distinctively together)

In practice, results are similar for large vocabularies.
```

### The ## Prefix Convention

```
WordPiece uses ## to mark CONTINUATION tokens:

"unbelievable" → ["un", "##believ", "##able"]

## means "this token continues the previous word (no space before)"
Without ##: "un" is a standalone word
With ##: "##believ" is part of a longer word
```

---

## Unigram LM — Statistical Approach

### Key Insight: Start Big, Prune Down

```
BPE: Start with characters, ADD merges    (bottom-up)
Unigram: Start with huge vocab, REMOVE tokens  (top-down)

How it works:
  1. Start with a LARGE initial vocabulary (all substrings up to length N)
  2. Assign a probability to each token based on the training data
  3. Compute the overall loss of tokenizing the corpus
  4. For each token, compute: "how much would loss increase if I removed it?"
  5. Remove tokens that increase loss the LEAST (least useful tokens)
  6. Repeat until vocabulary reaches desired size
```

### The Probability Model

```
Unigram assumes each token appears independently:

P("recommendation") = max over all segmentations:
  P("recommend") × P("ation")
  P("re") × P("commend") × P("ation")
  P("r") × P("e") × P("c") × ... (character level)

Find the segmentation with highest probability using the Viterbi algorithm.

This is more principled than BPE's greedy merging:
  BPE: "Whatever merge was most frequent when I got to this step"
  Unigram: "Whatever segmentation maximizes the total probability"
```

### SentencePiece Uses Unigram (for Gemma)

```
Gemma's tokenizer is trained with SentencePiece Unigram model:
  - Start with ~1M candidate tokens
  - Iteratively prune to 256,128 tokens
  - Each remaining token has an optimized probability

Result: More linguistically meaningful subwords than BPE
```

---

## SentencePiece — Language-Independent Tokenization

### What Makes SentencePiece Special

```
Standard tokenizers:
  1. Pre-tokenize text into words (split on spaces/punctuation)
  2. Apply BPE/WordPiece to each word

SentencePiece:
  1. Treat the ENTIRE text as a raw byte/character sequence
  2. Learn subwords directly from raw text
  
  No pre-tokenization needed!
```

### Why Language-Independence Matters

```
Pre-tokenization fails for languages without spaces:

Japanese: "東京は日本の首都です" (Tokyo is the capital of Japan)
  No spaces! How do you split into words?

Chinese: "我喜欢机器学习" (I like machine learning)
  No spaces either!

Thai: "ฉันชอบการเรียนรู้เครื่อง"
  Same problem.

SentencePiece handles ALL of these by treating text as raw characters.
```

### The ▁ (Underscore) Convention

```
SentencePiece uses ▁ (Unicode U+2581) to represent spaces:

Input: "The cat sat"
SentencePiece: ["▁The", "▁cat", "▁sat"]

Why? Because spaces are treated as regular characters in the input.
The ▁ helps reconstruct the original text after tokenization.

This means "The" at the start of a sentence and "the" mid-sentence
are DIFFERENT tokens: ["The"] vs ["▁the"]
```

### SentencePiece Configuration for Gemma

```
model_type: unigram           (not BPE)
vocab_size: 256128            (large vocabulary)
character_coverage: 0.9995    (covers 99.95% of characters in training data)
byte_fallback: true           (unknown characters → byte sequences)
split_digits: true            (each digit becomes a separate token)
```

---

## Byte-Level BPE — Handling Any Text

### The Problem with Unicode

```
Unicode has ~150,000 characters across all scripts.
Even with subword tokenization, rare characters might not appear
in the vocabulary.

"The café costs €15" → What if 'é' and '€' aren't in vocab?
```

### Byte-Level Solution

```
Byte-level BPE (used by GPT-2, GPT-3, LLaMA):
  1. Convert text to UTF-8 bytes (256 possible byte values)
  2. Apply BPE on BYTES, not characters
  3. Every possible text is representable (just 256 base tokens!)

"café" in UTF-8: [99, 97, 102, 195, 169]
  99='c', 97='a', 102='f', then 'é' = 2 bytes [195, 169]

With BPE: "café" might become ["caf", "é"] or ["café"] depending on frequency.
Unknown characters are never a problem — they're just byte sequences.
```

### Gemma Uses SentencePiece with Byte Fallback

```
Primary: Unigram model on Unicode characters
Fallback: Unknown characters → UTF-8 byte tokens

This gives the best of both worlds:
  - Common characters (Latin, CJK, etc.) → single tokens
  - Rare characters (emojis, ancient scripts) → byte sequences
```

---

## Vocabulary Size: How to Choose

### Size Trade-offs

```
Vocabulary Size ▲
                │
  Too Large     │  Embedding matrix becomes huge
  (>500K)       │  Many tokens are rarely used
                │  Train data per token is sparse → poor embeddings
                │
  Sweet Spot    │  Gemma: 256K ─── Most characters covered
  (32K-256K)    │  LLaMA: 32K  ─── Efficient, well-tested
                │  GPT-4: 100K ─── Good balance
                │
  Too Small     │  Sequences become very long
  (<1K)         │  Each token carries little meaning
                │  Attention cost explodes (O(n²))
                │
                └────────────────────────────────────▶
```

### Impact on Sequence Length

```
Same text with different vocab sizes:

"The product recommendation engine works great!"

vocab=100 (character-like):
  ['T','h','e',' ','p','r','o','d','u','c','t',' ','r','e','c','o',
   'm','m','e','n','d','a','t','i','o','n',' ','e','n','g','i','n',
   'e',' ','w','o','r','k','s',' ','g','r','e','a','t','!']
  = 46 tokens

vocab=32K (GPT-2):
  ['The', ' product', ' recommendation', ' engine', ' works', ' great', '!']
  = 7 tokens

vocab=256K (Gemma):
  ['The', ' product', ' recommendation', ' engine', ' works', ' great!']
  = 6 tokens

46 tokens vs 6 tokens for the same text!
Attention cost: 46² = 2116 vs 6² = 36 → 59× cheaper!
```

---

## Tokenization Artifacts and Edge Cases

### Fertility (Tokens Per Word)

```
"the"           → 1 token    (common word, fertility=1)
"recommendation"→ 2 tokens   (common, moderate fertility)
"antidisestablishmentarianism" → 6 tokens (rare, high fertility)
"asdfghjkl"     → 9 tokens   (nonsense → character-level, very high)

Average fertility for English: ~1.3 tokens/word
Average fertility for code: ~2.5 tokens/word (more variable)
```

### Leading Space Issue

```
In SentencePiece (Gemma):
  "Hello" at sentence start → ["Hello"]
  " Hello" with leading space → ["▁Hello"]
  
  These are DIFFERENT tokens! The space is part of the token.
  
  This matters for inference:
    prompt = "What is " + "AI?"
    Tokenize("What is AI?") ≠ Tokenize("What is ") + Tokenize("AI?")
    
    Always tokenize the COMPLETE text, not pieces separately!
```

### Number Tokenization

```
"2024" can tokenize as:
  ["2024"]                     (if common number)
  ["20", "24"]                 (split into 2-digit groups)
  ["2", "0", "2", "4"]        (digit by digit)

Gemma uses split_digits=True:
  "2024" → ["2", "0", "2", "4"]  (always splits digits)

Why? Numbers have compositional meaning:
  "2024" = 2×1000 + 0×100 + 2×10 + 4
  The model can learn arithmetic patterns from individual digits.
```

---

## Practical: Analyzing Gemma's Tokenizer

```python
from transformers import AutoTokenizer

# Load Gemma's tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# ===== BASIC TOKENIZATION =====
text = "This product recommendation engine is amazing!"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {ids}")
print(f"Number of tokens: {len(tokens)}")

# ===== VOCABULARY ANALYSIS =====
print(f"\nVocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.all_special_tokens}")

# ===== FERTILITY ANALYSIS =====
words = ["the", "product", "recommendation", "unbelievable",
         "cryptocurrency", "antidisestablishmentarianism"]
for word in words:
    toks = tokenizer.tokenize(word)
    print(f"  '{word}' → {toks} ({len(toks)} tokens)")

# ===== DECODE BACK =====
reconstructed = tokenizer.decode(ids)
print(f"\nReconstructed: {reconstructed}")
print(f"Match: {text == reconstructed}")

# ===== BATCH TOKENIZATION =====
texts = [
    "Great phone, love the camera!",
    "Terrible product, waste of money.",
    "Average quality, decent price.",
]
batch = tokenizer(texts, padding=True, truncation=True,
                   max_length=512, return_tensors="pt")
print(f"\nBatch shapes:")
print(f"  input_ids: {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
```
