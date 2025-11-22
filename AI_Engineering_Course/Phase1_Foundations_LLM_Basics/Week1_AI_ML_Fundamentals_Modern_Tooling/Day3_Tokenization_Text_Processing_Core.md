# Day 3: Tokenization & Text Processing
## Core Concepts & Theory

### The Tokenization Problem

Text is continuous, but neural networks operate on discrete tokens. How do we split text into meaningful units?

**Options:**
1. **Character-level**: ["H", "e", "l", "l", "o"] - Too granular, long sequences
2. **Word-level**: ["Hello", "world"] - Huge vocabulary (100K+ words), OOV problems
3. **Subword-level**: ["Hello", "world"] or ["Hel", "##lo", "wor", "##ld"] - Best trade-off

### Why Subword Tokenization?

**Problem with Words:**
- English has 170,000+ words
- With morphology: "run", "running", "runs", "ran" - should these be separate?
- New words: "COVID-19", "ChatGPT" - not in vocabulary
- Rare words: Waste embedding capacity

**Subword Solution:**
- Finite vocabulary (30K-50K tokens)
- Covers infinite text through composition
- "unbelievable" → ["un", "##believe", "##able"]
- Learns morphology: "un-" prefix, "-able" suffix

### Byte-Pair Encoding (BPE)

**Algorithm:**

1. Start with character vocabulary: ['a', 'b', ..., 'z']
2. Count all adjacent byte pairs in corpus
3. Merge most frequent pair → new token
4. Repeat until vocabulary size reached

**Example:**

Corpus: "low low low lower lowest"

```
Initial: l o w _ l o w _ l o w _ l o w e r _ l o w e s t
Count pairs: ('l','o')=5, ('o','w')=5, ('w','_')=4, ('w','e')=2

Merge ('l','o') → 'lo':
lo w _ lo w _ lo w _ lo w e r _ lo w e s t

Merge ('lo','w') → 'low':
low _ low _ low _ low e r _ low e s t

Merge ('low','_') → 'low_':
low_ low_ low_ low e r _ low e s t

... continues until vocab_size reached
```

**Result Vocabulary:**
```
['low', 'low_', 'lower', 'lowest', 'e', 'r', 's', 't', '_']
```

**Used by:** GPT-2, GPT-3, RoBERTa, BART

### WordPiece (BERT Tokenizer)

Similar to BPE but uses **likelihood** instead of frequency.

**Algorithm:**
1. Start with character vocab
2. For each pair, calculate: P(pair) / (P(first) × P(second))
   - High score = pair appears together more than independently
3. Merge highest-scoring pair
4. Repeat

**Example:**

"unhappiness" → ["un", "##happi", "##ness"]

**Markers:**
- No marker: Start of word ("un")
- `##`: Continuation ("##happi", "##ness")

**Used by:** BERT, DistilBERT, ELECTRA

### SentencePiece (Unigram LM)

**Key Innovation:** Treats text as raw bytes (no pre-tokenization).

**Algorithm:**
1. Start with large vocabulary (all substrings)
2. Iteratively remove tokens that increase loss least
3. Use Unigram Language Model to score tokenizations

**Advantages:**
- Language-agnostic (no whitespace assumption)
- Works for Japanese, Chinese (no word boundaries)
- Reversible (can recover original text exactly)

**Used by:** T5, ALBERT, XLNet, LLaMA, Mistral

**Example Configuration:**
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,  # Covers 99.95% of characters
    model_type='unigram'  # or 'bpe'
)
```

### Modern Tokenization with HuggingFace

**Training a BPE Tokenizer:**

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize BPE model
tokenizer = Tokenizer(models.BPE())

# Pre-tokenization (split on whitespace and punctuation)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train on corpus
files = ["corpus1.txt", "corpus2.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("tokenizer.json")
```

**Using Tokenizer:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Encode
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 11, 703, 389, 345, 30]

# Decode
text = tokenizer.decode(tokens)
print(text)  # "Hello, how are you?"

# Get token strings
print(tokenizer.convert_ids_to_tokens(tokens))
# ['Hello', ',', ' how', ' are', ' you', '?']
```

### Special Tokens

**Common Special Tokens:**
# Day 3: Tokenization & Text Processing
## Core Concepts & Theory

### The Tokenization Problem

Text is continuous, but neural networks operate on discrete tokens. How do we split text into meaningful units?

**Options:**
1. **Character-level**: ["H", "e", "l", "l", "o"] - Too granular, long sequences
2. **Word-level**: ["Hello", "world"] - Huge vocabulary (100K+ words), OOV problems
3. **Subword-level**: ["Hello", "world"] or ["Hel", "##lo", "wor", "##ld"] - Best trade-off

### Why Subword Tokenization?

**Problem with Words:**
- English has 170,000+ words
- With morphology: "run", "running", "runs", "ran" - should these be separate?
- New words: "COVID-19", "ChatGPT" - not in vocabulary
- Rare words: Waste embedding capacity

**Subword Solution:**
- Finite vocabulary (30K-50K tokens)
- Covers infinite text through composition
- "unbelievable" → ["un", "##believe", "##able"]
- Learns morphology: "un-" prefix, "-able" suffix

### Byte-Pair Encoding (BPE)

**Algorithm:**

1. Start with character vocabulary: ['a', 'b', ..., 'z']
2. Count all adjacent byte pairs in corpus
3. Merge most frequent pair → new token
4. Repeat until vocabulary size reached

**Example:**

Corpus: "low low low lower lowest"

```
Initial: l o w _ l o w _ l o w _ l o w e r _ l o w e s t
Count pairs: ('l','o')=5, ('o','w')=5, ('w','_')=4, ('w','e')=2

Merge ('l','o') → 'lo':
lo w _ lo w _ lo w _ lo w e r _ lo w e s t

Merge ('lo','w') → 'low':
low _ low _ low _ low e r _ low e s t

Merge ('low','_') → 'low_':
low_ low_ low_ low e r _ low e s t

... continues until vocab_size reached
```

**Result Vocabulary:**
```
['low', 'low_', 'lower', 'lowest', 'e', 'r', 's', 't', '_']
```

**Used by:** GPT-2, GPT-3, RoBERTa, BART

### WordPiece (BERT Tokenizer)

Similar to BPE but uses **likelihood** instead of frequency.

**Algorithm:**
1. Start with character vocab
2. For each pair, calculate: P(pair) / (P(first) × P(second))
   - High score = pair appears together more than independently
3. Merge highest-scoring pair
4. Repeat

**Example:**

"unhappiness" → ["un", "##happi", "##ness"]

**Markers:**
- No marker: Start of word ("un")
- `##`: Continuation ("##happi", "##ness")

**Used by:** BERT, DistilBERT, ELECTRA

### SentencePiece (Unigram LM)

**Key Innovation:** Treats text as raw bytes (no pre-tokenization).

**Algorithm:**
1. Start with large vocabulary (all substrings)
2. Iteratively remove tokens that increase loss least
3. Use Unigram Language Model to score tokenizations

**Advantages:**
- Language-agnostic (no whitespace assumption)
- Works for Japanese, Chinese (no word boundaries)
- Reversible (can recover original text exactly)

**Used by:** T5, ALBERT, XLNet, LLaMA, Mistral

**Example Configuration:**
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,  # Covers 99.95% of characters
    model_type='unigram'  # or 'bpe'
)
```

### Modern Tokenization with HuggingFace

**Training a BPE Tokenizer:**

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize BPE model
tokenizer = Tokenizer(models.BPE())

# Pre-tokenization (split on whitespace and punctuation)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train on corpus
files = ["corpus1.txt", "corpus2.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("tokenizer.json")
```

**Using Tokenizer:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Encode
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 11, 703, 389, 345, 30]

# Decode
text = tokenizer.decode(tokens)
print(text)  # "Hello, how are you?"

# Get token strings
print(tokenizer.convert_ids_to_tokens(tokens))
# ['Hello', ',', ' how', ' are', ' you', '?']
```

### Special Tokens

**Common Special Tokens:**

- `[PAD]`: Padding (make sequences same length)
- `[UNK]`: Unknown (out-of-vocabulary)
- `[CLS]`: Classification token (BERT)
- `[SEP]`: Separator (between sentences)
- `[MASK]`: Masked token (for MLM pre-training)
- `<BOS>`: Beginning of sequence (GPT models)
- `<EOS>`: End of sequence

**Purpose:**
- Structure input for model
- Enable batching (pad to same length)
- Multi-task learning (task-specific tokens)

### Vocabulary Size Trade-offs

**Small Vocabulary (10K tokens):**
- ✓ Smaller embedding matrix (less memory)
- ✓ Faster softmax (fewer logits)
- ✗ Longer sequences (less efficient)
- ✗ More UNK tokens

**Large Vocabulary (100K tokens):**
- ✓ Shorter sequences (more efficient)
- ✓ Fewer UNK tokens
- ✗ Huge embedding matrix
- ✗ Slow softmax (100K-way classification)

**Sweet Spot:** 30K-50K tokens (BERT: 30K, GPT-2: 50K, LLaMA: 32K)

### Challenges in Tokenization

**1. Multilingual Tokenization:**

English-centric tokenizers waste capacity on non-English text.

Example (GPT-2 tokenizer on Thai):
```
"สวัสดี" → 20 tokens! (Each character multiple bytes)
```

**Solution:** 
- Train on multilingual corpus
- Ensure character_coverage includes all scripts
- Modern: mT5, XLM-R use multilingual SentencePiece

**2. Code Tokenization:**

Code has different statistics than natural language:
- Indentation (spaces/tabs) important
- Case-sensitive
- Special characters ({},[],->)
- Long identifiers (`calculateUserProfileScore`)

**Solution:**
- CodeGen, CodeLLaMA: Train tokenizers on code + natural language
- Preserve whitespace
- Split camelCase appropriately

**3. Number Tokenization:**

"1234567" → ["1234", "567"] or ["1", "2", "3", "4", "5", "6", "7"]?

**Problem:** Rare numbers split into many tokens, lose numeric meaning.

**Solutions:**
- Byte-level encoding (GPT-2): Treats each digit separately
- Number-aware tokenization: Special handling for numeric sequences
- Still an open problem for LLMs

**4. Tokenization Bias:**

Tokenizer affects model behavior.

Example: 
- "African American" → ["African", "American"] (2 tokens)
- "white" → ["white"] (1 token)

More tokens required → more parameters needed to represent → less efficient learning.

Leads to representation bias in models!

### Practical Tokenization

**Padding and Truncation:**

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = ["Short text", "This is a much longer text that will be truncated or padded"]

# Pad to max length in batch
encoded = tokenizer(
    text,
    padding=True,  # Pad to longest in batch
    truncation=True,  # Truncate to max_length
    max_length=20,
    return_tensors="pt"
)

print(encoded['input_ids'].shape)  # (2, 20)
print(encoded['attention_mask'])   # Marks real vs padding tokens
```

**Attention Masks:**

```python
# attention_mask: 1 for real tokens, 0 for padding
# Shape: (batch_size, seq_len)

# Model ignores padding tokens using attention_mask
outputs = model(input_ids, attention_mask=attention_mask)
```

### Tokenizer Comparison

| Feature | BPE (GPT) | WordPiece (BERT) | SentencePiece (T5) |
|---------|-----------|------------------|---------------------|
| Merge Criterion | Frequency | Likelihood | Unigram LM |
| Pre-tokenization | Whitespace | Whitespace | None (byte-level) |
| Language Support | Good | Good | Excellent (no whitespace assumption) |
| Reversibility | Approximate | Approximate | Exact |
| Speed | Fast | Fast | Moderate |

### Summary

Tokenization is the bridge between raw text and neural networks:

- **BPE (GPT):** Frequency-based merging, good for English
- **WordPiece (BERT):** Likelihood-based merging, similar performance
- **SentencePiece (T5, LLaMA):** Unigram LM, language-agnostic, modern choice

Key considerations:
- Vocabulary size (30K-50K sweet spot)
- Special tokens for structure
- Multilingual and code support
- Bias and fairness implications

Modern LLMs spend significant effort on tokenizer design - it fundamentally affects model efficiency, behavior, and fairness.
