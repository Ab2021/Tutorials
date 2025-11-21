# Day 27: Tokenization - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: BPE, WordPiece, and SentencePiece

## 1. Theoretical Foundation: Why Subwords?

*   **Word-level**: Vocab too large (1M+). OOV issues.
*   **Char-level**: Sequence too long. Loss of semantics.
*   **Subword-level**: Sweet spot. "unhappily" $\to$ "un", "happy", "ly".
    *   Common words = Single token.
    *   Rare words = Multiple tokens.

## 2. Algorithms

### Byte Pair Encoding (BPE) - GPT-2/3, RoBERTa
1.  Start with characters.
2.  Count most frequent pair of adjacent tokens (e.g., "e", "s" $\to$ "es").
3.  Merge them into a new token.
4.  Repeat until vocab size reached.

### WordPiece - BERT
Similar to BPE, but merges based on **Likelihood** (Language Model probability) rather than just frequency.
Maximizes $P(training\_data)$.

### SentencePiece - T5, LLaMA
*   Treats input as raw stream of unicode characters (including spaces).
*   Language agnostic (doesn't need pre-tokenization like splitting by space).
*   Implements BPE or Unigram.

## 3. Implementation: Training a Tokenizer

Using Hugging Face `tokenizers` library (Rust-based, fast).

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# 1. Initialize BPE Tokenizer
tokenizer = Tokenizer(models.BPE())

# 2. Pre-tokenization (Split by whitespace)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Trainer
trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=30000)

# 4. Train on files
files = ["data/wiki.txt"]
tokenizer.train(files, trainer)

# 5. Save
tokenizer.save("my_bpe_tokenizer.json")

# 6. Use
encoded = tokenizer.encode("Hello, PyTorch!")
print(encoded.tokens)
# ['Hello', ',', 'Py', 'Torch', '!']
```

## 4. Byte-Level BPE (GPT-2)

Standard BPE requires a base vocabulary of characters.
Unicode has 140k characters. Too big.
**Byte-Level**: Process UTF-8 bytes directly. Base vocab size = 256.
*   Ensures *every* string is tokenizable.
*   No `<UNK>` token needed!
