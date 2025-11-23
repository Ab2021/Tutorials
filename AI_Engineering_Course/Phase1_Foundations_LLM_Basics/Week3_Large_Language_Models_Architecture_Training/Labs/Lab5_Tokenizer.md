# Lab 5: Custom Tokenizer (BPE)

## Objective
Train a Byte-Pair Encoding (BPE) tokenizer from scratch.
We will use the `tokenizers` library (Rust backend).

## 1. Setup
```bash
pip install tokenizers
```

## 2. The Script (`train_tokenizer.py`)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Initialize
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Train
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = ["wiki.txt"] # Create this file with some text
tokenizer.train(files, trainer)

# 3. Save
tokenizer.save("custom_bpe.json")

# 4. Test
output = tokenizer.encode("Hello World")
print(output.tokens)
```

## 3. Challenge
Compare the vocabulary size vs text compression rate.
Smaller vocab = longer sequences. Larger vocab = shorter sequences.

## 4. Submission
Submit the `custom_bpe.json` file.
