# Day 25: BERT & Encoder Models - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: BERT, RoBERTa, and Masked Language Modeling

## 1. Theoretical Foundation: Bidirectionality

GPT (Decoder) is Unidirectional (Left-to-Right). Good for generation.
**BERT (Encoder)** is Bidirectional. Good for understanding.
*   It sees the whole sentence at once.
*   Problem: We can't use standard Language Modeling (predict next word) because the model would "see" the answer from future tokens.

## 2. Pre-training Objectives

### Masked Language Modeling (MLM)
*   Mask 15% of tokens.
*   Predict the masked tokens.
*   Input: `The [MASK] sat on the mat.`
*   Target: `cat`.
*   Forces model to use context from both left and right.

### Next Sentence Prediction (NSP)
*   Input: `[CLS] Sentence A [SEP] Sentence B`
*   Binary Classification: Is B the actual next sentence after A?
*   Teaches relationship between sentences (QA, NLI).

## 3. Architecture Variants

*   **BERT (Bidirectional Encoder Representations from Transformers)**: The original.
*   **RoBERTa (Robustly optimized BERT)**:
    *   Removed NSP (it didn't help much).
    *   More data, longer training, dynamic masking.
    *   Better performance.
*   **DistilBERT**: 40% smaller, 97% performance. Knowledge Distillation.

## 4. Implementation: Fine-Tuning BERT for Classification

We don't train BERT from scratch. We **Fine-Tune**.
Add a Linear Layer on top of the `[CLS]` token embedding.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Load Pre-trained Model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Tokenize Input
text = "This movie was fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 3. Forward Pass
outputs = model(**inputs)
logits = outputs.logits
probs = torch.softmax(logits, dim=1)

# 4. Training Loop (Standard PyTorch)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) # Low LR!
loss = outputs.loss
loss.backward()
optimizer.step()
```

## 5. The [CLS] Token

Special token added to the start of every sequence.
The final hidden state of `[CLS]` is designed to aggregate the entire sequence representation.
Used for Classification tasks.
