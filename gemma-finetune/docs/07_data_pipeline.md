# 7. Data Pipeline — Tokenization, Prompts & Dataset Preparation

## Table of Contents
- [The Data Pipeline Overview](#the-data-pipeline-overview)
- [Tokenization: Text to Numbers](#tokenization-text-to-numbers)
- [SentencePiece Tokenizer](#sentencepiece-tokenizer)
- [Prompt Engineering for Fine-Tuning](#prompt-engineering-for-fine-tuning)
- [Instruction Format: Why It Matters](#instruction-format-why-it-matters)
- [Data Cleaning](#data-cleaning)
- [Train/Validation/Test Splits](#trainvalidationtest-splits)
- [Padding and Truncation](#padding-and-truncation)
- [Dataset Size: How Much Data Do You Need?](#dataset-size-how-much-data-do-you-need)
- [The Amazon Reviews Dataset](#the-amazon-reviews-dataset)

---

## The Data Pipeline Overview

```
Raw Data (from HuggingFace Hub)
    │
    ▼
┌────────────────┐
│  1. DOWNLOAD   │  Fetch from HuggingFace Hub → cached locally
└───────┬────────┘
        ▼
┌────────────────┐
│  2. CLEAN      │  Remove HTML, normalize whitespace, filter bad examples
└───────┬────────┘
        ▼
┌────────────────┐
│  3. FORMAT     │  Convert to instruction-style prompts
│                │  (user turn + model turn + Gemma chat tokens)
└───────┬────────┘
        ▼
┌────────────────┐
│  4. SPLIT      │  80% train / 10% validation / 10% test
└───────┬────────┘
        ▼
┌────────────────┐
│  5. TOKENIZE   │  Text → token IDs → padded tensors
└───────┬────────┘
        ▼
Ready for Training!
```

---

## Tokenization: Text to Numbers

### Why Tokenize?

Neural networks work with **numbers**, not text. Tokenization converts text into sequences of numbers that the model can process.

```
"The product is amazing!" → [651, 2857, 338, 11294, 100]
                             ↑     ↑     ↑     ↑      ↑
                            "The" "product" "is" "amazing" "!"
```

### Types of Tokenization

**1. Word-Level (Old Approach)**
```
"The product is amazing!" → ["The", "product", "is", "amazing", "!"]

Problems:
  - Huge vocabulary (millions of words)
  - Can't handle new/misspelled words → <UNK> token
  - "running" and "run" are completely separate tokens
```

**2. Character-Level**
```
"cat" → ["c", "a", "t"]

Problems:
  - Very long sequences (each character = 1 token)
  - Model needs to learn spelling from scratch
  - Slow training due to long sequences
```

**3. Subword Tokenization (Modern Approach — Used by Gemma)**
```
"unhappiness" → ["un", "happi", "ness"]

Benefits:
  - Compact vocabulary (~256K tokens for Gemma)
  - Can handle ANY word (even made-up ones)
  - Common words = 1 token, rare words = multiple tokens
  - Morphologically aware ("un" + "happi" + "ness")
```

---

## SentencePiece Tokenizer

### What Is SentencePiece?

Gemma uses a **SentencePiece** tokenizer — a language-independent text tokenizer that treats the entire text as a sequence of Unicode characters and learns optimal subword units.

### How It Was Trained

```
1. Take a massive text corpus
2. Start with all individual characters as the vocabulary
3. Iteratively merge the most frequent pair:
   
   Iteration 1: "t" + "h" → "th"  (most common pair)
   Iteration 2: "th" + "e" → "the" (now common)
   Iteration 3: "i" + "n" → "in"
   ...
   Iteration 256128: Stop (vocabulary size reached)

Result: A vocabulary of 256,128 subword units that optimally
covers the training data.
```

### Gemma's Vocabulary

```
Gemma vocab size: 256,128 tokens

Examples of tokens:
  Single characters:  "a", "b", "!", "?", " "
  Common words:       "the", "and", "is", "product"
  Subword pieces:     "amaz", "ing", "rev", "iew"
  Special tokens:     <bos> (begin), <eos> (end), <pad>
  
Tokenization examples:
  "amazing"      → ["amaz", "ing"]        (2 tokens)
  "the"          → ["the"]                 (1 token)
  "recommendation" → ["recommend", "ation"]  (2 tokens)
```

### Special Tokens in Gemma

| Token | Name | ID | Purpose |
|-------|------|----|---------|
| `<bos>` | Begin of Sequence | 2 | Marks start of input |
| `<eos>` | End of Sequence | 1 | Marks end of input/generation |
| `<pad>` | Padding | 0 | Fills unused positions |
| `<start_of_turn>` | Turn Start | - | Marks start of a speaker turn |
| `<end_of_turn>` | Turn End | - | Marks end of a speaker turn |

---

## Prompt Engineering for Fine-Tuning

### Why Prompt Format Matters

The format of your training data determines what the model learns. Bad format = bad model.

### Our Prompt Template

```
<start_of_turn>user
You are a product recommendation and strategy expert. Based on the
following product review, provide:
1. A recommendation (buy/skip/consider) with reasoning
2. Key product strengths and weaknesses
3. A brief strategy suggestion for the product brand

Product Category: {category}
Product Title: {title}
Rating: {rating}/5
Review: {review}
<end_of_turn>
<start_of_turn>model
{response}
<end_of_turn>
```

### Why This Format?

**1. Uses Gemma's native chat format**
```
<start_of_turn>user ... <end_of_turn>
<start_of_turn>model ... <end_of_turn>
```
Gemma was pre-trained to understand these markers. Using them means the model already knows "user = input, model = I should respond."

**2. Clear instruction**
The system prompt tells the model exactly what to output:
- Recommendation (buy/skip/consider)
- Strengths and weaknesses
- Brand strategy

**3. Structured input**
Providing category, title, rating, and review as labeled fields makes it easy for the model to parse.

**4. Consistent structure**
Every training example follows the SAME format. The model learns the pattern, not just individual examples.

### Bad vs Good Prompt Design

```
❌ BAD: Inconsistent format
  Example 1: "Review: Great phone → Buy it"
  Example 2: "This laptop is bad. Rating: 2. Don't buy."
  Example 3: "Product: Chair, 5 stars. Recommendation: BUY"

✅ GOOD: Consistent format (our approach)
  Every example follows the exact same template:
  <start_of_turn>user ... <end_of_turn>
  <start_of_turn>model ... <end_of_turn>
```

---

## Instruction Format: Why It Matters

### SFT (Supervised Fine-Tuning) Format

For fine-tuning, every example must have:
1. **Input context** (what the model receives)
2. **Expected output** (what the model should generate)

The model's loss is computed ONLY on the expected output tokens:

```
<start_of_turn>user                      ← Not used for loss
... review text ...                       ← Not used for loss
<end_of_turn>                             ← Not used for loss
<start_of_turn>model                      ← Not used for loss
**Recommendation: BUY**                   ← LOSS COMPUTED HERE
**Reasoning:** This product...            ← LOSS COMPUTED HERE
<end_of_turn>                             ← LOSS COMPUTED HERE

The model learns to generate the "model" turn given the "user" turn.
```

### Alternative Formats

| Format | Example | When to Use |
|--------|---------|-------------|
| **Chat/Instruction** (ours) | User turn → Model turn | Best for task-specific fine-tuning |
| **Completion** | Just raw text continuation | Pre-training style, less structured |
| **Question-Answer** | Q: ... A: ... | Simple Q&A tasks |
| **JSON structured** | {input: ..., output: ...} | When you need structured output |

---

## Data Cleaning

### Why Clean Data?

Raw web-scraped reviews often contain:
- HTML tags: `<br>`, `<b>great</b>`
- HTML entities: `&amp;`, `&lt;`
- Excessive whitespace and newlines
- Non-printable characters
- Empty or near-empty reviews
- Duplicate reviews

### Our Cleaning Process

```python
def clean_text(text):
    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    #    "This is <b>great</b>!" → "This is  great !"
    
    # 2. Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    #    "Tom &amp; Jerry" → "Tom   Jerry"
    
    # 3. Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    #    "Tom   Jerry" → "Tom Jerry"
    
    # 4. Strip edges
    text = text.strip()
    
    return text
```

### Why Each Step Matters

| Problem | Impact on Training | Our Fix |
|---------|-------------------|---------|
| HTML tags | Model wastes capacity learning tags | Remove with regex |
| Extra whitespace | Wastes token budget | Collapse to single spaces |
| Empty reviews | No useful signal, wastes compute | Filter (min 10 chars) |
| Duplicates | Model memorizes instead of learning | HuggingFace handles dedup |

---

## Train/Validation/Test Splits

### Why Three Splits?

```
┌───────────────────────────────────────────────────────┐
│                    YOUR DATASET                        │
│                                                       │
│  ┌──────────────────────────────────┐                 │
│  │       TRAINING SET (80%)          │                 │
│  │  Model LEARNS from this data      │                 │
│  └──────────────────────────────────┘                 │
│                                                       │
│  ┌─────────────────┐  ┌─────────────────┐            │
│  │ VALIDATION (10%) │  │   TEST (10%)    │            │
│  │ Monitor during   │  │ Final eval ONCE │            │
│  │ training to      │  │ after training  │            │
│  │ detect overfit   │  │ is complete     │            │
│  └─────────────────┘  └─────────────────┘            │
└───────────────────────────────────────────────────────┘
```

**Training set:** The model learns from these examples.
**Validation set:** Used DURING training to check if the model is overfitting. The model never learns from validation data — it's only used to compute validation loss.
**Test set:** Used AFTER training is complete to get a final, unbiased evaluation. Never looked at during training.

### Why Not Just Train/Test?

```
Without validation set:
  You train for 10 epochs
  After training, you find the test loss is bad
  But you don't know WHICH epoch was best!
  You can't go back in time.

With validation set:
  Every 100 steps, you check validation loss
  You keep the checkpoint with the LOWEST validation loss
  This is the "best model" even if training continued too long
```

### Our Split

```python
validation_split = 0.1  # 10% for validation
# We then split the holdout in half: 10% val + 10% test

Given 5000 total samples:
  Train: 4000 (80%)
  Val:   500  (10%)
  Test:  500  (10%)
```

---

## Padding and Truncation

### The Problem

Different reviews have different lengths:

```
Review 1: "Great phone!"                    → 3 tokens
Review 2: "The battery life is incredible,  → 120 tokens
           the camera takes stunning photos,
           and the design is modern..."
Review 3: "This is the worst product I      → 450 tokens
           have ever purchased in my entire
           life. Let me explain in detail..."
```

But neural networks need FIXED-size inputs (all tensors in a batch must have the same shape).

### Solution: Padding + Truncation

```
max_seq_length = 512

Review 1 (3 tokens):   [651, 2857, 100, PAD, PAD, PAD, ..., PAD]
                        ↑                ↑─── padded to 512 ───↑

Review 2 (120 tokens):  [651, ..., 120 real tokens, PAD, ..., PAD]
                        ↑                            ↑── padded ─↑

Review 3 (450 tokens):  [651, ..., 450 real tokens, PAD, ..., PAD]
                        ↑                            ↑── padded ─↑

Review 4 (800 tokens):  [651, ..., first 512 tokens]  ← TRUNCATED!
                        Last 288 tokens are lost
```

### Padding Side Matters

**Training: Right Padding**
```
[real, real, real, PAD, PAD, PAD]
 ↑ model processes real tokens first, then padding
```

**Inference: Left Padding**
```
[PAD, PAD, PAD, real, real, real]
 ↑ padding at start, model generates after real tokens
```

### Attention Mask

The attention mask tells the model which tokens are real (1) and which are padding (0):

```
Tokens:         [651, 2857, 100, PAD, PAD]
Attention mask: [1,   1,    1,   0,   0  ]

The model ignores padding tokens in attention computation.
```

---

## Dataset Size: How Much Data Do You Need?

### General Guidelines

| Dataset Size | Expected Result | Training Time |
|-------------|-----------------|---------------|
| 100 examples | Smoke test only | 5 minutes |
| 1,000 | Model learns basic patterns | 15 minutes |
| **5,000** | **Good for initial fine-tuning (our default)** | **30-60 min** |
| 20,000 | Solid production quality | 2-4 hours |
| 100,000+ | Excellent quality | 12+ hours |

### Quality vs Quantity

```
1,000 high-quality, diverse examples
  >
10,000 repetitive, low-quality examples

Quality matters MORE than quantity for fine-tuning.
```

### When to Add More Data

- Validation loss is still decreasing at the end of training → more data helps
- Model produces repetitive outputs → needs more diverse examples
- Model fails on specific categories → add more examples from those categories

---

## The Amazon Reviews Dataset

### About the Dataset

**Name:** McAuley-Lab/Amazon-Reviews-2023
**Source:** HuggingFace Hub
**Size:** 571 million reviews across 33 product categories
**Fields:**

| Field | Type | Example |
|-------|------|---------|
| `rating` | float | 4.0 |
| `title` | string | "Great phone case" |
| `text` | string | "This case fits perfectly..." |
| `parent_asin` | string | "B07X4WY3K2" |
| `user_id` | string | "A123..." |
| `timestamp` | int | 1679356800 |
| `verified_purchase` | bool | True |

### Subsets We Can Use

| Subset | Size | Recommended For |
|--------|------|-----------------|
| **`raw_review_All_Beauty`** | **~370K** | **Our default — manageable** |
| `raw_review_Electronics` | ~20M | Large-scale training |
| `raw_review_Books` | ~25M | Book recommendations |
| `raw_review_Clothing_Shoes_and_Jewelry` | ~11M | Fashion |
| `raw_review_Home_and_Kitchen` | ~8M | Home products |

### Why This Dataset?

1. **Real-world data** — actual customer reviews, not synthetic
2. **Large enough** — even the smallest subset has 370K examples
3. **Structured** — has ratings, titles, text, categories
4. **Free and open** — available on HuggingFace Hub
5. **Perfect for our task** — product reviews → recommendations
