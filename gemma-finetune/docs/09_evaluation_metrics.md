# 9. Evaluation Metrics — Measuring Model Quality

## Table of Contents
- [Why Evaluate?](#why-evaluate)
- [Perplexity: The Training Metric](#perplexity-the-training-metric)
- [ROUGE: Recall-Oriented Evaluation](#rouge-recall-oriented-evaluation)
- [BLEU: Precision-Oriented Evaluation](#bleu-precision-oriented-evaluation)
- [ROUGE vs BLEU: When to Use Which](#rouge-vs-bleu-when-to-use-which)
- [Human Evaluation](#human-evaluation)
- [Interpreting Scores](#interpreting-scores)
- [Our Evaluation Pipeline](#our-evaluation-pipeline)

---

## Why Evaluate?

Training loss tells you the model is learning, but it doesn't tell you if the model is generating **good outputs**. We need external metrics to measure output quality.

```
Training Loss:  "The model is getting better at predicting next tokens"
Evaluation Metrics: "The model's GENERATED outputs are similar to expected outputs"

These are different things!
A model can have low training loss but generate repetitive or off-topic text.
```

---

## Perplexity: The Training Metric

### What Is Perplexity?

Perplexity measures how "surprised" the model is by the data. Lower = better.

```
perplexity = 2^(cross_entropy_loss)

Or equivalently:
perplexity = e^(cross_entropy_loss)  (using natural log)

Examples:
  loss = 1.0  → perplexity = e^1.0 = 2.72
  loss = 2.0  → perplexity = e^2.0 = 7.39
  loss = 3.0  → perplexity = e^3.0 = 20.09
  loss = 0.5  → perplexity = e^0.5 = 1.65
```

### Interpretation

```
Perplexity of N means: "On average, the model is as uncertain as if it
were choosing uniformly between N tokens at each step."

perplexity = 1:     Perfect prediction (impossible in practice)
perplexity = 10:    Model narrows choices to ~10 plausible tokens
perplexity = 100:   Model is quite uncertain
perplexity = 50000: Random guessing (vocab_size = 256,128 for Gemma)

For fine-tuned LLMs:
  perplexity < 5:    Excellent (possibly overfitting if too low)
  perplexity 5-15:   Very good  
  perplexity 15-50:  Decent
  perplexity > 50:   Model hasn't learned the task well
```

### When to Use

- ✅ During training (automatic — reported as loss, which is log-perplexity)
- ✅ Quick quality check on test set
- ❌ Not great for comparing generated text quality

---

## ROUGE: Recall-Oriented Evaluation

### What Is ROUGE?

ROUGE (**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation) measures overlap between generated text and reference text. Originally designed for summarization.

### ROUGE-1: Unigram Overlap

Counts matching individual words.

```
Reference: "buy this great product for everyday use"
Generated: "buy this amazing product for daily use"

Matching words: {buy, this, product, for, use} = 5 matches

Precision = matches / generated_words = 5/7 = 0.71
  "What fraction of generated words appear in the reference?"

Recall = matches / reference_words = 5/7 = 0.71
  "What fraction of reference words appear in the generated text?"

F1 = 2 × (Precision × Recall) / (Precision + Recall) = 0.71
  We report F1 (harmonic mean of precision and recall)
```

### ROUGE-2: Bigram Overlap

Counts matching word PAIRS (captures phrase structure).

```
Reference: "buy this great product for everyday use"
Reference bigrams: {"buy this", "this great", "great product",
                     "product for", "for everyday", "everyday use"}

Generated: "buy this amazing product for daily use"
Generated bigrams: {"buy this", "this amazing", "amazing product",
                     "product for", "for daily", "daily use"}

Matching bigrams: {"buy this", "product for"} = 2 matches

ROUGE-2 Precision = 2/6 = 0.33
ROUGE-2 Recall = 2/6 = 0.33
ROUGE-2 F1 = 0.33
```

### ROUGE-L: Longest Common Subsequence

Finds the longest sequence of words that appear in the same order in both texts (not necessarily consecutive).

```
Reference: "buy this great product for everyday use"
Generated: "buy this amazing product for daily use"

Longest Common Subsequence: "buy this product for use" (length 5)
  Position in ref: buy(1) this(2) ___ product(4) for(5) ___ use(7)
  Position in gen: buy(1) this(2) ___ product(4) for(5) ___ use(7)

ROUGE-L captures sequential structure better than ROUGE-1
It rewards outputs that follow the same logical flow, even if
individual words differ.

ROUGE-L = based on LCS length relative to reference and generated lengths
```

### Stemming

We enable stemming in our scorer:
```python
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                   use_stemmer=True)

Stemming: "running" → "run", "products" → "product"

With stemming, "products" and "product" are counted as a match.
This gives more meaningful scores for text generation tasks.
```

---

## BLEU: Precision-Oriented Evaluation

### What Is BLEU?

BLEU (**B**ilingual **E**valuation **U**nderstudy) measures n-gram precision: what fraction of the generated n-grams appear in the reference.

### Step-by-Step Computation

```
Reference: "the cat sat on the mat"
Generated: "the the the the"

Step 1: Count n-gram matches for n=1,2,3,4

  Unigrams (n=1):
    Generated: {"the": 4}
    Reference: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
    Clipped matches: min(4, 2) = 2  (can't match more than ref has)
    Precision_1 = 2/4 = 0.50

  Bigrams (n=2):
    Generated: {"the the": 3}
    Reference: {"the cat": 1, "cat sat": 1, "sat on": 1, "on the": 1, "the mat": 1}
    Clipped matches: 0
    Precision_2 = 0/3 = 0.00

  Trigrams (n=3):
    Precision_3 = 0

  4-grams (n=4):
    Precision_4 = 0

Step 2: Geometric mean (with smoothing)
  BLEU = (p1 × p2 × p3 × p4)^(1/4)
  BLEU ≈ 0 (because p2, p3, p4 are 0)

Step 3: Brevity Penalty (BP)
  If generated is shorter than reference:
    BP = exp(1 - ref_length/gen_length) = exp(1 - 6/4) = exp(-0.5) = 0.61
  If generated is longer:
    BP = 1.0

Final BLEU = BP × geometric_mean = 0.61 × 0 ≈ 0
```

### A Better Example

```
Reference: "buy this great product for everyday use"
Generated: "buy this product for daily use"

Unigram precision: 5/6 = 0.83
Bigram precision:  2/5 = 0.40  {"buy this", "product for"}
Trigram precision: 1/4 = 0.25  {"buy this product"? no — "this product for"? no}
                   Actually checking: 0/4 = 0.00
4-gram precision:  0/3 = 0.00

With smoothing (avoids zero):
BLEU ≈ 0.15 (smoothed)

Brevity penalty: gen=6, ref=7 → BP = exp(1-7/6) = 0.85

Final BLEU ≈ 0.85 × 0.15 = 0.13
```

### Corpus BLEU vs Sentence BLEU

```
Sentence BLEU: Compute BLEU for each pair, then average
  Problem: Short sentences get BLEU=0 easily (4-gram = 0)
  Results are unstable

Corpus BLEU: Aggregate ALL n-gram counts across all pairs, compute once
  More stable and reliable
  Standard in research

We use CORPUS BLEU (the standard).
```

### Smoothing

```python
smoothing = SmoothingFunction().method4

Without smoothing:
  If any n-gram precision = 0, entire BLEU = 0
  This happens often for individual short sentences

With smoothing (method 4 — exponential):
  Adds small pseudo-counts to prevent zeros
  BLEU is still low for bad outputs, but not exactly 0
```

---

## ROUGE vs BLEU: When to Use Which

| Metric | Measures | Best For | Emphasis |
|--------|----------|----------|----------|
| ROUGE | Recall (coverage) | Summarization, paraphrasing | "Did the output cover all key info?" |
| BLEU | Precision (accuracy) | Translation, structured output | "Is the output accurate and precise?" |
| Both | Different perspectives | **Our approach — use both** | Complete picture |

```
High ROUGE, Low BLEU:
  → Output covers all key points but adds lots of extra text
  → May be verbose or rambling

Low ROUGE, High BLEU:
  → Output is precise but incomplete
  → May be too short or miss important information

High ROUGE, High BLEU:
  → Output is both comprehensive AND precise ← GOAL

Low ROUGE, Low BLEU:
  → Output is neither comprehensive nor precise ← MODEL NEEDS WORK
```

### For Our Task (Product Recommendation)

```
ROUGE is more relevant because:
  - Recommendations can be phrased many valid ways
  - We care more about COVERAGE (did it mention strengths/weaknesses?)
  - The exact wording doesn't need to match

BLEU is still useful because:
  - We want structured output (recommendation, reasoning, strategy)
  - Structural keywords (BUY/SKIP/CONSIDER) should be present
```

---

## Human Evaluation

### Why Automated Metrics Aren't Enough

```
Reference: "Recommendation: BUY. This phone has excellent battery life."
Generated 1: "Recommendation: BUY. This phone has great battery performance."
Generated 2: "Recommendation: BUY. great battery excellent phone this has."

ROUGE-1 scores:
  Generated 1: 0.70 (good output)
  Generated 2: 0.70 (gibberish but same word overlap!)

A human would immediately see Generated 2 is nonsensical.
```

### Manual Evaluation Criteria

| Criterion | Question to Ask | Scale |
|-----------|----------------|-------|
| **Relevance** | Is the recommendation related to the review? | 1-5 |
| **Coherence** | Does the text flow logically? | 1-5 |
| **Completeness** | Does it cover strengths, weaknesses, strategy? | 1-5 |
| **Correctness** | Is the BUY/SKIP/CONSIDER appropriate for the rating? | Yes/No |
| **Helpfulness** | Would this be useful to a real consumer/brand? | 1-5 |

### When to Do Human Evaluation

- After achieving decent automated scores (ROUGE-L > 0.2)
- Before deploying to production
- When comparing two fine-tuned models with similar automated scores

---

## Interpreting Scores

### Score Ranges for Our Task

```
ROUGE-1    │ Quality        │ What It Means
───────────┼────────────────┼──────────────────────────
> 0.5      │ 🟢 Excellent   │ Output closely matches expected format
0.3 - 0.5  │ 🟡 Good        │ Captures main themes, some differences
0.15 - 0.3 │ 🟠 Moderate    │ Partially learned, needs more training
< 0.15     │ 🔴 Poor        │ Model hasn't learned the task yet

ROUGE-L    │ Quality
───────────┼────────────────
> 0.4      │ 🟢 Excellent
0.25 - 0.4 │ 🟡 Good
0.1 - 0.25 │ 🟠 Moderate
< 0.1      │ 🔴 Poor

BLEU       │ Quality
───────────┼────────────────
> 0.3      │ 🟢 Excellent
0.15 - 0.3 │ 🟡 Good
0.05 - 0.15│ 🟠 Moderate
< 0.05     │ 🔴 Poor
```

### Important Caveats

1. **Scores are relative** — compare between YOUR runs, not to other tasks
2. **Creative tasks get lower scores** — many valid phrasings = lower overlap
3. **Template-based references inflate scores** — our generated references have a fixed structure
4. **Use multiple metrics** — no single metric tells the full story

---

## Our Evaluation Pipeline

```python
# In evaluate.py, we run:

1. Load fine-tuned model
2. Load test dataset (held out during training)
3. For each test example:
   a. Extract review + rating from formatted prompt
   b. Generate recommendation using the model
   c. Compare against template reference
4. Compute ROUGE-1, ROUGE-2, ROUGE-L, BLEU
5. Print results with interpretation
6. Save to JSON file

# Usage:
python evaluate.py --model_dir ./outputs/run_xxx/final_model --max_samples 50
```
