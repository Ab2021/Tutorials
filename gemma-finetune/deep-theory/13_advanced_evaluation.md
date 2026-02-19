# 13. Advanced Evaluation — Beyond ROUGE and BLEU

## Table of Contents
- [Limitations of ROUGE and BLEU](#limitations-of-rouge-and-bleu)
- [BERTScore: Semantic Similarity](#bertscore-semantic-similarity)
- [METEOR: Improved Translation Metric](#meteor-improved-translation-metric)
- [Perplexity: Model Confidence](#perplexity-model-confidence)
- [Human Evaluation Frameworks](#human-evaluation-frameworks)
- [LLM-as-Judge](#llm-as-judge)
- [Task-Specific Metrics](#task-specific-metrics)
- [Statistical Significance](#statistical-significance)
- [Practical: Multi-Metric Evaluation Pipeline](#practical-multi-metric-evaluation-pipeline)

---

## Limitations of ROUGE and BLEU

```
ROUGE and BLEU are LEXICAL metrics — they compare WORDS, not MEANING.

Example that breaks lexical metrics:
  Reference:  "This phone is fantastic, definitely purchase it!"
  Generated1: "This phone is great, definitely buy it!"        ← GOOD output
  Generated2: "phone great purchase this it definitely is!"    ← GIBBERISH

  ROUGE-1 scores:
    Generated1: 0.71  (misses "fantastic" and "purchase")
    Generated2: 0.71  (same individual words, but scrambled!)

  Both get the SAME score despite Generated2 being nonsensical.

Other failure modes:
  Synonyms:  "happy" vs "joyful" → score as non-matching
  Negation:  "BUY this" vs "Don't BUY this" → high overlap!
  Paraphrase: "The battery is excellent" vs "Great power cell" → low overlap
```

---

## BERTScore: Semantic Similarity

### How It Works

```
Instead of matching words, match EMBEDDINGS (semantic representations):

Reference tokens:  ["This", "phone", "is", "fantastic"]
Generated tokens:  ["This", "phone", "is", "great"]

Standard ROUGE: "fantastic" ≠ "great" → no match

BERTScore:
  1. Get BERT embeddings for each token:
     embed("fantastic") = [0.23, 0.87, -0.12, ...]
     embed("great")     = [0.21, 0.85, -0.10, ...]
  
  2. Compute cosine similarity between ALL pairs:
            fantastic  great
     fantastic  1.00    0.92  ← HIGH similarity!
     great      0.92    1.00
  
  3. Greedy match: each generated token matches with its most
     similar reference token
     
  4. Precision: avg similarity of generated→reference matches
     Recall: avg similarity of reference→generated matches
     F1: harmonic mean
```

### Practical Example

```
Reference:  "Recommendation: BUY. This phone has excellent battery life."
Generated:  "Suggestion: PURCHASE. This phone has outstanding battery duration."

ROUGE-1 F1: ~0.50 (many words don't match)
BERTScore F1: ~0.92 (semantically almost identical!)

BERTScore correctly captures that "Recommendation"≈"Suggestion",
"BUY"≈"PURCHASE", "excellent"≈"outstanding", "life"≈"duration"
```

### Implementation

```python
from bert_score import score

refs = ["This phone has excellent battery life"]
cands = ["This phone has outstanding battery duration"]

P, R, F1 = score(cands, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli")
print(f"BERTScore P={P.item():.3f} R={R.item():.3f} F1={F1.item():.3f}")
# BERTScore P=0.94 R=0.93 F1=0.93

# Use rescale_with_baseline for interpretable absolute scores:
P, R, F1 = score(cands, refs, lang="en", rescale_with_baseline=True)
# Rescaled: 0 = random, 1 = identical
```

---

## METEOR: Improved Translation Metric

### Beyond Exact Matches

```
METEOR (Metric for Evaluation of Translation with Explicit ORdering)
matches words using multiple strategies:

Level 1: Exact match        "buy" = "buy"                     ✅
Level 2: Stem match          "buying" = "buy" (same stem)      ✅
Level 3: Synonym match      "purchase" = "buy" (WordNet)       ✅
Level 4: Paraphrase match   "acquire" = "buy" (paraphrase DB)  ✅

BLEU: Only Level 1 → misses most linguistic variation
METEOR: All 4 levels → captures meaning much better
```

### METEOR's Penalty for Reordering

```
METEOR also penalizes SCRAMBLED but matching content:

Reference:  "The cat sat on the mat"
Generated1: "The cat sat on the mat"      → penalty = 0 (same order)
Generated2: "on the sat mat the cat"      → penalty = 0.5 (reordered)

Penalty = 0.5 × (chunks / matched_words)^3
Where chunks = number of contiguous matched groups

Generated1: [The cat sat on the mat] → 1 chunk  → penalty low
Generated2: [on] [the] [sat] [mat] [the cat] → 5 chunks → penalty high
```

---

## Perplexity: Model Confidence

### For Evaluating Language Models

```
Perplexity measures how well the model PREDICTS the reference text:

perplexity = exp(-1/N × Σ log P(tokenᵢ | context))

Lower perplexity = model assigns higher probability to correct tokens
= model is more confident and accurate

Computing perplexity of the REFERENCE using our model:
  Input the reference text
  At each position, check: what probability did the model assign
  to the ACTUAL next token?
  
  If model is well-trained: high probability → low perplexity
  If model is poorly trained: low probability → high perplexity
```

### Practical Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

# Example:
ppl = compute_perplexity(model, tokenizer,
    "Recommendation: BUY. This phone has excellent battery life.")
print(f"Perplexity: {ppl:.2f}")
# Good model: ppl ≈ 3-10
# Bad model:  ppl > 50
```

---

## Human Evaluation Frameworks

### The Gold Standard

```
No automated metric perfectly captures output QUALITY.
Human evaluation is the ultimate test.

Framework 1: ABSOLUTE SCORING
  Rate each output on a scale of 1-5:
    1 = Completely wrong/incoherent
    2 = Partially relevant but major issues
    3 = Acceptable but could be better
    4 = Good, minor issues
    5 = Excellent, could not improve

Framework 2: PAIRWISE COMPARISON
  "Which output is better, A or B?"
  More reliable than absolute scoring.
  Avoids calibration differences between raters.

Framework 3: BEST-WORST SCALING
  Show 4 outputs. "Which is BEST? Which is WORST?"
  Efficient: each comparison gives 2 data points.
```

### Designing An Evaluation Study

```
Step 1: Define criteria (for product recommendations)
  ┌──────────────┬─────────────────────────────────────┬───────┐
  │ Criterion    │ Question                            │ Scale │
  ├──────────────┼─────────────────────────────────────┼───────┤
  │ Relevance    │ Is it about the right product?      │ 1-5   │
  │ Accuracy     │ Does BUY/SKIP match the rating?     │ Y/N   │
  │ Completeness │ Does it cover pros/cons/strategy?   │ 1-5   │
  │ Coherence    │ Is it well-written and logical?     │ 1-5   │
  │ Helpfulness  │ Would this help a real user/brand?  │ 1-5   │
  └──────────────┴─────────────────────────────────────┴───────┘

Step 2: Select test examples (50-100 diverse examples)

Step 3: Recruit evaluators (at least 3 per example for reliability)

Step 4: Compute Inter-Annotator Agreement
  Cohen's Kappa (2 annotators) or Fleiss' Kappa (3+)
  κ > 0.60 = acceptable agreement
  κ > 0.80 = strong agreement
```

---

## LLM-as-Judge

### Using AI to Evaluate AI

```
Instead of (expensive) human evaluation, use a STRONG LLM to judge:

Prompt for GPT-4 / Claude as evaluator:
  "You are evaluating the quality of a product recommendation.
   
   REVIEW: {review}
   RATING: {rating}
   
   MODEL OUTPUT: {model_output}
   
   Rate from 1-5 on:
   1. Relevance (is it about this product?)
   2. Accuracy (correct recommendation for the rating?)
   3. Helpfulness (useful for consumers/brands?)
   
   Provide scores and brief justification."

Benefits:
  ✅ Cheap and fast ($0.01/evaluation vs $1/human)
  ✅ Consistent (no annotator fatigue)
  ✅ Scalable (can evaluate thousands of outputs)

Limitations:
  ❌ May have systematic biases
  ❌ May favor verbose outputs
  ❌ Correlation with human judgment is ~0.7-0.8 (not perfect)
```

---

## Task-Specific Metrics

### For Product Recommendation

```
Classification Accuracy:
  Does the model output BUY/SKIP/CONSIDER correctly for the rating?
  
  Rating 4-5 → should output BUY
  Rating 3   → should output CONSIDER
  Rating 1-2 → should output SKIP
  
  accuracy = correct_recommendations / total_examples

Format Compliance:
  Does the output follow the expected structure?
  Check for:
    ✅ "Recommendation:" label present
    ✅ Strengths section present
    ✅ Strategy section present
  
  compliance = well_formatted / total_examples

Information Extraction Accuracy:
  Does the model correctly identify product features from the review?
  Compare extracted features against manually labeled features.
```

---

## Statistical Significance

### Why You Need It

```
Scenario:
  Model A ROUGE-L: 0.35
  Model B ROUGE-L: 0.37

Is Model B actually better, or is this just random noise?
Without statistical testing, you can't tell!
```

### Bootstrap Confidence Intervals

```python
import numpy as np

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(scores), lower, upper

# Example:
rouge_scores = [0.32, 0.41, 0.28, 0.45, 0.35, ...]
mean, lower, upper = bootstrap_ci(rouge_scores)
print(f"ROUGE-L: {mean:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
# ROUGE-L: 0.350 (95% CI: [0.312, 0.388])
```

### Paired Bootstrap Test

```python
def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    """Test if model B is significantly better than model A."""
    count_b_better = 0
    n = len(scores_a)
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample_a = np.mean([scores_a[i] for i in indices])
        sample_b = np.mean([scores_b[i] for i in indices])
        if sample_b > sample_a:
            count_b_better += 1
    
    p_value = 1 - count_b_better / n_bootstrap
    return p_value

# Usage:
p = paired_bootstrap_test(model_a_scores, model_b_scores)
print(f"p-value: {p:.4f}")
# p < 0.05 → Model B is significantly better
# p > 0.05 → Difference is not statistically significant
```

---

## Practical: Multi-Metric Evaluation Pipeline

```python
"""
Complete evaluation pipeline using multiple metrics.
"""
import torch
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# pip install bert-score  (optional, for BERTScore)

class MultiMetricEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method4
    
    def evaluate(self, references, predictions):
        results = {}
        
        # ROUGE scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for ref, pred in zip(references, predictions):
            scores = self.rouge.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        for key in rouge_scores:
            mean, lower, upper = self._bootstrap_ci(rouge_scores[key])
            results[key] = {"mean": mean, "ci_lower": lower, "ci_upper": upper}
        
        # BLEU score
        refs_tokenized = [[ref.split()] for ref in references]
        preds_tokenized = [pred.split() for pred in predictions]
        bleu = corpus_bleu(refs_tokenized, preds_tokenized,
                          smoothing_function=self.smoothing)
        results["bleu"] = {"mean": bleu}
        
        # Format compliance
        compliance = self._check_format_compliance(predictions)
        results["format_compliance"] = {"mean": compliance}
        
        return results
    
    def _check_format_compliance(self, predictions):
        compliant = 0
        for pred in predictions:
            pred_lower = pred.lower()
            has_recommendation = "recommendation:" in pred_lower
            has_keyword = any(w in pred_lower for w in ["buy", "skip", "consider"])
            if has_recommendation and has_keyword:
                compliant += 1
        return compliant / len(predictions) if predictions else 0
    
    def _bootstrap_ci(self, scores, n=1000):
        scores = np.array(scores)
        means = [np.mean(np.random.choice(scores, len(scores), replace=True))
                 for _ in range(n)]
        return np.mean(scores), np.percentile(means, 2.5), np.percentile(means, 97.5)
    
    def print_results(self, results):
        print("\n" + "="*60)
        print("  EVALUATION RESULTS (Multi-Metric)")
        print("="*60)
        for metric, values in results.items():
            if "ci_lower" in values:
                print(f"  {metric:20s}: {values['mean']:.4f} "
                      f"(95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}])")
            else:
                print(f"  {metric:20s}: {values['mean']:.4f}")
        print("="*60)

# Usage:
evaluator = MultiMetricEvaluator()
results = evaluator.evaluate(references, predictions)
evaluator.print_results(results)
```
