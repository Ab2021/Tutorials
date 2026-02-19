# 7. Transfer Learning Theory — Standing on the Shoulders of Giants

## Table of Contents
- [What Is Transfer Learning?](#what-is-transfer-learning)
- [Why Transfer Learning Works](#why-transfer-learning-works)
- [Types of Transfer Learning](#types-of-transfer-learning)
- [Catastrophic Forgetting](#catastrophic-forgetting)
- [Domain Adaptation](#domain-adaptation)
- [The Pre-training → Fine-tuning Paradigm](#the-pre-training--fine-tuning-paradigm)
- [Feature Extraction vs Fine-Tuning](#feature-extraction-vs-fine-tuning)
- [Continual Learning](#continual-learning)
- [Practical: When Transfer Fails](#practical-when-transfer-fails)

---

## What Is Transfer Learning?

### Definition

Transfer learning means using knowledge learned from **Task A** to improve performance on **Task B**.

```
Traditional ML:
  Task A: Train model from scratch on Task A data
  Task B: Train DIFFERENT model from scratch on Task B data
  No knowledge shared between tasks!

Transfer Learning:
  Task A: Train model on Task A data (large, generic)
  Task B: Use Task A model as starting point, adapt to Task B (small, specific)
  Task A knowledge TRANSFERS to Task B!

In our project:
  Task A = Language modeling (predicting next token, done by Google, trillions of tokens)
  Task B = Product recommendation (our task, thousands of examples)
  
  Gemma's language knowledge transfers to recommendation generation!
```

### Real-World Analogy

```
Without transfer: Teaching someone physics from birth
  - Learn alphabet → basic math → algebra → calculus → physics
  - Takes 20 years!

With transfer: Teaching a math PhD student physics
  - They already know calculus (transferred knowledge)
  - Just teach the physics-specific parts
  - Takes months!

Our case:
  Gemma already knows English, reasoning, product concepts (pre-training)
  We just teach it the specific recommendation format (fine-tuning)
```

---

## Why Transfer Learning Works

### Hierarchical Feature Learning

```
Layer 1 (earliest): Basic features
  Language: character patterns, word boundaries, punctuation rules
  These are UNIVERSAL across all language tasks

Layer 6 (early-mid): Syntactic features
  Subject-verb agreement, phrase structure, dependency parsing
  Mostly universal across language tasks

Layer 12 (mid): Semantic features
  Word meaning in context, sentiment, entity types
  Somewhat task-specific but broadly useful

Layer 18 (deepest): Task-specific features
  Review quality assessment, recommendation patterns
  Highly task-specific

KEY INSIGHT:
  Early layers learn GENERAL features → transfer well
  Late layers learn SPECIFIC features → need adaptation

  LoRA adds adapters to ALL layers, but:
  - Early layer adapters learn small adjustments (general features mostly unchanged)
  - Late layer adapters learn larger adjustments (task-specific)
```

### Mathematical Perspective

```
Pre-trained model: f(x; θ₀)
Fine-tuned model: f(x; θ₀ + Δθ)

If Δθ is small relative to θ₀ (which LoRA enforces):
  f(x; θ₀ + Δθ) ≈ f(x; θ₀) + J(x; θ₀) × Δθ  (first-order Taylor)

The fine-tuned model is a LINEAR PERTURBATION of the pre-trained model.
  - Pre-trained model provides the base prediction
  - LoRA adapters provide task-specific corrections
  
This preserves pre-trained knowledge while adding new capabilities.
```

---

## Types of Transfer Learning

### 1. Inductive Transfer (Our Approach)

```
Source task: Next-token prediction on web text
Target task: Product recommendation from reviews

Different tasks, but the knowledge (language understanding) transfers.

Method: Fine-tune with labeled target data (supervised).
```

### 2. Transductive Transfer (Domain Adaptation)

```
Source domain: Product reviews in English
Target domain: Product reviews in French

Same task, different distributions.

Method: Adapt the model to the new domain, possibly unsupervised.
```

### 3. Unsupervised Transfer (Feature Reuse)

```
Pre-train: Learn general text embeddings
Apply: Use embeddings as features for downstream classifier

No fine-tuning of the pre-trained model.
Method: Feature extraction (freeze model, train classifier on top).
```

---

## Catastrophic Forgetting

### The Problem

```
Before fine-tuning:
  Gemma knows: English, math, science, coding, common sense, ...
  
After AGGRESSIVE fine-tuning:
  Gemma knows: Product recommendations!
  Gemma FORGOT: Math, coding, common sense, ...

This is "catastrophic forgetting" — new knowledge overwrites old.
```

### Why It Happens

```
Neural network weights are SHARED across all tasks.
Updating weights for Task B modifies the SAME weights used for Task A.

Analogy: Imagine your brain has limited capacity.
  Learning to play guitar might make you forget piano
  (if they used the same neural circuits).
```

### How LoRA Prevents Catastrophic Forgetting

```
1. FROZEN BASE MODEL:
   100% of pre-trained weights remain unchanged.
   All language knowledge is perfectly preserved.
   LoRA ONLY adds new parameters — doesn't modify old ones.

2. LOW-RANK CONSTRAINT:
   LoRA adapters have very few parameters (0.13%).
   The adaptation is a SMALL perturbation to the model's behavior.
   Original capabilities are mostly preserved.

3. ZERO INITIALIZATION:
   At the start of training, LoRA output = 0.
   The model starts as EXACTLY the pre-trained model.
   Changes happen gradually during training.
```

### Measuring Forgetting

```
Before fine-tuning:
  General benchmark (MMLU): 45.2%
  Product recommendation: 12.0% (no training yet)

After LoRA fine-tuning:
  General benchmark (MMLU): 44.8%  (only -0.4% drop!)
  Product recommendation: 78.5%    (+66.5% improvement!)

After FULL fine-tuning (for comparison):
  General benchmark (MMLU): 38.1%  (-7.1% drop — significant!)
  Product recommendation: 80.2%    (slightly better, but at what cost?)

LoRA preserves 99.1% of general knowledge.
Full fine-tuning loses 15.7% of general knowledge.
```

---

## Domain Adaptation

### When Domains Differ

```
Source domain (pre-training): Wikipedia, web text, books
  Vocabulary: standard English
  Style: informative, formal
  Topics: everything

Target domain (fine-tuning): Product reviews
  Vocabulary: informal, abbreviations, slang, brand names
  Style: casual, emotional, opinionated
  Topics: consumer electronics, beauty products, etc.

The model must ADAPT from general text to review-specific text.
```

### Levels of Domain Gap

```
Small gap (easy transfer):
  Wikipedia articles → News articles
  English books → English blog posts
  
Medium gap (moderate transfer):
  Web text → Product reviews        ← Our case
  English articles → Code comments
  
Large gap (difficult transfer):
  English text → Medical records
  Web text → Legal contracts
  Text → Chemical formulas

The larger the gap, the more training data and higher LoRA rank you need.
```

---

## The Pre-training → Fine-tuning Paradigm

### The Modern ML Pipeline

```
Phase 1: PRE-TRAINING (done by Google/Meta/etc.)
  Data: Trillions of tokens
  Cost: $1-100 million
  Time: Weeks-months
  Hardware: Thousands of GPUs
  Result: Foundation model

Phase 2: SUPERVISED FINE-TUNING (SFT) — What we do
  Data: Thousands of (input, output) pairs
  Cost: $5-50
  Time: Hours
  Hardware: 1 GPU
  Result: Task-specific model

Phase 3: ALIGNMENT (optional, advanced)
  Data: Human preference rankings
  Method: RLHF or DPO
  Result: Model aligned with human preferences

Phase 4: INFERENCE
  Deploy the fine-tuned model
  Generate predictions on new inputs
```

---

## Feature Extraction vs Fine-Tuning

### Feature Extraction (Freeze Everything)

```python
# Feature extraction: use model as a fixed feature extractor
model = load_pretrained_model()
for param in model.parameters():
    param.requires_grad = False  # Freeze ALL weights

# Add a new classification head
classifier = nn.Linear(2048, num_classes)  # Only this is trained

# Forward: model generates features, classifier makes predictions
features = model(input)[-1]  # Use last hidden state
prediction = classifier(features)
```

**When to use:**
- Very small dataset (< 100 examples)
- Extremely limited compute
- Quick baseline

### Full Fine-Tuning

```python
# Full fine-tuning: update EVERY parameter
model = load_pretrained_model()
# All parameters already have requires_grad=True

optimizer = AdamW(model.parameters(), lr=2e-5)  # Very small LR!
```

**When to use:**
- Massive dataset (> 100K examples)
- Abundant GPU memory (> 40 GB)
- Need maximum quality

### LoRA Fine-Tuning (Our Approach)

```python
# LoRA: freeze base, train small adapters
model = load_pretrained_model()
model = apply_lora(model, r=16)
# Only LoRA parameters have requires_grad=True

optimizer = AdamW(model.parameters(), lr=2e-4)  # Can use larger LR
```

**When to use:**
- Most scenarios (default choice)
- Consumer GPUs (8-24 GB)
- Moderate dataset size (1K-50K examples)

---

## Continual Learning

### The Challenge

```
After fine-tuning on product reviews, what if we want to ALSO
fine-tune on customer support conversations?

Naive approach: Fine-tune sequentially
  Step 1: Train on reviews → Model A (good at reviews)
  Step 2: Train Model A on support → Model B (good at support, bad at reviews!)
  
  Catastrophic forgetting strikes again!
```

### Solutions

```
1. MULTI-TASK LEARNING:
   Train on reviews AND support data SIMULTANEOUSLY.
   Model learns both tasks at once.
   ❌ Need all data available at the same time.

2. ELASTIC WEIGHT CONSOLIDATION (EWC):
   Penalize changes to weights that are important for previous tasks.
   ✅ Can learn sequentially.
   ❌ Complex to implement, scales poorly.

3. MULTIPLE LoRA ADAPTERS (Recommended):
   Keep one base model + separate LoRA adapters per task:
   
   Base Gemma + Reviews LoRA  → Product recommender
   Base Gemma + Support LoRA  → Support assistant
   Base Gemma + Code LoRA     → Code helper
   
   ✅ No forgetting (each adapter is independent)
   ✅ Easy to manage (swap adapters for different tasks)
   ✅ Efficient (only store small adapters, not full models)
```

---

## Practical: When Transfer Fails

### Signs That Transfer Isn't Working

```
1. Val loss doesn't improve from epoch 1:
   → Domain gap may be too large
   → Fix: More training data, higher rank, pre-fine-tune on domain text

2. Model outputs are in the wrong "style":
   → Pre-trained model's style dominates
   → Fix: More diverse training examples, more epochs

3. Model hallucinates product features:
   → Pre-trained knowledge is overriding task-specific input
   → Fix: Clearer prompt format, explicit instruction to use ONLY the review

4. Performance is worse than random:
   → Something is fundamentally wrong (data format, tokenizer mismatch)
   → Fix: Verify data pipeline, check tokenizer compatibility
```

### Transfer Learning Best Practices

```
Practice                           │ Why
───────────────────────────────────┼─────────────────────────────
Use LR 10-100× smaller than       │ Pre-trained weights are already good,
training from scratch              │ small adjustments are sufficient
                                   │
Freeze early layers (or use LoRA)  │ Early layers capture universal features,
                                   │ only adapt later layers
                                   │
Use the SAME tokenizer as          │ Different tokenizer = different token IDs
pre-training                       │ = model can't understand input
                                   │
Match the pre-training data format │ Gemma expects <start_of_turn> format
if possible                        │ Using it → model already understands
                                   │
Start with the smallest model that │ Smaller models are faster to iterate
achieves acceptable quality       │ Only scale up when needed
```
