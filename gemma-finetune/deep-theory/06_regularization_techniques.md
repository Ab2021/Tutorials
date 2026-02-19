# 6. Regularization Techniques — Preventing Overfitting

## Table of Contents
- [What Is Overfitting?](#what-is-overfitting)
- [Dropout: Random Neuron Silencing](#dropout-random-neuron-silencing)
- [Weight Decay (L2 Regularization)](#weight-decay-l2-regularization)
- [Early Stopping](#early-stopping)
- [Data Augmentation for NLP](#data-augmentation-for-nlp)
- [Label Smoothing](#label-smoothing)
- [Gradient Noise](#gradient-noise)
- [LoRA Dropout: Our Regularization Strategy](#lora-dropout-our-regularization-strategy)
- [Regularization Decision Framework](#regularization-decision-framework)

---

## What Is Overfitting?

### The Problem

```
Training data:  10 reviews like "Great battery!" → BUY
Test data:      New review "Excellent battery performance" → ???

Overfitting: Model memorizes EXACT training phrases.
  "Great battery!" → BUY     ✅ (seen during training)
  "Excellent battery" → ???  ❌ (never saw this exact phrase)

Generalizing: Model learns the CONCEPT.
  High rating + positive words about battery → BUY  ✅ (any phrasing works)
```

### The Bias-Variance Trade-off

```
Model Complexity ▶

Error ▲
      │     Total Error
      │    ╱╲
      │   ╱  ╲              ╱  Variance (overfitting)
      │  ╱    ╲            ╱
      │ ╱      ╲          ╱
      │╱        ╲________╱
      │          ╲ Sweet Spot
      │           ╲
      │            ╲╲╲╲    Bias (underfitting)
      └──────────────────────────────▶ Complexity

Underfitting (high bias):    Too simple → misses patterns
Sweet spot:                  Just right → learns real patterns without memorizing
Overfitting (high variance): Too complex → memorizes training data
```

---

## Dropout: Random Neuron Silencing

### How It Works

```
During TRAINING: Randomly set p% of neuron outputs to zero.

Layer output: [0.5, 0.3, 0.8, 0.2, 0.7, 0.1]
Dropout mask: [1,   0,   1,   0,   1,   1  ]   (p=0.33, ~33% dropped)
After drop:   [0.5, 0.0, 0.8, 0.0, 0.7, 0.1]

Then SCALE surviving values by 1/(1-p) to maintain expected magnitude:
Scaled:       [0.75, 0.0, 1.2, 0.0, 1.05, 0.15]

During INFERENCE: No dropout (use all neurons).
```

### Why It Works

```
1. ENSEMBLE EFFECT:
   Each forward pass uses a different random subset of neurons.
   This is like training many different sub-networks.
   The final model is an ensemble of these sub-networks.

2. PREVENTS CO-ADAPTATION:
   Without dropout, neurons can become dependent on each other:
     Neuron A always relies on Neuron B to detect sentiment.
     If B fails at test time, A fails too.
   
   With dropout, each neuron must learn to be useful INDEPENDENTLY.
   More robust features emerge.

3. IMPLICIT REGULARIZATION:
   Dropout adds noise → smoother loss landscape → flatter minima
   → better generalization.
```

### Practical: LoRA Dropout

```python
# In our config:
lora_dropout = 0.05  # 5% of LoRA activations dropped

# This is applied BETWEEN the LoRA A and B matrices:
# lora_output = dropout(x @ A) @ B × (alpha/r)

# 5% is conservative — good starting point.
# If overfitting: increase to 0.1-0.2
# If underfitting: decrease to 0.0
```

---

## Weight Decay (L2 Regularization)

### The Idea

Add a penalty for large weights to the loss:

```
L_total = L_data + (λ/2) × Σᵢ θᵢ²

The derivative adds a term:
  ∂L_total/∂θᵢ = ∂L_data/∂θᵢ + λ × θᵢ

Update rule:
  θᵢ = θᵢ - η × (∂L_data/∂θᵢ + λ × θᵢ)
  θᵢ = (1 - ηλ) × θᵢ - η × ∂L_data/∂θᵢ
         ↑ weights "decay" toward 0 by factor (1 - ηλ) each step

In our config: weight_decay = 0.01
  Each step, weights shrink by factor (1 - 2e-4 × 0.01) = 0.999998
  Very gentle decay — prevents weights from growing too large
  without significantly constraining learning.
```

### Why Smaller Weights Generalize Better

```
Large weights → model is very sensitive to small input changes
  W = [100, -50, 200] → small change in x produces HUGE output change
  This captures noise in training data, not real patterns.

Small weights → model makes smooth, gradual predictions
  W = [0.5, -0.3, 0.8] → small change in x produces proportional output change
  This captures underlying trends, not noise.
```

---

## Early Stopping

### The Simplest Regularization

```
Monitor validation loss during training.
Stop when it starts INCREASING.

Step  │ Train Loss │ Val Loss │ Action
──────┼────────────┼──────────┼───────────
100   │ 2.50       │ 2.55     │ Continue
200   │ 2.00       │ 2.05     │ Continue
300   │ 1.50       │ 1.55     │ Continue
400   │ 1.00       │ 1.10     │ Continue
500   │ 0.70       │ 0.90     │ Val increasing... patience=1
600   │ 0.50       │ 0.95     │ Val still increasing... patience=2
700   │ 0.30       │ 1.05     │ STOP! Use checkpoint from step 400.

We use the best_model checkpoint from HuggingFace's Trainer:
  load_best_model_at_end = True
  metric_for_best_model = "loss"
```

---

## Data Augmentation for NLP

### Techniques

```
1. SYNONYM REPLACEMENT:
   Original: "This phone has amazing battery life"
   Augmented: "This phone has incredible battery duration"
   
2. RANDOM INSERTION:
   Original: "Great product, highly recommend"
   Augmented: "Great excellent product, highly recommend"

3. RANDOM SWAP:
   Original: "The camera quality is superb"
   Augmented: "The quality camera is superb"

4. RANDOM DELETION:
   Original: "I absolutely love this amazing product"
   Augmented: "I love this amazing product"

5. BACK-TRANSLATION:
   Original: "The battery lasts all day"
   → Translate to French: "La batterie dure toute la journée"
   → Translate back: "The battery lasts the entire day"
```

### When to Use for Fine-Tuning

```
For LLM fine-tuning, data augmentation is LESS common because:
  ✅ The pre-trained model already understands paraphrases
  ✅ LoRA's low parameter count naturally prevents overfitting
  ❌ Augmented text might introduce noise into the instruction format

Better approach: Get more REAL data or increase diversity.
```

---

## Label Smoothing

### The Problem with Hard Labels

```
Hard labels: P(correct class) = 1.0, P(all others) = 0.0

The model tries to output:
  P("BUY") = 1.0000000
  P("SKIP") = 0.0000000
  P("CONSIDER") = 0.0000000

This requires infinitely large logits → encourages overconfident predictions.
Overconfident = poorly calibrated = less robust.
```

### Label Smoothing Solution

```
Label smoothing with ε = 0.1:
  P(correct) = 1 - ε + ε/V = 0.9 + 0.1/256128 ≈ 0.9
  P(each other) = ε/V = 0.1/256128 ≈ 0.0000004

Now the model is penalized for being TOO confident.
It can achieve loss ≈ 0 with P(correct) = 0.9 instead of 1.0.

Benefits:
  ✅ Better calibrated probabilities
  ✅ Smoother training dynamics
  ✅ Slight regularization effect
```

---

## LoRA Dropout: Our Regularization Strategy

### What We Use

```python
# Our regularization strategy in config.py:

# 1. LoRA Dropout (main regularization)
lora_dropout = 0.05

# 2. Weight Decay (gentle)
weight_decay = 0.01

# 3. Early Stopping (implicit via best model selection)
load_best_model_at_end = True

# 4. Low Rank (implicit regularization — fewer params = less overfitting)
lora_r = 16

# 5. Gradient Clipping (prevents explosive gradients, indirectly regularizes)
max_grad_norm = 1.0
```

---

## Regularization Decision Framework

```
Is the model overfitting? (train loss ↓, val loss ↑)
│
├── YES: Apply regularization (in order):
│   1. LoRA dropout:    0.05 → 0.10 → 0.15 → 0.20
│   2. Reduce epochs:   5 → 3 → 2
│   3. More data:       5K → 10K → 20K samples
│   4. Lower rank:      r=32 → r=16 → r=8
│   5. Weight decay:    0.01 → 0.05 → 0.1
│
├── NO (underfitting): Remove regularization:
│   1. LoRA dropout:    0.05 → 0.0
│   2. Higher rank:     r=16 → r=32 → r=64
│   3. More epochs:     3 → 5 → 10
│   4. Higher LR:       2e-4 → 5e-4
│
└── BALANCED: Keep current settings ✅
```
