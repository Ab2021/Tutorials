# 1. What Is Fine-Tuning? — A Complete Guide

## Table of Contents
- [The Big Picture](#the-big-picture)
- [Pre-Training vs Fine-Tuning](#pre-training-vs-fine-tuning)
- [Why Not Just Use Prompting?](#why-not-just-use-prompting)
- [Types of Fine-Tuning](#types-of-fine-tuning)
- [When to Use Which Method](#when-to-use-which-method)
- [The Fine-Tuning Lifecycle](#the-fine-tuning-lifecycle)
- [Risks and Pitfalls](#risks-and-pitfalls)

---

## The Big Picture

Imagine you hire a brilliant college graduate. They know a lot about the world, can write well, and understand language. But they don't know anything about *your specific business* — they don't know your products, your customers, or your internal terminology.

**Fine-tuning** is like giving that graduate specialized on-the-job training. You're not teaching them language from scratch (that was pre-training). You're teaching them *your specific task*.

```
┌─────────────────────────────────────────────────────────────┐
│                    PRE-TRAINING (by Google)                  │
│                                                             │
│  Gemma reads trillions of words from books, websites,       │
│  code, and conversations. Learns language, facts, reasoning │
│                                                             │
│  Cost: ~$10 million                                         │
│  Time: Weeks on thousands of GPUs                           │
│  Data: Trillions of tokens                                  │
│  Result: A model that "understands" language                │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINE-TUNING (by you)                      │
│                                                             │
│  You take Google's pre-trained Gemma and train it further   │
│  on your specific dataset (product reviews → recommendations│
│                                                             │
│  Cost: ~$5-50                                               │
│  Time: Hours on a single GPU                                │
│  Data: Thousands of examples                                │
│  Result: A model that does YOUR specific task well          │
└─────────────────────────────────────────────────────────────┘
```

---

## Pre-Training vs Fine-Tuning

### Pre-Training
- **Who does it:** Google, Meta, OpenAI — large companies with massive compute
- **Data:** Trillions of tokens from the entire internet
- **Goal:** Learn language, reasoning, world knowledge
- **Cost:** Millions of dollars
- **Result:** A "foundation model" that knows everything but isn't specialized

### Fine-Tuning
- **Who does it:** You! Anyone with a GPU
- **Data:** Your specific task data (thousands to millions of examples)
- **Goal:** Specialize the model for a particular task
- **Cost:** A few dollars in compute
- **Result:** A model that excels at your exact use case

### Analogy

| Stage | Education Analogy |
|-------|-------------------|
| Pre-training | K-12 + College — broad education |
| Fine-tuning | Medical school — specialized training |
| Prompting | Giving instructions to a generalist |
| RAG | Giving the generalist a reference book |

---

## Why Not Just Use Prompting?

You might wonder: "Can't I just write a good prompt and use Gemma as-is?"

### Prompting (Zero-Shot / Few-Shot)

```
Prompt: "You are a product recommendation expert. Given this review,
provide a recommendation: 'Great phone, love the camera.' Rating: 5"
```

**Pros:**
- No training needed
- Works immediately
- Can change instructions on the fly

**Cons:**
- Output format is inconsistent
- Quality is limited by the model's pre-training
- Long prompts waste tokens ($$$ in production)
- Can't learn domain-specific patterns
- Each request needs the full instruction

### Fine-Tuning

**Pros:**
- Consistent output format (the model learned it)
- Higher quality for your specific task
- Shorter prompts at inference time (cheaper)
- Can learn domain-specific terminology
- Handles nuanced patterns the base model can't

**Cons:**
- Requires training data
- Takes time to train
- Need to retrain when requirements change
- Risk of overfitting or catastrophic forgetting

### Decision Framework

```
                    Do you need high-quality,
                    consistent output?
                         │
                    ┌────┴────┐
                    │ YES     │ NO
                    ▼         ▼
              Do you have    Use prompting
              training data? (zero/few-shot)
                    │
               ┌────┴────┐
               │ YES     │ NO
               ▼         ▼
          Fine-tune!   Use RAG or
                       collect data first
```

### When to Use Each Method

| Method | Best For | Example |
|--------|----------|---------|
| **Zero-shot prompting** | Simple, one-off tasks | "Summarize this paragraph" |
| **Few-shot prompting** | Tasks with clear patterns | "Classify sentiment: positive/negative" |
| **RAG** (Retrieval-Augmented Gen) | Factual Q&A over documents | "What's our refund policy?" |
| **Fine-tuning** | Complex, domain-specific tasks | Product recommendation from reviews |
| **Full pre-training** | Brand new language/domain | Medical language model from scratch |

---

## Types of Fine-Tuning

### 1. Full Fine-Tuning
**Update ALL model parameters.**

```
Model: 2 billion parameters
Updated: 2 billion parameters (100%)
Memory: ~50 GB VRAM
```

**When to use:**
- You have massive amounts of data (millions of examples)
- You need maximum quality and have the hardware
- You're fundamentally changing what the model does

**When NOT to use:**
- You have limited GPU memory (< 40 GB)
- Your dataset is small (< 50K examples)
- This is your first fine-tuning project

### 2. LoRA (Low-Rank Adaptation)
**Freeze original weights, add small trainable adapters.**

```
Model: 2 billion parameters
Frozen: 1.997 billion parameters (99.87%)
Updated: 2.6 million parameters (0.13%)
Memory: ~8 GB VRAM
```

**When to use:**
- Most fine-tuning scenarios (this is the default choice)
- Consumer GPU (8-24 GB VRAM)
- Want to maintain base model knowledge
- Need to compare multiple fine-tuned versions

**When NOT to use:**
- The base model fundamentally doesn't understand your domain

### 3. QLoRA (Quantized LoRA)
**Same as LoRA, but the base model is compressed to 4-bit.**

```
Model: 2 billion parameters (stored in 4-bit)
Updated: 2.6 million LoRA parameters (in 16-bit)
Memory: ~6 GB VRAM
```

**When to use:**
- Limited VRAM (8-16 GB) — THIS IS WHAT WE USE
- First-time fine-tuning projects
- Rapid prototyping

**When NOT to use:**
- You need absolute maximum quality
- You have abundant GPU memory

### 4. Prefix Tuning
**Add trainable "prefix" tokens to the input.**

Instead of modifying model weights, you prepend learned embeddings to the input.

**When to use:**
- Very parameter-efficient (even fewer params than LoRA)
- Multiple tasks sharing one base model

**When NOT to use:**
- Complex tasks requiring deep model adaptation

### 5. Adapter Tuning
**Insert small bottleneck layers between existing layers.**

Similar idea to LoRA but structured differently.

**When to use:**
- Multi-task learning (different adapter per task)
- Older approach, largely replaced by LoRA

### 6. RLHF (Reinforcement Learning from Human Feedback)
**Fine-tune using human preference rankings.**

```
Step 1: Fine-tune model (SFT — what we do)
Step 2: Train a reward model on human preferences
Step 3: Use PPO/DPO to optimize the model against the reward model
```

**When to use:**
- Making the model more helpful, harmless, and honest
- Aligning output with human preferences
- Production chatbots

**When NOT to use:**
- You just need task-specific output format
- You don't have human preference data

### Comparison Table

| Method | Params Updated | VRAM | Quality | Complexity | Our Choice? |
|--------|---------------|------|---------|------------|-------------|
| Full FT | 100% | 50+ GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ Too expensive |
| LoRA | 0.13% | 8+ GB | ⭐⭐⭐⭐ | ⭐⭐ | ✅ Could use |
| **QLoRA** | **0.13%** | **6+ GB** | **⭐⭐⭐⭐** | **⭐⭐** | **✅ Our choice!** |
| Prefix Tuning | <0.1% | 6+ GB | ⭐⭐⭐ | ⭐⭐⭐ | ❌ Less proven |
| RLHF | 100% or LoRA | 24+ GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ Overkill |

---

## The Fine-Tuning Lifecycle

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. DEFINE    │ →  │ 2. COLLECT   │ →  │ 3. PREPARE   │
│    TASK      │    │    DATA      │    │    DATA      │
│              │    │              │    │              │
│ What should  │    │ Find/create  │    │ Clean,       │
│ the model    │    │ examples of  │    │ format into  │
│ output?      │    │ input→output │    │ prompts      │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
        ┌──────────────────────────────────────┘
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 4. TRAIN     │ →  │ 5. EVALUATE  │ →  │ 6. DEPLOY    │
│              │    │              │    │              │
│ Fine-tune    │    │ Test on held │    │ Merge LoRA,  │
│ with QLoRA   │    │ out data,    │    │ serve model  │
│              │    │ compute      │    │              │
│              │    │ metrics      │    │              │
└──────────────┘    └──────┬───────┘    └──────────────┘
                           │
                    Not good enough?
                           │
                    ┌──────┴───────┐
                    │ ITERATE      │
                    │              │
                    │ Adjust hyper │
                    │ params, add  │
                    │ more data    │
                    └──────────────┘
```

---

## Risks and Pitfalls

### 1. Catastrophic Forgetting
**What:** The model forgets its pre-trained knowledge while learning the new task.
**Why:** When you update too many weights too aggressively, you overwrite the original knowledge.
**Fix:** Use LoRA (only updates 0.13% of weights), use low learning rate, train for fewer epochs.

### 2. Overfitting
**What:** Model memorizes training data instead of learning general patterns.
**Signs:** Training loss decreases, but validation loss increases.
**Fix:** More data, more dropout, fewer epochs, lower LoRA rank.

### 3. Underfitting
**What:** Model doesn't learn the task well enough.
**Signs:** Both training and validation loss remain high.
**Fix:** Higher LoRA rank, more epochs, higher learning rate, more data.

### 4. Distribution Shift
**What:** Training data doesn't represent real-world inputs.
**Example:** Training on electronics reviews, testing on food reviews.
**Fix:** Ensure training data covers the full range of expected inputs.

### 5. Data Quality Issues
**What:** Garbage in = garbage out.
**Common problems:** Duplicates, mislabeled examples, too-short examples.
**Fix:** Clean data thoroughly, inspect samples manually.

---

## Our Choice: Why QLoRA for This Project

We chose **QLoRA** because:

1. **Accessibility** — Runs on consumer GPUs (8 GB VRAM)
2. **Quality** — 4-bit quantization with NF4 loses minimal quality
3. **Speed** — Trains in under an hour on 5K examples
4. **Simplicity** — Well-supported by HuggingFace PEFT and TRL
5. **Proven** — Widely used in research and production

The quality gap between QLoRA and full fine-tuning is typically < 1% on benchmarks, while using 8x less memory. For product recommendation, this trade-off is excellent.
