# 2. Understanding Gemma — Architecture Deep Dive

## Table of Contents
- [What is Gemma?](#what-is-gemma)
- [The Transformer Architecture](#the-transformer-architecture)
- [How Self-Attention Works](#how-self-attention-works)
- [Gemma's Specific Architecture](#gemmas-specific-architecture)
- [How LLMs Generate Text](#how-llms-generate-text)
- [Gemma vs Other Models](#gemma-vs-other-models)
- [Which Gemma Model to Choose](#which-gemma-model-to-choose)

---

## What is Gemma?

**Gemma** is Google's family of open-source large language models (LLMs). The name comes from the Latin word *gemma*, meaning "precious stone."

Key facts:
- **Created by:** Google DeepMind
- **Released:** February 2024
- **Sizes:** 2B and 7B parameters
- **License:** Open weights with permissive terms
- **Based on:** Same research as Gemini (Google's flagship AI)
- **Architecture:** Decoder-only Transformer

---

## The Transformer Architecture

All modern LLMs (GPT, Llama, Gemma, Mistral) are built on the **Transformer** architecture, invented in the 2017 paper "Attention Is All You Need."

### The Building Blocks

```
┌─────────────────────────────────────────────────────┐
│                    GEMMA MODEL                       │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │            OUTPUT HEAD (lm_head)              │    │
│  │   Converts hidden states → token probs       │    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                           │
│  ┌──────────────────────┴──────────────────────┐    │
│  │         TRANSFORMER BLOCK × 18               │    │
│  │  ┌────────────────────────────────────────┐  │    │
│  │  │  Layer Norm                             │  │    │
│  │  │  ↓                                      │  │    │
│  │  │  Multi-Head Self-Attention              │  │    │
│  │  │    (q_proj, k_proj, v_proj, o_proj)     │  │    │
│  │  │  ↓                                      │  │    │
│  │  │  Residual Connection + Layer Norm       │  │    │
│  │  │  ↓                                      │  │    │
│  │  │  Feed-Forward Network (MLP)             │  │    │
│  │  │    (gate_proj, up_proj, down_proj)      │  │    │
│  │  │  ↓                                      │  │    │
│  │  │  Residual Connection                    │  │    │
│  │  └────────────────────────────────────────┘  │    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                           │
│  ┌──────────────────────┴──────────────────────┐    │
│  │         TOKEN EMBEDDING LAYER                │    │
│  │   Converts token IDs → dense vectors         │    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                           │
│                    [Input Tokens]                    │
└─────────────────────────────────────────────────────┘
```

### Step-by-Step: How a Forward Pass Works

**Input:** "The cat sat on the" → predict next token

```
Step 1: TOKENIZATION
   "The cat sat on the" → [651, 2857, 3856, 356, 651]
   Each word becomes a number (token ID) from the vocabulary.

Step 2: EMBEDDING
   [651, 2857, 3856, 356, 651] → [[0.12, -0.34, ...], [0.56, 0.78, ...], ...]
   Each token ID becomes a dense vector of size 2048 (Gemma-2B).
   This vector represents the token's meaning in a high-dimensional space.

Step 3: TRANSFORMER BLOCKS (×18)
   Each block refines the token representations by:
   a) Looking at ALL other tokens (self-attention)
   b) Processing each token independently (feed-forward)
   
   After 18 blocks, each token's vector contains rich contextual info
   about its meaning IN CONTEXT of all surrounding tokens.

Step 4: OUTPUT HEAD
   The final hidden state of the LAST token → probability distribution
   over ALL vocabulary tokens.
   
   Output: {"mat": 0.35, "floor": 0.20, "fence": 0.05, ...}
   
   The highest probability token becomes the prediction: "mat"
```

---

## How Self-Attention Works

Self-attention is the key innovation that makes Transformers powerful. It allows every token to "look at" every other token and gather relevant information.

### The Process

```
For each token, we compute three things:

Query (Q): "What am I looking for?"
   ┌─────────────────┐
   │ q_proj (linear)  │  hidden_state × W_q → query vector
   └─────────────────┘

Key (K): "What do I contain?"
   ┌─────────────────┐
   │ k_proj (linear)  │  hidden_state × W_k → key vector
   └─────────────────┘

Value (V): "What information do I provide?"
   ┌─────────────────┐
   │ v_proj (linear)  │  hidden_state × W_v → value vector
   └─────────────────┘
```

### Example: "The cat sat on the mat"

When processing the word "sat":

```
Q("sat") asks: "Who is sitting? Where?"

It computes attention scores with every key:
  Q("sat") · K("The")  = 0.1  (low — "The" isn't very relevant)
  Q("sat") · K("cat")  = 0.8  (HIGH — "cat" is the one sitting!)
  Q("sat") · K("sat")  = 0.3  (medium — self-reference)
  Q("sat") · K("on")   = 0.5  (medium — preposition context)
  Q("sat") · K("the")  = 0.1  (low)
  Q("sat") · K("mat")  = 0.6  (medium-high — location)

These scores become attention weights (via softmax):
  [0.03, 0.42, 0.08, 0.17, 0.03, 0.27]

The output for "sat" = weighted sum of values:
  0.03 × V("The") + 0.42 × V("cat") + 0.08 × V("sat") + ...

Result: The representation of "sat" now contains information
about WHO sat (the cat) and WHERE (on the mat).
```

### Multi-Head Attention

Instead of one set of Q/K/V, Gemma-2B uses **8 attention heads** in parallel. Each head learns to attend to different types of relationships:

```
Head 1: Syntax (subject-verb agreement)
Head 2: Semantics (word meanings)
Head 3: Position (nearby words)
Head 4: Reference (pronouns → nouns)
...
Head 8: Domain-specific patterns
```

The outputs from all heads are concatenated and passed through `o_proj` (output projection).

### Causal Masking (Decoder-Only)

Gemma is a **decoder-only** model, meaning each token can ONLY attend to tokens that came BEFORE it (and itself). This is enforced by **causal masking**:

```
         The  cat  sat   on  the  mat
The       ✅   ❌   ❌   ❌   ❌   ❌
cat       ✅   ✅   ❌   ❌   ❌   ❌
sat       ✅   ✅   ✅   ❌   ❌   ❌
on        ✅   ✅   ✅   ✅   ❌   ❌
the       ✅   ✅   ✅   ✅   ✅   ❌
mat       ✅   ✅   ✅   ✅   ✅   ✅

✅ = can attend    ❌ = cannot attend (masked)
```

**Why causal masking?** Because during generation, the model doesn't have access to future tokens — they haven't been generated yet! Training with causal masking simulates this condition.

---

## Gemma's Specific Architecture

### Gemma-2B Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `hidden_size` | 2048 | Dimension of token embeddings |
| `num_hidden_layers` | 18 | Number of transformer blocks |
| `num_attention_heads` | 8 | Number of parallel attention heads |
| `num_key_value_heads` | 1 | GQA uses 1 KV head shared across all Q heads |
| `intermediate_size` | 16384 | Size of feedforward hidden layer (8× hidden) |
| `vocab_size` | 256128 | Number of unique tokens |
| `max_position_embeddings` | 8192 | Maximum sequence length |
| `head_dim` | 256 | Dimension per attention head |

### Gemma-Specific Features

**1. Rotary Position Embeddings (RoPE)**
Instead of absolute position embeddings ("I am token #5"), RoPE encodes relative distances between tokens using rotation matrices. This allows the model to generalize to longer sequences than it was trained on.

**2. Grouped-Query Attention (GQA)**
Standard attention: Each query head has its own key-value pair.
GQA: Multiple query heads SHARE key-value pairs.
Gemma-2B uses 8 query heads and 1 key-value head (MQA — Multi-Query Attention), saving memory during inference.

**3. GeGLU Activation**
The feed-forward network uses GeGLU (Gated Linear Unit with GELU activation) instead of standard ReLU:
```
FFN(x) = GELU(x × W_gate) ⊗ (x × W_up)
```
This gives better gradient flow and training stability.

**4. RMSNorm (instead of LayerNorm)**
Simpler normalization that only normalizes by the root mean square, without re-centering. Faster and works just as well.

---

## How LLMs Generate Text

### Autoregressive Generation

LLMs generate text **one token at a time**, using their own previous output as input:

```
Step 1: Input: "The product is"
        Model predicts: "great" (probability: 0.35)

Step 2: Input: "The product is great"
        Model predicts: "for" (probability: 0.28)

Step 3: Input: "The product is great for"
        Model predicts: "everyday" (probability: 0.22)

... continues until max_tokens or <eos> token
```

### Decoding Strategies

At each step, the model outputs a probability distribution over ALL vocab tokens. How we choose the next token matters:

**1. Greedy Decoding** (`temperature=0, top_k=1`)
Always pick the highest probability token.
```
Probabilities: {"great": 0.35, "good": 0.30, "perfect": 0.20, ...}
Selected: "great" (always)
```
- ✅ Deterministic, consistent
- ❌ Repetitive, boring

**2. Temperature Sampling** (`temperature=0.7`)
Divide logits by temperature before softmax:
```
Before: {"great": 0.35, "good": 0.30, "perfect": 0.20}
After (T=0.7): {"great": 0.45, "good": 0.32, "perfect": 0.15}  ← sharper
After (T=1.5): {"great": 0.30, "good": 0.28, "perfect": 0.22}  ← flatter
```
Lower T → more focused. Higher T → more random.

**3. Top-K Sampling** (`top_k=50`)
Only consider the K most likely tokens:
```
All tokens: {"great": 0.35, "good": 0.30, ..., "banana": 0.0001}
top_k=50: Keep only top 50, discard rest, renormalize
```
Prevents selecting very unlikely (nonsensical) tokens.

**4. Nucleus Sampling (Top-P)** (`top_p=0.9`)
Keep tokens until cumulative probability exceeds P:
```
"great":   0.35 (cumsum: 0.35)  ← included
"good":    0.30 (cumsum: 0.65)  ← included
"perfect": 0.20 (cumsum: 0.85)  ← included
"nice":    0.08 (cumsum: 0.93)  ← included (exceeds 0.9)
"fine":    0.04 ← EXCLUDED
```
Adapts dynamically: with a confident prediction, considers fewer tokens.

---

## Gemma vs Other Models

| Model | Params | Creator | License | Best For |
|-------|--------|---------|---------|----------|
| **Gemma-2B** | 2B | Google | Open | Small/edge, our choice |
| **Gemma-7B** | 7B | Google | Open | Better quality, more VRAM |
| **Llama 3** | 8B/70B | Meta | Open | General-purpose, very popular |
| **Mistral 7B** | 7B | Mistral AI | Open | Efficient, strong performance |
| **Phi-2** | 2.7B | Microsoft | Open | Small model, surprisingly good |
| **GPT-4** | ~1.7T? | OpenAI | Closed | Best quality, API only |

### Why Gemma for This Project?

1. **Size:** 2B params fit on consumer GPUs with QLoRA
2. **Quality:** Punches above its weight (outperforms many 7B models)
3. **Open:** Can download, modify, and use commercially
4. **Google's backing:** Well-tested, good documentation
5. **HuggingFace integration:** Works out-of-the-box with Transformers

---

## Which Gemma Model to Choose

| Model | Use Case | VRAM (QLoRA) | Quality |
|-------|----------|-------------|---------|
| `google/gemma-2b` | **Development & learning** ← Our choice | ~6 GB | Good |
| `google/gemma-2b-it` | If you want a pre-instruction-tuned base | ~6 GB | Good for chat |
| `google/gemma-7b` | Production fine-tuning | ~16 GB | Better |
| `google/gemma-7b-it` | Production chat fine-tuning | ~16 GB | Best for chat |

### -it vs base models

- **Base models** (`gemma-2b`): Raw model that does next-token prediction. Best for fine-tuning from scratch on your task.
- **Instruction-tuned** (`gemma-2b-it`): Already fine-tuned to follow instructions. Better if you want to build ON TOP of existing chat abilities.

**Our recommendation:** Use the base model (`gemma-2b`) for fine-tuning. It gives you more control over what the model learns, without interference from Google's instruction tuning.
