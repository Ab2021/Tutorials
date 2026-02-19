# 3. Attention Mechanism — Complete Theory

## Table of Contents
- [Why Attention Was Invented](#why-attention-was-invented)
- [Scaled Dot-Product Attention: Full Derivation](#scaled-dot-product-attention-full-derivation)
- [Multi-Head Attention: Why Multiple Heads?](#multi-head-attention-why-multiple-heads)
- [Causal Masking: How Decoder-Only Works](#causal-masking-how-decoder-only-works)
- [Positional Encoding: RoPE Explained](#positional-encoding-rope-explained)
- [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
- [Flash Attention: Memory-Efficient Computation](#flash-attention-memory-efficient-computation)
- [KV Cache: Making Inference Fast](#kv-cache-making-inference-fast)
- [Attention Patterns: What the Model Learns](#attention-patterns-what-the-model-learns)
- [Practical: Building Attention from Scratch](#practical-building-attention-from-scratch)

---

## Why Attention Was Invented

### The Problem with RNNs

```
RNN processes text sequentially:

  "The cat sat on the mat"
   ↓    ↓   ↓   ↓   ↓   ↓
   h₁ → h₂ → h₃ → h₄ → h₅ → h₆

To understand "mat", the model must pass information through
h₁ → h₂ → h₃ → h₄ → h₅ → h₆ (6 sequential steps)

Problems:
  1. Information degrades over long distances (vanishing gradients)
  2. Sequential processing = cannot parallelize = SLOW
  3. "The" at position 0 has trouble influencing position 500+
```

### Attention Solves Both Problems

```
With attention, EVERY token can directly access EVERY other token:

  "The cat sat on the mat"
   ↕    ↕   ↕   ↕   ↕   ↕
   All tokens can attend to all other tokens in ONE step!

  "mat" directly attends to "cat" (distance irrelevant)
  Fully parallelizable (no sequential dependency)
```

---

## Scaled Dot-Product Attention: Full Derivation

### The Core Formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V

Where:
  Q ∈ ℝⁿˣᵈₖ  — Queries  (n tokens, dₖ dimensions per query)
  K ∈ ℝⁿˣᵈₖ  — Keys     (n tokens, dₖ dimensions per key)
  V ∈ ℝⁿˣᵈᵥ  — Values   (n tokens, dᵥ dimensions per value)
  dₖ          — Key dimension (256 for Gemma-2B)
```

### Step-by-Step with Real Numbers

```
Input: 3 tokens with dimension d=4: "cat sat mat"

Token embeddings (after linear projections):
  Q = [[1.0, 0.0, 1.0, 0.0],    ← query for "cat"
       [0.0, 1.0, 0.0, 1.0],    ← query for "sat"
       [1.0, 1.0, 0.0, 0.0]]    ← query for "mat"
       
  K = [[1.0, 0.0, 1.0, 0.0],    ← key for "cat"
       [0.0, 1.0, 0.0, 1.0],    ← key for "sat"
       [0.5, 0.5, 0.5, 0.5]]    ← key for "mat"
       
  V = [[1.0, 0.0, 0.0, 0.0],    ← value for "cat"
       [0.0, 1.0, 0.0, 0.0],    ← value for "sat"
       [0.0, 0.0, 1.0, 0.0]]    ← value for "mat"

STEP 1: Q × Kᵀ (attention scores)
  ┌              ┐     ┌              ┐ᵀ
  │ 1.0 0.0 1.0 0.0 │     │ 1.0 0.0 1.0 0.0 │
  │ 0.0 1.0 0.0 1.0 │  ×  │ 0.0 1.0 0.0 1.0 │
  │ 1.0 1.0 0.0 0.0 │     │ 0.5 0.5 0.5 0.5 │
  └              ┘     └              ┘

  = [[2.0, 0.0, 1.0],     cat→cat=2, cat→sat=0, cat→mat=1
     [0.0, 2.0, 1.0],     sat→cat=0, sat→sat=2, sat→mat=1
     [1.0, 1.0, 1.0]]     mat→cat=1, mat→sat=1, mat→mat=1

STEP 2: Scale by √dₖ = √4 = 2
  = [[1.00, 0.00, 0.50],
     [0.00, 1.00, 0.50],
     [0.50, 0.50, 0.50]]

WHY SCALE?
  Without scaling, for large dₖ (e.g., 256), the dot products become
  very large (on the order of dₖ). This pushes softmax into saturation
  where gradients are near zero. Dividing by √dₖ keeps values moderate.

  Mathematical justification:
    If Q and K entries are drawn from N(0,1),
    then Q·K ~ N(0, dₖ)  (variance = dₖ)
    After dividing by √dₖ: Q·K/√dₖ ~ N(0, 1)  (variance = 1)

STEP 3: Softmax (per row)
  Row 0: softmax([1.00, 0.00, 0.50]) = [0.43, 0.16, 0.26] (≈ normalized)
    e^1.0 = 2.72, e^0.0 = 1.00, e^0.5 = 1.65 → total = 5.37
    = [2.72/5.37, 1.00/5.37, 1.65/5.37] = [0.506, 0.186, 0.307]

  Row 1: softmax([0.00, 1.00, 0.50]) = [0.186, 0.506, 0.307]
  Row 2: softmax([0.50, 0.50, 0.50]) = [0.333, 0.333, 0.333]

  Attention weights:
  A = [[0.506, 0.186, 0.307],   ← "cat" attends mostly to itself
       [0.186, 0.506, 0.307],   ← "sat" attends mostly to itself
       [0.333, 0.333, 0.333]]   ← "mat" attends equally to all

STEP 4: Multiply by V (weighted sum of values)
  Output = A × V
  
  For "cat":  0.506×V(cat) + 0.186×V(sat) + 0.307×V(mat)
            = 0.506×[1,0,0,0] + 0.186×[0,1,0,0] + 0.307×[0,0,1,0]
            = [0.506, 0.186, 0.307, 0.000]
  
  The output for "cat" is now a BLEND of information from all tokens,
  weighted by relevance (attention weights).
```

---

## Multi-Head Attention: Why Multiple Heads?

### The Limitation of Single Head

```
One attention head computes ONE set of Q, K, V.
This means it can only capture ONE TYPE of relationship.

But language has MANY types of relationships:
  - Syntactic: "cat" → "sat" (subject-verb)
  - Semantic: "cat" → "mat" (location)
  - Positional: "cat" → "the" (adjacent)
  - Referential: "it" → "cat" (coreference)
```

### Multi-Head: Parallel Attention

```
Instead of one attention with dₖ=2048:
  Split into 8 heads, each with dₖ=256 (2048/8)

head₁ = Attention(xW₁ᵠ, xW₁ᴷ, xW₁ⱽ)   ← learns syntax
head₂ = Attention(xW₂ᵠ, xW₂ᴷ, xW₂ⱽ)   ← learns semantics
head₃ = Attention(xW₃ᵠ, xW₃ᴷ, xW₃ⱽ)   ← learns position
...
head₈ = Attention(xW₈ᵠ, xW₈ᴷ, xW₈ⱽ)   ← learns other patterns

output = Concat(head₁, ..., head₈) × Wᵒ

Each head: 256 dims → concat: 8 × 256 = 2048 dims → Wᵒ: 2048 dims
```

### Why This Is Better

```
Parameters:
  Single head:  W_Q (2048×2048) + W_K (2048×2048) + W_V (2048×2048)
                = 3 × 2048² = 12.6M params

  8 heads:      8 × [W_Q (2048×256) + W_K (2048×256) + W_V (2048×256)]
                = 8 × 3 × 2048 × 256 = 12.6M params
                + W_O (2048×2048) = 4.2M params

  SAME parameter count, but MORE EXPRESSIVE!
  Each head can learn a different attention pattern.
```

### Gemma-2B Configuration

```
num_attention_heads = 8       (8 query heads)
num_key_value_heads = 1       (1 KV head, shared across all queries)
head_dim = 256                (dimension per head)
hidden_size = 2048            (total dimension)

This is Multi-Query Attention (MQA): 8 Q heads share 1 KV head.
Saves memory during inference (KV cache is 8× smaller).
```

---

## Causal Masking: How Decoder-Only Works

### Why Mask?

```
During text generation, position t can only see positions 0 to t.
It CANNOT see future positions (they haven't been generated yet).

Training must simulate this constraint by MASKING future positions.
```

### The Mask Matrix

```
For sequence length = 5:

            Pos 0  Pos 1  Pos 2  Pos 3  Pos 4
  Pos 0  [  0     -∞      -∞     -∞     -∞  ]   "The"   sees only itself
  Pos 1  [  0      0      -∞     -∞     -∞  ]   "cat"   sees "The", "cat"
  Pos 2  [  0      0       0     -∞     -∞  ]   "sat"   sees "The", "cat", "sat"
  Pos 3  [  0      0       0      0     -∞  ]   "on"    sees first 4 tokens
  Pos 4  [  0      0       0      0      0  ]   "the"   sees all tokens

The -∞ values, after softmax, become 0:
  softmax([1.0, 0.5, -∞, -∞]) = softmax([1.0, 0.5]) = [0.62, 0.38, 0, 0]

So future positions contribute ZERO attention weight.
```

### Implementation

```python
# Create causal mask
seq_len = 512
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
# triu = upper triangular (above diagonal) → -inf
# diagonal and below → 0

# Applied during attention:
scores = Q @ K.T / math.sqrt(d_k)
scores = scores + mask  # Add -inf to future positions
weights = torch.softmax(scores, dim=-1)  # -inf → 0 after softmax
```

---

## Positional Encoding: RoPE Explained

### The Problem

Attention is **permutation-invariant** — it treats "cat sat" the same as "sat cat" because it just computes dot products between all pairs.

```
Q("cat") · K("sat") = same regardless of position order!

But "The cat sat on the mat" ≠ "mat the on sat cat The"
```

### Rotary Position Embeddings (RoPE) — Used by Gemma

RoPE is an elegant way to encode position by ROTATING the query and key vectors:

```
Intuition: Rotate each vector by an angle proportional to its position.
  Position 0: rotate by 0°
  Position 1: rotate by θ × 1
  Position 2: rotate by θ × 2
  ...

When computing Q_m · K_n (attention between positions m and n):
  The dot product naturally depends on (m - n) = RELATIVE distance!
  
  Q(position=5) · K(position=3) depends on the distance 5-3=2
  Q(position=10) · K(position=8) also depends on distance 10-8=2
  
  Same relative distance → same attention pattern!
```

### The Math

```
For a pair of dimensions (2i, 2i+1) at position m:

  q'₂ᵢ(m) = q₂ᵢ × cos(mθᵢ) - q₂ᵢ₊₁ × sin(mθᵢ)
  q'₂ᵢ₊₁(m) = q₂ᵢ × sin(mθᵢ) + q₂ᵢ₊₁ × cos(mθᵢ)

Where θᵢ = 10000^(-2i/d)

This is a ROTATION of the (2i, 2i+1) plane by angle mθᵢ.
The rotation angle scales with position m.

For low-frequency dimensions (large θ): captures LONG-range position
For high-frequency dimensions (small θ): captures SHORT-range position
```

### Why RoPE Is Better Than Absolute Position Embeddings

```
Absolute PE (GPT-2):
  embed("cat", position=5) = word_embed("cat") + pos_embed(5)
  ❌ Max position = training max (e.g., 512)
  ❌ Cannot generalize to longer sequences

RoPE (Gemma):
  embed("cat", position=5) = rotate(word_embed("cat"), angle=5θ)
  ✅ Theoretically works for ANY position
  ✅ Encodes RELATIVE distance (more useful than absolute)
  ✅ Better generalization to longer sequences than seen in training
```

---

## Grouped-Query Attention (GQA)

### The KV Cache Memory Problem

```
During inference, we cache K and V for all previous tokens
to avoid recomputing them:

Standard Multi-Head Attention (MHA):
  KV cache per layer = 2 × num_heads × seq_len × head_dim × bytes
  For Gemma with 8 heads: 2 × 8 × seq_len × 256 × 2 bytes
  For 2048 token sequence: 2 × 8 × 2048 × 256 × 2 = 16.8 MB per layer
  18 layers: 302 MB just for KV cache!
```

### GQA Reduces KV Cache

```
Standard MHA:     8 Q heads, 8 K heads, 8 V heads (1:1 matching)
Grouped-Query:    8 Q heads, 2 K heads, 2 V heads (4:1 grouping)
Multi-Query (MQA): 8 Q heads, 1 K head, 1 V head   (8:1 grouping)

Gemma-2B uses MQA (1 KV head):
  KV cache: 2 × 1 × 2048 × 256 × 2 = 2.1 MB per layer
  18 layers: 37.7 MB — 8× smaller than standard MHA!
```

### How It Works

```
Standard MHA:
  Q₁ attends using K₁, V₁
  Q₂ attends using K₂, V₂
  ...
  Q₈ attends using K₈, V₈

MQA (Gemma-2B):
  Q₁ attends using K₁, V₁  ┐
  Q₂ attends using K₁, V₁  │  Same K,V shared
  Q₃ attends using K₁, V₁  │  across ALL queries
  ...                        │
  Q₈ attends using K₁, V₁  ┘

Each query head still has its own learned W_Q,
but they all share the same W_K and W_V.

Quality impact: < 1% degradation in most benchmarks.
Speed impact: 8× faster inference (much smaller KV cache).
```

---

## Flash Attention: Memory-Efficient Computation

### The Standard Attention Memory Problem

```
Standard attention computation:
  S = Q × Kᵀ           ← Shape: (n, n) for n tokens
  S = S / √dₖ
  P = softmax(S)        ← Shape: (n, n)
  O = P × V             ← Shape: (n, dᵥ)

For n=2048 sequence, the attention matrix S is:
  2048 × 2048 = 4.2 million elements × 2 bytes (fp16) = 8.4 MB
  
  Per head, per layer. With 8 heads × 18 layers:
  8.4 MB × 8 × 18 = 1.2 GB just for attention matrices!

For n=8192 (Gemma's max):
  8192 × 8192 × 2 = 134 MB per head = 19.3 GB total!
  DOES NOT FIT IN MEMORY!
```

### Flash Attention: Tiled Computation

```
Instead of materializing the full n×n attention matrix,
FlashAttention computes attention in TILES:

Standard:
  ┌─────────────────┐
  │                 │  Full n×n matrix in GPU HBM (slow memory)
  │   4.2 million   │
  │   elements      │
  │                 │
  └─────────────────┘

FlashAttention:
  ┌───┐ ┌───┐ ┌───┐
  │   │ │   │ │   │  Small tiles in GPU SRAM (fast memory)
  └───┘ └───┘ └───┘
  Process one tile at a time
  Never materialize full matrix

Algorithm:
  1. Split Q into blocks of size B_Q
  2. Split K, V into blocks of size B_KV
  3. For each Q block:
     For each K,V block:
       Compute partial attention scores
       Update running softmax (online softmax trick)
       Accumulate partial output
  4. Reconstruct exact attention output

Results:
  ✅ Same exact output as standard attention (not approximate!)
  ✅ O(n) memory instead of O(n²)
  ✅ 2-4× faster (better memory access patterns)
```

### Online Softmax: The Key Trick

```
Standard softmax: Need ALL scores to compute the denominator:
  softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
  
  Can't compute until we have ALL xⱼ values!

Online softmax: Process blocks, update running statistics:
  Block 1: m₁ = max(x₁,...,x_B), d₁ = Σᵢ e^(xᵢ - m₁)
  Block 2: m₂ = max(m₁, max(x_{B+1},...,x_{2B}))
            d₂ = d₁ × e^(m₁-m₂) + Σᵢ e^(xᵢ - m₂)
  ...
  
  At the end: m_total and d_total give exact softmax!
  
  Never need all scores in memory simultaneously.
```

---

## KV Cache: Making Inference Fast

### The Problem

```
Autoregressive generation at step t:
  Input: "The cat sat on the"
  Model processes ALL 5 tokens through ALL layers
  Output: next token = "mat"

At step t+1:
  Input: "The cat sat on the mat"
  Model processes ALL 6 tokens through ALL layers
  Output: next token = "."

Problem: We're re-processing "The", "cat", "sat", "on", "the"
EVERY step. These computations give the same results!
```

### KV Caching Solution

```
Step 1: "The" → compute and CACHE K₁, V₁ for all layers
Step 2: "cat" → compute K₂, V₂, attend to K₁₂, V₁₂ (cached + new)
Step 3: "sat" → compute K₃, V₃, attend to K₁₂₃, V₁₂₃
...

At each step:
  Only compute Q, K, V for the NEW token (not all previous tokens)
  Use cached K, V for all previous tokens
  Append new K, V to cache

Speedup: O(n × d²) per step → O(d²) per step
  For a 2048-token sequence: ~2048× faster inference!
```

### Cache Size Calculation

```
Per layer, per token:
  K cache: 1 KV head × head_dim × dtype = 1 × 256 × 2 bytes = 512 bytes
  V cache: 1 KV head × head_dim × dtype = 1 × 256 × 2 bytes = 512 bytes
  Total per layer: 1024 bytes (1 KB)

For 2048 tokens × 18 layers:
  2048 × 18 × 1024 = 37.7 MB

For 8192 tokens × 18 layers:
  8192 × 18 × 1024 = 150.9 MB

Gemma-2B with MQA: very manageable KV cache!
(A model with standard MHA would need 8× more.)
```

---

## Attention Patterns: What the Model Learns

### Common Patterns Observed in Trained Models

```
Pattern 1: DIAGONAL (Local attention)
  Each token primarily attends to itself and immediate neighbors.
  ┌─────────────┐
  │■ ░           │
  │░ ■ ░         │
  │  ░ ■ ░       │
  │    ░ ■ ░     │
  │      ░ ■ ░   │
  │        ░ ■   │
  └─────────────┘
  Purpose: Processing local syntax and grammar.

Pattern 2: VERTICAL STRIPE (attend to special token)
  All tokens attend to one specific position (e.g., <bos>).
  ┌─────────────┐
  │■ ░ ░ ░ ░ ░ │
  │■ ░ ░ ░ ░ ░ │
  │■ ░ ░ ░ ░ ░ │
  │■ ░ ░ ░ ░ ░ │
  │■ ░ ░ ░ ░ ░ │
  │■ ░ ░ ░ ░ ░ │
  └─────────────┘
  Purpose: <bos> acts as a "no-op" attention sink.

Pattern 3: BLOCK (segment attention)
  Tokens within a clause attend to each other.
  ┌─────────────┐
  │■ ■ ■ ░ ░ ░ │
  │■ ■ ■ ░ ░ ░ │
  │■ ■ ■ ░ ░ ░ │
  │░ ░ ░ ■ ■ ■ │
  │░ ░ ░ ■ ■ ■ │
  │░ ░ ░ ■ ■ ■ │
  └─────────────┘
  Purpose: Clause-level processing.

Pattern 4: SPARSE (specific word relationships)
  Specific tokens attend to specific related tokens.
  Used for: pronoun resolution, semantic relationships.
```

---

## Practical: Building Attention from Scratch

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute attention from scratch.
    
    Args:
        Q: (batch, heads, seq_len, head_dim)
        K: (batch, heads, seq_len, head_dim)
        V: (batch, heads, seq_len, head_dim)
        mask: (seq_len, seq_len) causal mask
    
    Returns:
        output: (batch, heads, seq_len, head_dim)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, n, n)
    
    # Step 2: Scale
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply causal mask (optional)
    if mask is not None:
        scores = scores + mask  # -inf for future positions
    
    # Step 4: Softmax
    weights = F.softmax(scores, dim=-1)  # (batch, heads, n, n)
    
    # Step 5: Weighted sum of values
    output = torch.matmul(weights, V)  # (batch, heads, n, head_dim)
    
    return output, weights


class MultiHeadAttention(torch.nn.Module):
    """Complete multi-head attention implementation."""
    
    def __init__(self, d_model=2048, num_heads=8, num_kv_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads  # For GQA
        
        # Projection matrices (these are what LoRA attaches to!)
        self.q_proj = torch.nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * self.head_dim, d_model, bias=False)
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Repeat KV heads for GQA/MQA
        if self.num_groups > 1:
            K = K.repeat(1, self.num_groups, 1, 1)  # (batch, 8, seq, dim)
            V = V.repeat(1, self.num_groups, 1, 1)
        
        # Causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )
        
        # Apply RoPE here (omitted for clarity)
        
        # Compute attention
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()  # (batch, seq, heads, dim)
        output = output.view(batch, seq_len, -1)        # (batch, seq, d_model)
        output = self.o_proj(output)
        
        return output


# === DEMO ===
if __name__ == "__main__":
    batch_size = 1
    seq_len = 10
    d_model = 2048
    
    # Random input (simulating token embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=8, num_kv_heads=1)
    
    # Forward pass
    output = mha(x)
    print(f"Input shape:  {x.shape}")      # [1, 10, 2048]
    print(f"Output shape: {output.shape}")  # [1, 10, 2048]
    
    # Count parameters
    total = sum(p.numel() for p in mha.parameters())
    print(f"Total attention parameters: {total:,}")
    # q_proj: 2048×2048 = 4.2M
    # k_proj: 2048×256 = 524K (MQA)
    # v_proj: 2048×256 = 524K (MQA)
    # o_proj: 2048×2048 = 4.2M
    # Total: ~9.4M per layer
```
