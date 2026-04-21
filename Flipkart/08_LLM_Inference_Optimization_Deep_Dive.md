# ⚡ LLM Inference Optimization — Deep Dive
### KV Cache · Flash Attention · Paged Attention (vLLM) · Quantization · Speculative Decoding · Batching

> **Why this matters for Flipkart:**
> SLAP (shopping agent) + Seller AI + Fraud Investigator — all LLM-powered, all serving 350M+ users.
> Inference efficiency = latency SLAs met + cost controlled + throughput scaled.
> Alex (PhD, ML systems) WILL ask about inference if you claim production LLM experience.

---

## 🗺️ THE INFERENCE OPTIMIZATION LANDSCAPE

```
THE CORE PROBLEM:
A 70B-parameter LLM generating 1 token:
  - Must load 70B × 2 bytes (FP16) = 140GB of weights
  - From GPU HBM bandwidth of ~2TB/s → 70ms just for memory I/O per token
  - At 100 tokens/s target → 10ms budget per token → 7x too slow on naive implementation

USER WANTS: Fast responses (< 2s first token), smooth streaming (30+ tok/s)
ENGINEER MUST: Maximize throughput, minimize latency, minimize cost

FOUR AXES OF OPTIMIZATION:
┌─────────────────────────────────────────────────────────────────┐
│ 1. MEMORY OPTIMIZATION     → KV Cache, Paged Attention          │
│    (reduce memory traffic and footprint)                        │
│                                                                 │
│ 2. COMPUTE OPTIMIZATION    → Flash Attention, Fused Ops         │
│    (reduce FLOPs and I/O, maximize hardware utilization)        │
│                                                                 │
│ 3. MODEL COMPRESSION       → Quantization (INT8, INT4, GPTQ)    │
│    (smaller weights → faster load → less memory)               │
│                                                                 │
│ 4. DECODING OPTIMIZATION   → Speculative Decoding, Batching     │
│    (better use of parallel hardware per decoding step)          │
└─────────────────────────────────────────────────────────────────┘
```

---

# PART 1: MEMORY OPTIMIZATION

---

## 📦 TOPIC 1: KV CACHE — THE FOUNDATION OF FAST INFERENCE

### 1.1 Why KV Cache Exists

Autoregressive generation processes one token at a time:

```
GENERATING "The cat sat on the mat":

Step 1: Input="The"        → generate "cat"
Step 2: Input="The cat"    → generate "sat"
Step 3: Input="The cat sat"→ generate "on"
...

For each step t, attention computes:
  Attention(Q_t, K_{1..t}, V_{1..t}) = softmax(Q_t K_{1..t}^T / √d_k) V_{1..t}

WITHOUT CACHING:
  At step t, we RECOMPUTE K and V for ALL previous tokens 1..t-1
  But K_1, K_2, ..., K_{t-1} haven't changed since last step → WASTED COMPUTATION

WITH KV CACHE:
  Store K_1, K_2, ..., K_{t-1} in GPU memory from previous steps
  At step t:
  - Only compute K_t, V_t for the NEW token
  - Concatenate: K_{cached} + K_t, V_{cached} + V_t
  - Run attention using full K, V
```

### 1.2 KV Cache Memory Math

```
KV Cache size = 2 (K and V) 
              × num_layers 
              × num_heads 
              × seq_len 
              × head_dim 
              × batch_size 
              × bytes_per_element

Example: LLaMA-2-70B at seq_len=4096, batch_size=1, FP16:
  = 2 × 80 layers × 64 heads × 4096 seq × 128 head_dim × 1 × 2 bytes
  = 2 × 80 × 64 × 4096 × 128 × 2
  = ~10.7 GB just for KV cache!

For batch_size=10 concurrent users: ~107 GB → exceeds single A100 (80GB)!

LLaMA-2-70B FULL MODEL WEIGHTS: ~140GB (FP16) → need at least 2× A100
KV cache adds: ~10GB per user per 4K context → quickly dominates
```

### 1.3 KV Cache Reduction Techniques

#### Multi-Query Attention (MQA) — Shazeer, 2019

```
IDEA: Instead of H separate K,V heads, use ONE shared K,V for all H query heads

STANDARD MHA:
  Q_1, K_1, V_1  → head_1 output
  Q_2, K_2, V_2  → head_2 output
  ...
  Q_H, K_H, V_H  → head_H output
  KV Cache: H × seq_len × d_k × 2 tensors

MQA:
  Q_1, K,   V    → head_1 output  ┐
  Q_2, K,   V    → head_2 output  │ all attend to same K, V
  ...                             │
  Q_H, K,   V    → head_H output  ┘
  KV Cache: 1 × seq_len × d_k × 2 tensors  (H× reduction!)

COST: Slight quality degradation (shared K,V is less expressive)
BENEFIT: KV cache reduced by factor H (e.g., 32x for 32-head model)
USED IN: PaLM (5B+), Falcon
```

#### Grouped Query Attention (GQA) — Ainslie et al., 2023

```
IDEA: Middle ground between MHA and MQA
     Divide H query heads into G groups; each group shares one K,V

EXAMPLE: 32 query heads → 8 groups → 8 K,V heads (4.0× KV reduction)

         Q heads (32)        K,V heads (8)
         ─────────────────   ───────────────
         Q_1  Q_2  Q_3  Q_4 → K_1, V_1   (group 1)
         Q_5  Q_6  Q_7  Q_8 → K_2, V_2   (group 2)
         ...
         Q_29..Q_32         → K_8, V_8   (group 8)

KV Cache reduction: H/G = 32/8 = 4×
Quality: Better than MQA (G>1 preserves more diversity), similar to MHA

USED IN: LLaMA-2 (70B model), Mistral-7B, Gemma, Falcon-40B+
Most modern production-grade LLMs use GQA
```

### 1.4 KV Cache for Multi-turn Conversations

```
CHALLENGE: User sends 5 turns of conversation = 5000 tokens of history
           Each new turn must re-attend to all previous turns

SOLUTIONS:
├── Full KV cache: Keep all 5000 tokens' K,V in memory → expensive but exact
├── Sliding Window: Only keep last W tokens' K,V → loses far context
├── H2O (Heavy Hitter Oracle): Keep "important" K,V based on attention scores
│   Track cumulative attention each KV received → evict least-attended ones
├── Streaming LLM (Xiao et al. 2023): Always keep first 4 tokens (initial attention sinks)
│   + recent W tokens → enables infinite context with bounded memory
└── Compression: Summarize old turns into fewer tokens before adding to cache
```

---

## 📄 TOPIC 2: PAGED ATTENTION (vLLM)

### 2.1 The Fragmentation Problem

```
STANDARD KV CACHE APPROACH: Pre-allocate max_seq_len for every request
Request comes in with max_seq_len=2048 → allocate 2048 KV slots → use 150 → 1898 WASTED

PROBLEM: MEMORY FRAGMENTATION — like a hard drive with blocks scattered everywhere

Consider 3 requests with different lengths:
┌─────────────────────────────────────────────────────────────────┐
│ GPU Memory (10GB)                                               │
│ [Request A: 4096 slots allocated, 200 used]│[Request B: 2048]  │
│ [WASTED 3896]                              │[WASTED 1000]      │
│ [Request C: 1024 slots allocated, 50 used] │ [FREE: 1GB]        │
│ [WASTED 974]                               │                   │
└─────────────────────────────────────────────────────────────────┘

Internal fragmentation: 3896 + 1000 + 974 = 5870 slots wasted within requests
External fragmentation: Free 1GB but can't fit new request needing 2GB contiguously

RESULT: GPU utilization ~40-60% of theoretical maximum, high memory waste
```

### 2.2 Paged Attention — The Solution

> Kwon et al., 2023 — "Efficient Memory Management for Large Language Model Serving with PagedAttention"
> The paper behind **vLLM**, the most popular high-throughput LLM serving system.

**Key insight:** Apply OS virtual memory paging concepts to KV cache management.

```
CORE IDEA: Divide KV cache into fixed-size PAGES (like OS memory pages)
Requests get pages allocated ON DEMAND, not pre-allocated contiguously

PAGE: A fixed block of KV slots
      Typical page size: 16 or 32 tokens

BLOCK TABLE: Each request maintains a table mapping logical → physical pages
             Like OS page table for virtual → physical address mapping

             Logical Page 0 → Physical Page 7  (holds tokens 0-15)
             Logical Page 1 → Physical Page 2  (holds tokens 16-31)
             Logical Page 2 → Physical Page 12 (holds tokens 32-47)

Pages can be ANYWHERE in physical memory — no need for contiguous allocation!
```

### 2.3 Paged Attention Memory Operations

```
PAGE ALLOCATION (on new token generation):
┌───────────────────────────────────────────────────────────┐
│ 1. New request arrives → allocate 1 page (for first 16 tok)│
│ 2. After 16 tokens generated → allocate another page      │
│ 3. When request done → FREE all pages → available pool    │
│                                                           │
│ Physical Memory (pages):                                  │
│ [Page 0: FREE] [Page 1: Req_A tok 0-15] [Page 2: Req_B]  │
│ [Page 3: Req_A tok 16-31] [Page 4: FREE] [Page 5: Req_C] │
│                                                           │
│ Req_A's block table: { 0→1, 1→3 } (non-contiguous!)      │
│ Still works — attention gather operation handles this     │
└───────────────────────────────────────────────────────────┘
```

### 2.4 Attention Computation with Paged Memory

```
Standard attention: contiguous K,V tensors → simple matrix multiply
Paged attention: K,V stored in non-contiguous pages → need custom CUDA kernel

PAGED ATTENTION KERNEL:
For each query token q:
  result = 0
  For each physical page p in block_table[request]:
    k_block = load K from page p  (16 tokens × head_dim)
    v_block = load V from page p
    scores  = q · k_block^T / √d_k   (16 scores)
    result += softmax(scores) · v_block   (partial weighted sum)
  Final attention = result (accumulated across all pages)

KEY: Custom CUDA kernel handles page-level scattered memory access
     This was the main technical challenge — existing attention kernels assumed contiguous memory
```

### 2.5 Two Game-Changing Features Enabled by Paged Attention

#### Feature 1: Copy-on-Write for Parallel Sampling

```
SCENARIO: For each user request, generate 4 candidate responses (beam search or parallel sampling)
          4 sequences share the SAME prefix (the user's question)

WITHOUT PAGED ATTENTION:
  Copy full KV cache 4 times → 4× memory for prefix → wasteful

WITH PAGED ATTENTION (Copy-on-Write):
  4 sequences SHARE physical pages for the prefix
  Each process has its OWN block table pointing to SAME pages
  When a sequence diverges (writes new token) → only THEN copy that page

  Prompt KV pages: [Page_0, Page_1, Page_2] (shared by all 4 sequences)
  After 1 generated token: [Page_0, Page_1, Page_2, Page_new_seq1] → seq1's own page
                            [Page_0, Page_1, Page_2, Page_new_seq2] → seq2's own page

MEMORY SAVING: Prompt tokens (often 80-90% of tokens) stored only ONCE
→ Enables beam width of 4-8 vs 1-2 without OOM
```

#### Feature 2: Dynamic Batching

```
PROBLEM: Without Paged Attention, must pre-allocate max_seq_len per request
         → Can't safely add new request if you're not sure it'll fit

WITH PAGED ATTENTION:
  Memory allocated one page at a time
  New request added if ≥ 1 free page exists (even if only 16 tokens of space)
  System can PREEMPT lower-priority requests (evict their pages to CPU) to serve urgent ones

CONTINUOUS BATCHING (Orca, Yu et al. 2022):
  Without: Process full batch → wait for ALL to complete → add new requests
  With:    When ANY sequence finishes → immediately slot in new request
           → GPU never idle waiting for long requests to finish
           → Throughput improves 2-4× vs static batching
```

### 2.6 vLLM Performance Numbers

```
vLLM vs HuggingFace Transformers (LLaMA-13B, A100, same quality):
├── Throughput: 14-24× higher
├── Memory efficiency: ~80% GPU utilization vs 40-60%
├── Latency: Similar or better (continuous batching fills GPU better)
└── Max concurrent requests: 3-5× more before OOM

vLLM is now standard for production LLM serving (Anyscale, Together AI, many companies)
```

---

# PART 2: COMPUTE OPTIMIZATION

---

## ⚡ TOPIC 3: FLASH ATTENTION

### 3.1 The Standard Attention Memory Bottleneck

```
STANDARD ATTENTION COMPUTATION:
Input: Q, K, V matrices each of shape [N × d_k]

Step 1: S = Q · K^T           → [N × N] matrix   HBM read: Q, K (2 × N×d_k × 2 bytes)
                                                  HBM write: S (N² × 2 bytes)
Step 2: P = softmax(S)        → [N × N] matrix   HBM read: S (N² × 2 bytes)
                                                  HBM write: P (N² × 2 bytes)
Step 3: O = P · V             → [N × d_v] matrix HBM read: P, V (N² + N×d_v)
                                                  HBM write: O (N × d_v)

TOTAL HBM TRAFFIC: O(N²) — the N×N attention matrix must be written and read back

For N=2048: N² = 4M elements × 2 bytes = 8MB per head per layer — each time!
For LLaMA-2-70B: 80 layers × 64 heads × 8MB = 40GB of HBM traffic per forward pass
```

**The real bottleneck:** Memory bandwidth, not compute!

```
ROOFLINE MODEL:
GPU has two limits:
├── Compute bound: FLOPs / FLOPS_peak > time to load data
└── Memory bound: FLOPs / FLOPS_peak < time to load data

Standard attention: FLOPs = O(N² × d_k), Memory = O(N²)
FLOPs/Memory = O(d_k) = O(64-128) — LOW → MEMORY BOUND
→ The matrix multiply of O(N²×d_k) is SLOWER than expected because
  loading O(N²) from HBM is the bottleneck, not the arithmetic

SOLUTION: Avoid materializing the N×N attention matrix in HBM
```

### 3.2 Flash Attention — The Core Idea

> Dao et al., 2022 — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

**Key insight:** GPU has a fast on-chip SRAM (shared memory) separate from slow HBM.
If we tile the computation to fit in SRAM, we avoid writing/reading the N×N matrix to HBM.

```
GPU MEMORY HIERARCHY:
HBM (High Bandwidth Memory):  80GB on A100, ~2TB/s bandwidth, SLOW (us latency)
SRAM (L1/Shared memory):      20MB on A100, ~19TB/s bandwidth, FAST (ns latency)

SRAM is 9.5× faster than HBM — if we can fit computation in SRAM, huge speedup!

CHALLENGE: N×N attention matrix doesn't fit in SRAM (for N=2048, 8MB per head)
SOLUTION: Compute attention in TILES that fit in SRAM
          Use mathematical tricks to compute exact softmax without the full matrix
```

### 3.3 Flash Attention Algorithm — Step by Step

```
TILED ATTENTION (simplified):

CORE MATHEMATICAL TRICK: Online softmax (Milakov & Gimelshein, 2018)
Can compute softmax incrementally without seeing all values first:

Standard softmax: m = max(x), softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)
Must see ALL x_j to compute Σ_j exp(x_j - m)

ONLINE SOFTMAX:
Process blocks of x:
Block 1: x_1..x_B1
  m_1 = max(x_1..x_B1)
  ℓ_1 = Σ_{i=1}^{B1} exp(x_i - m_1)
  o_1 = Σ_{i=1}^{B1} exp(x_i - m_1) · v_i

Block 2: x_{B1+1}..x_{B1+B2}
  m_2 = max(x)  [update running max]
  ℓ_2 = exp(m_1 - m_2) × ℓ_1  +  Σ_i exp(x_i - m_2)  [rescale old ℓ]
  o_2 = exp(m_1 - m_2) × o_1  +  Σ_i exp(x_i - m_2) × v_i  [rescale old partial result]

Final: output = o_final / ℓ_final  [normalize by accumulated denominator]

KEY: We can compute EXACT softmax while processing one tile at a time!
No need to store the full N×N matrix
```

### 3.4 Flash Attention Tiling Algorithm

```
ALGORITHM:

Outer loop over Q tiles (Q_block) — loaded from HBM once:
  Inner loop over K,V tiles (K_block, V_block) — loaded from HBM sequentially:
    1. Load Q_block, K_block, V_block into SRAM        [HBM → SRAM]
    2. Compute S_block = Q_block · K_block^T / √d_k    [in SRAM]
    3. Update running max m_block and running sum ℓ_block [in SRAM]
    4. Compute partial output O_block += softmax_partial · V_block [in SRAM]
  Write O_block back to HBM when done with all K,V tiles [SRAM → HBM]

MEMORY:
Stored in HBM: Q, K, V, O (linear in N) — NOT the N×N attention matrix!
Stored in SRAM: Tiles of Q_block, K_block, V_block + running stats

IO COMPLEXITY:
Standard attention: O(N² × d_k) HBM reads/writes
Flash Attention:    O(N × d_k²/M) where M = SRAM size
For M >> d_k (typical): vastly fewer HBM reads — often 5-20× less IO
```

### 3.5 Flash Attention Backward Pass

```
CHALLENGE FOR TRAINING:
Backpropagation through attention needs the attention matrix P for gradient computation
We never stored P (that's the whole point) — how do we backprop?

SOLUTION: RECOMPUTE attention in backward pass using stored softmax statistics

Forward pass stores: output O, softmax normalizer ℓ (one scalar per query row)
Backward pass: from O, ℓ, and Q, K, V → recompute P on-the-fly during backward

COST: Extra FLOPs for recomputation in backward
BENEFIT: No need to store N×N matrix for gradient → huge memory saving in training
```

### 3.6 Flash Attention 2 (Dao, 2023)

```
Improvements over FA-1:
1. FEWER NON-MATMUL FLOPS: Reorganize algorithm to maximize time in tensor cores
   (tensor cores do matmuls fast; other ops like softmax are slower)
2. PARALLELISM: Parallelize across sequence length dimension (not just batch × heads)
   → Better GPU utilization for long sequences / small batches
3. WORK PARTITIONING: Smarter split of Q,K,V work across warps (GPU thread groups)
   → Fewer shared memory access conflicts

RESULT vs FA-1:
  Forward: 2× faster
  Backward: ~2.5× faster
  Can handle seq_len up to 256K tokens on A100 (with proper chunking)
```

### 3.7 Flash Attention 3 (2024)

```
Targets H100 (Hopper architecture) specifically:
1. WGMMA (Warpgroup-level Matrix Multiply): Larger matmul granularity → better efficiency
2. TMA (Tensor Memory Accelerator): Hardware-level async memory transfers
3. FP8 support: 2× more compute throughput than FP16 on H100
4. 3-stage pipeline: Overlap compute + memory transfer + WGMMA → hide latency

RESULT: ~75% of H100 theoretical peak FLOPS utilization (industry-leading)
```

---

# PART 3: MODEL COMPRESSION

---

## 🔢 TOPIC 4: QUANTIZATION — SMALLER WEIGHTS, FASTER INFERENCE

### 4.1 Why Quantize?

```
STORAGE AND BANDWIDTH PROBLEM:
  LLaMA-2-70B in FP32: 70B × 4 bytes = 280 GB → 4× A100 80GB needed
  LLaMA-2-70B in FP16: 70B × 2 bytes = 140 GB → 2× A100 80GB needed
  LLaMA-2-70B in INT8:  70B × 1 byte  =  70 GB → 1× A100 80GB needed ✓
  LLaMA-2-70B in INT4:  70B × 0.5 bytes = 35 GB → 1× smaller GPU ✓

COMPUTE BENEFIT:
  INT8 matrix multiply is 2× faster than FP16 on most hardware
  INT4 is 4× faster (uses Tensor Core's INT4 throughput)
  HBM bandwidth: less data to read → proportional speedup

QUANTIZATION GOAL:
  Represent FP32 weights (32-bit floating point) as INT8 or INT4 (integer)
  With minimal loss in model quality
```

### 4.2 Quantization Basics — The Math

```
UNIFORM QUANTIZATION:
  Real value x (FP32) → quantized value q (INT8)

  q = round(x / scale) + zero_point

  scale = (x_max - x_min) / (q_max - q_min)
        = (x_max - x_min) / 255            [for INT8: 0..255]

  zero_point = round(-x_min / scale)

DEQUANTIZATION (for actual compute):
  x_reconstructed = scale × (q - zero_point)

  Quantization error: ε = x - x_reconstructed  (rounding error)

TYPES BY WHEN YOU QUANTIZE:
├── Post-Training Quantization (PTQ): Quantize after training — no retraining needed
│   Fast, but quality can drop more
│
└── Quantization-Aware Training (QAT): Simulate quantization during training
    Model adapts to quantization noise → better quality, requires training compute
```

### 4.3 Weight Quantization vs Activation Quantization

```
WEIGHT QUANTIZATION (W-only):
  Quantize weight matrices W to INT8/INT4
  During inference: dequantize W back to FP16 for compute, then requantize
  OR: keep quantized + use INT8/INT4 matmul units

  ADVANTAGE: Easy — weights are static (known at quantization time)
  The distribution of weights is Gaussian → easy to set scale

ACTIVATION QUANTIZATION (W8A8 = weights and activations both INT8):
  Quantize BOTH weights AND activations
  Enables INT8 matmul throughout → maximum speedup

  CHALLENGE: Activations are DYNAMIC (different for each input)
  Must estimate scale at runtime (dynamic quantization) or pre-determine (static)

  PROBLEM: OUTLIER ACTIVATIONS
  Some activation dimensions have values 10-100× larger than typical
  These outliers force large scale values → poor precision for normal values
  Example: OPT/BERT models have ~0.1% of dimensions with huge outliers
```

### 4.4 LLM.int8() — The First Practical 8-bit LLM

> Dettmers et al., 2022 — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"

```
PROBLEM: Activation outliers break naive INT8 quantization for large LLMs
         70B+ parameter models have significant outliers (smaller models don't)

INSIGHT: The outlier dimensions are always the SAME dimensions across tokens
         (not random) → we can identify them ahead of time

MIXED-PRECISION DECOMPOSITION:
  1. Identify outlier feature dimensions (threshold: |activation| > 6.0)
  2. Split matrix multiplication into two parts:
     ├── Outlier dimensions (typically ~0.1%): Compute in FP16 (full precision)
     └── Normal dimensions (99.9%): Compute in INT8
  3. Combine results

  XW = [X_outlier × W_outlier]_{FP16}  +  [X_normal × W_normal]_{INT8}

RESULT:
  99.9% of multiplications use INT8 → 2× speedup, 2× less memory
  0.1% uses FP16 → maintains quality
  Enables 70B model on 1× A100 instead of 2× A100
  Quality loss: < 1% perplexity increase
```

### 4.5 GPTQ — Post-Training Quantization to 4-bit

> Frantar et al., 2022 — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

```
GOAL: 4-bit quantization with minimal quality loss

CORE ALGORITHM: Optimal Brain Quantization (OBQ) applied to LLMs

IDEA:
  Naive 4-bit: round each weight independently → large errors
  GPTQ: When you round weight w_i, use a compensation term to update other weights
        to partially correct for the introduced error

  For each weight w_q being quantized:
    w_q_quantized = quant(w_q)
    error = w_q - w_q_quantized
    W_remaining -= (error / H_qq) × H_q:   [update other weights to compensate]
    H = Fisher information matrix (second-order curvature information)

PRACTICAL IMPLEMENTATION:
  Process columns of weight matrix one by one (or in groups)
  Each column quantization uses Cholesky decomposition of H for efficiency
  Total: ~10-30 minutes to quantize 70B model on 1 GPU (one-time cost)

QUALITY:
  4-bit GPTQ LLaMA-65B ≈ FP16 LLaMA-30B in perplexity
  Significant capability retention at 4× compression
```

### 4.6 GGUF / llama.cpp — 4-bit with CPU Support

```
GGUF FORMAT: Universal model format used by llama.cpp
  Supports mixed-precision: some layers in Q4, some in Q6 or Q8
  Designed for inference on consumer hardware (CPU, Apple M1/M2, no datacenter GPU)

QUANTIZATION LEVELS IN GGUF:
  Q2_K:   2-bit (aggressive, significant quality loss)
  Q4_K_M: 4-bit, K-quant (medium variant) — best quality/size tradeoff
  Q5_K_M: 5-bit K-quant — better quality, 25% larger
  Q6_K:   6-bit — very close to full precision
  Q8_0:   8-bit — near-lossless

"K-quant" (K = Super-block quantization):
  Group weights into "super-blocks"
  Quantize most weights to Q4, but use Q6/Q8 for certain important layers
  (embedding layer, attention output projections get higher precision)

USED BY: llama.cpp (runs on Mac, CPU, Raspberry Pi!), Ollama, LM Studio
```

### 4.7 AWQ — Activation-Aware Weight Quantization

> Lin et al., 2023 — "AWQ: Activation-aware Weight Quantization for LLM Compression"

```
KEY INSIGHT: Not all weights are equally important
             Weights corresponding to HIGH ACTIVATION channels matter more
             (if activation x_i is large, error in W_ij has big impact on output)

METHOD:
  1. Run calibration data through model → measure per-channel activation magnitude
  2. SCALE important channels up (before quantization) to reduce their relative error
  3. Then quantize all channels → important channels get effectively higher precision
  4. Un-scale after quantization (equivalent scaling in next layer)

FORMALLY: For important channel i with scaling factor s_i:
  W̃_i = W_i / s_i    (scale down weight i)
  X̃_i = X_i × s_i    (scale up corresponding activation)
  W̃_i × X̃_i = W_i × X_i   (mathematically equivalent)
  But W̃_i = W_i/s_i can be quantized with less error when s_i > 1

ADVANTAGE vs GPTQ: No need for Hessian (computationally cheaper calibration)
QUALITY: Comparable to GPTQ, often better on edge cases
USED IN: TinyChat, AutoAWQ library, many Hugging Face quantized models
```

### 4.8 SmoothQuant — W8A8 with Outlier Migration

> Xiao et al., 2022 — Enables W8A8 (both weights AND activations to INT8)

```
PROBLEM: Activation outliers make W8A8 hard (huge outliers force high scale → low precision elsewhere)
SOLUTION: MIGRATE difficulty from activations to weights (which are easier to quantize)

MATH:
  Y = X × W   (standard matmul)

  Equivalent: Y = (X / s) × (W × s)   [multiply and divide by scale s]
                   X̃         W̃

  Choose s: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)   (j = channel index, α ∈ [0,1])

  Effect: X̃ = X/s (activations become smoother — outliers divided by s)
          W̃ = W×s (weights become slightly more extreme — but weights are static!)

  α=0: all difficulty stays in activations (same as before)
  α=0.5: balanced migration
  α=1: all difficulty moves to weights (fully migrated)

SWEET SPOT: α=0.5 works for most models
RESULT: Enables W8A8 INT8 matmul throughout → 2× speedup vs FP16 with < 1% quality loss
SUPPORTED IN: TensorRT-LLM from NVIDIA (used in production NLP serving)
```

---

# PART 4: HARDWARE-AWARE INFERENCE SYSTEMS

---

## 🚀 TOPIC 5: TENSORRT-LLM (NVIDIA) & TURBOMIND (LMDeploy)

### 5.1 TensorRT-LLM

```
WHAT IT IS: NVIDIA's production LLM inference library
            (Layer above PyTorch, optimized for NVIDIA GPUs)

KEY OPTIMIZATIONS INCLUDED:
├── Flash Attention (FA-2/FA-3) for H100
├── In-flight batching (continuous batching)
├── INT8 / FP8 quantization (W8A8 via SmoothQuant integration)
├── Fused multi-head attention + MLP kernels
├── Multi-GPU tensor parallelism (split one model across GPUs)
├── Pipeline parallelism (different layers on different GPUs)
└── Auto-tuned GEMM (matrix multiply) kernels per GPU model

WORKFLOW:
FP32/FP16 PyTorch model
    ↓
TensorRT-LLM compilation (builds optimal CUDA execution graph)
    ↓
TRT Engine (hardware-specific, GPU-version-specific binary)
    ↓
Serve with Triton Inference Server (NVIDIA's model serving framework)

SPEEDUP vs Naive PyTorch: 2-5× depending on model and hardware
```

### 5.2 TurboMind (Part of LMDeploy)

```
WHAT IT IS: Alibaba's LLM inference engine (optimized for large models)
            Used internally at Alibaba Cloud; open-sourced as LMDeploy

KEY FEATURES:
├── Persistent batch: Smart request scheduling + continuous batching
├── Blocked KV cache: Similar to Paged Attention; blocks of fixed size
├── INT4-/INT8 weight quantization (AWQ integration)
├── Tensor parallelism: Split QKV projections across GPUs
├── Optimized attention kernels (custom CUDA)
└── Supports: LLaMA, QWen, Baichuan, InternLM architectures

BENCHMARK (LLaMA-13B, A100, throughput):
  HuggingFace:    ~60 tokens/s
  vLLM:           ~350 tokens/s
  TurboMind:      ~480 tokens/s (16% faster than vLLM in some configs)
```

---

## 🔄 TOPIC 6: SPECULATIVE DECODING — DETAILED DEEP DIVE

### 6.1 The Core Problem: Memory-Bandwidth Bound Decoding

```
AUTOREGRESSIVE DECODING IS COMPUTE-INEFFICIENT:

At each step: generate 1 token
  → Load all 140GB model weights (once per step)
  → Perform matrix multiplications
  → Output: probability over 32K vocab → sample 1 token

GPU is MEMORY-BANDWIDTH BOUND during token generation:
  FLOPs per step = 2 × model_params = 2 × 70B = 140G FLOPs
  Memory traffic = 140 GB (load all weights)
  Arithmetic intensity = 140G FLOPs / 140 GB = 1 FLOP/byte
  GPU optimal intensity = 100+ FLOP/byte → 100× underutilized!

WHY: We process ONE token at a time → too little work to justify loading all weights
     If we processed 100 tokens at once → 100 FLOP/byte → fully utilized

BATCH PROCESSING SAVES: If 100 different users' tokens are batched → full utilization
BUT: For a single user, can't increase batch→ tokens must be generated serially
SPECULATIVE DECODING: Find a way to verify MULTIPLE tokens in ONE forward pass
```

### 6.2 Speculative Decoding — Full Algorithm

> Chen et al., 2023 (Google); Leviathan et al., 2023 (Google)

```
SETUP:
├── Draft model M_q: Small, fast (e.g., 7B params)
└── Target model M_p: Large, slow (e.g., 70B params)

GOAL: Generate tokens with distribution IDENTICAL to M_p
      But using M_q to amortize M_p's cost

ALGORITHM:

Step 1: DRAFT — Generate K tokens with small model
  x_1 = M_q.sample(prefix)           → draft token 1
  x_2 = M_q.sample(prefix + x_1)     → draft token 2
  ...
  x_K = M_q.sample(prefix + x_{1..K-1}) → draft token K
  
  Cost: K fast forward passes through M_q
  M_q stores draft token probabilities: q(x_k | prefix + x_{<k})

Step 2: VERIFY — Score all K draft tokens with target model IN PARALLEL
  p(x_1..x_K | prefix) = M_p.forward(prefix + [x_1, x_2, ..., x_K])
  
  ONE forward pass → probabilities for ALL K positions simultaneously!
  Cost: 1 slow forward pass through M_p (but processing K+1 tokens in parallel)
  
  NOTE: This is NOT autoregressive — M_p scores all K tokens simultaneously
        using causal masked attention (each position only sees previous positions)

Step 3: ACCEPTANCE-REJECTION — Which draft tokens to keep?
  
  For token k = 1, 2, ..., K:
    r = Uniform[0, 1]
    if r < p(x_k | ...) / q(x_k | ...):   [acceptance probability]
      Accept x_k — keep token
    else:
      Reject x_k — stop; sample replacement from adjusted distribution
      break

  If token k is rejected, sample corrected token from:
    p'(x) = normalize(max(0, p(x) - q(x)))   [residual distribution]

Step 4: BONUS TOKEN
  If ALL K draft tokens accepted → M_p provides K+1th token "for free"
  (it was already computed in the parallel verification pass)

GUARANTEES:
  • Exact same distribution as sampling from M_p alone
  • Each step generates 1 to K+1 tokens (vs exactly 1 for standard decoding)
```

### 6.3 Speculative Decoding — Acceptance Rate Analysis

```
EXPECTED TOKENS PER STEP:
  If draft acceptance rate is α (probability each draft token is accepted):
  E[tokens per step] = K × α^K + ... from acceptance-rejection math
                     ≈ K × α  for small K
  
  More precisely: E[tokens] = (1 - α^{K+1}) / (1 - α)

SPEEDUP FACTOR:
  Without speculative decoding: 1 token per M_p pass
  With speculative decoding:    E[tokens] tokens per (K × M_q pass + 1 × M_p pass)

  Speedup ≈ (1 - α^{K+1}) / ((1 - α) × (c + 1))
  where c = cost ratio M_q / M_p

  For α=0.8, K=4, c=0.1 (M_q is 10% cost of M_p):
  Speedup ≈ (1 - 0.8^5) / (0.2 × (0.4 + 1)) ≈ 2.7×

REAL-WORLD NUMBERS:
  GPT-4-level model + GPT-3.5-level draft model: ~2-3× speedup
  High acceptance rate (domain-specific pair): up to 4-5×

WHEN ACCEPTANCE IS HIGH:
│ ✓ Target and draft model from same family (LLaMA-70B + LLaMA-7B)
│ ✓ Predictable output pattern (code, structured output, copying)
│ ✓ High temperature in draft, lower in target (draft is aggressive, target selective)

WHEN ACCEPTANCE IS LOW:
│ ✗ Very creative/random output (high temperature targets)
│ ✗ Draft model is too small/weak vs target
│ ✗ Out-of-domain inputs
```

### 6.4 Medusa Heads — Speculative Decoding Without a Draft Model

> Cai et al., 2024

```
PROBLEM: Need to maintain two separate models (draft + target) → complexity

MEDUSA: Add extra "heads" to the SAME model to predict future tokens

┌─────────────────────────────────────────────────────────┐
│  Standard LM head:   h_t → softmax → P(x_1 | prefix)   │
│                                                          │
│  Medusa head 1:      h_t → MLP → softmax → P(x_2 | ..) │
│  Medusa head 2:      h_t → MLP → softmax → P(x_3 | ..) │
│  Medusa head 3:      h_t → MLP → softmax → P(x_4 | ..) │
└─────────────────────────────────────────────────────────┘

INFERENCE:
1. Forward pass → h_t produced
2. LM head produces x_1 draft
3. Medusa heads simultaneously produce x_2, x_3, x_4 drafts (from same h_t)
4. Build "tree" of possible token continuations from all combinations
5. Run ONE verification pass with causal masking over the tree
6. Accept/reject via tree-based sampling

ADVANTAGE: No separate draft model needed — single model inference
TRAINING: Only Medusa heads are trained (main model frozen)
          Requires 1-2% extra compute during training
SPEEDUP: 2-2.5× (slightly less than ideal speculative decoding, but simpler)
```

### 6.5 Self-Speculative Decoding (Layer Skipping)

```
IDEA: Use the SAME model but with fewer layers as the "draft"

Early exit: Skip layers 15-32 of LLaMA-33B → use only layers 1-14 as draft model
Full model: All 40 layers as target for verification

IMPLEMENTATION:
1. Draft: Run input through layers 1-14 → generate K tokens (2× faster)
2. Verify: Run same input through all 40 layers with K tokens as context
3. Accept/reject

ADVANTAGE: No separate model deployment; shared weights
SPEEDUP: ~1.5-2× (less than separate model, but zero extra memory)
```

---

# PART 5: BATCHING & SCHEDULING STRATEGIES

---

## 📦 TOPIC 7: BATCHING STRATEGIES FOR HIGH THROUGHPUT

### 7.1 Static Batching

```
TRADITIONAL APPROACH:
  Collect N requests
  Run N requests as a batch
  Wait for ALL N to finish
  Collect next batch

              Request A (100 tok, done)  XXXXX done  ←─────  GPU IDLE WAITING ─────┐
              Request B (200 tok)        XXXXXXXXXXXXXXXXXX done                     │
              Request C (300 tok)        XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX done  ←──────┘
              
              ├── Batch 1 ──────────────────────────────────────────────────────────┤

              New requests can only start AFTER all of batch 1 finishes!
              If one request is very long → ALL subsequent requests wait
```

### 7.2 Continuous Batching (Orca)

```
IDEA: Treat each ITERATION (forward pass) as an opportunity to add/remove requests

  Step 1:  Process [Req_A, Req_B, Req_C] → generate token for each
  Step 2:  Process [Req_A, Req_B, Req_C] → Req_A finishes!
           → IMMEDIATELY insert Req_D into the batch
  Step 3:  Process [Req_B, Req_C, Req_D] → no waiting!
  ...

ITERATION-LEVEL SCHEDULING:
  After each forward pass:
  1. Check if any sequences finished (hit EOS token)
  2. Free their KV cache pages
  3. Admit new requests from queue (if memory available)
  4. Continue with updated batch

BENEFIT:
  GPU never idle between requests
  Short requests don't block behind long ones
  Throughput improvement: 2-4× vs static batching

REAL-TIME PERFORMANCE:
  vLLM implementation: can serve 20-50 concurrent streaming users on A100
  With static batching: maybe 5-10
```

### 7.3 Chunked Prefill

```
PROBLEM: Long user prompts (1000 tokens) monopolize GPU for many steps
         Other requests waiting = high time-to-first-token for queued requests

CHUNKED PREFILL:
  Split long prompt into chunks (e.g., 256 tokens at a time)
  Interleave chunks with generation steps of OTHER requests

Example:
  Request A (1000 token prompt) + Request B (generating tokens)

  Without chunked prefill:
    Step 1-4: Process Req_A's 1000 tokens (4 chunks of 256)  → Req_B waits 4 steps
    Step 5+:  Process both Req_A generation and Req_B generation
    → Req_B's latency increased by 4 steps

  With chunked prefill:
    Step 1: [Req_A chunk 1 (256 tok)] + [Req_B generate 1 tok]
    Step 2: [Req_A chunk 2 (256 tok)] + [Req_B generate 1 tok]
    Step 3: [Req_A chunk 3 (256 tok)] + [Req_B generate 1 tok]
    Step 4: [Req_A chunk 4 (256 tok)] + [Req_B generate 1 tok]
    Step 5: [Req_A generate] + [Req_B generate]
    → Req_B's latency NOT increased; Req_A prefill takes same time but shared

TRADE-OFF: Slight throughput cost (mixing prefill + decode is less efficient)
           vs better tail latency (p99 time-to-first-token)
```

---

# PART 6: PARALLELISM FOR MULTI-GPU SERVING

---

## 🖥️ TOPIC 8: TENSOR PARALLELISM & PIPELINE PARALLELISM

### 8.1 Why We Need Multi-GPU for Large Models

```
LLAMA-2-70B:   140GB FP16 weights → 2× A100 80GB minimum
LLAMA-2-70B + KV cache (batch=16, seq=4096): 140 + 50GB = 190GB → 3× A100 minimum
GPT-3-175B:   350GB → 5× A100 minimum

PARALLELISM TYPES:
├── Tensor Parallelism (TP): Split weight matrices across GPUs (within a layer)
├── Pipeline Parallelism (PP): Split layers across GPUs (different GPUs = different layers)
└── Data Parallelism (DP): Same model replicated; different data batches per GPU
```

### 8.2 Tensor Parallelism (Megatron-LM Style)

```
SPLIT THE ATTENTION/MLP WEIGHT MATRICES:

For a linear layer Y = XW (X: [batch × d_model], W: [d_model × d_out]):

GPU 0: W_0 = W[:, :d_out/2]   → computes Y_0 = X × W_0   [partial output, first half]
GPU 1: W_1 = W[:, d_out/2:]   → computes Y_1 = X × W_1   [partial output, second half]
AllGather: Y = [Y_0, Y_1]     [combine across GPUs]

For Attention MHA (split over heads):
GPU 0: head_0 to head_H/2    (W_Q_0, W_K_0, W_V_0 for first half of heads)
GPU 1: head_H/2 to head_H    (W_Q_1, W_K_1, W_V_1 for second half)
→ Each GPU computes attention for its heads independently
→ AllReduce after output projection to combine

COMMUNICATION COST:
AllReduce after each layer → high bandwidth requirement (NVLink preferred)
NVLink: 600GB/s → TP works well within a single node (8 GPUs with NVLink)
Across nodes (PCIe/InfiniBand): TP too expensive → use Pipeline Parallelism
```

### 8.3 Pipeline Parallelism

```
SPLIT LAYERS ACROSS GPUS:

GPU 0: Embedding + Layers 0-19
GPU 1: Layers 20-39
GPU 2: Layers 40-59
GPU 3: Layers 60-79 + LM head

MICRO-BATCH PIPELINING (GPipe / PipeDream):
                   t=1    t=2    t=3    t=4
GPU 0 (Layer 0-19): M1     M2     M3     M4
GPU 1 (Layer 20-39):        M1     M2     M3
GPU 2 (Layer 40-59):               M1     M2
GPU 3 (Layer 60-79):                       M1

BUBBLE TIME (idle time):
  Without optimization: 3/7 = 43% bubble (3 GPUs waiting for first micro-batch to propagate)
  With pipeline schedule (1F1B): reduce bubble to (p-1)/(m+p-1) where p=pipeline stages, m=microbatches
  For p=4, m=8: bubble = 3/11 ≈ 27%

COMBINED TP + PP:
  LLaMA-2-70B on 8 A100s: TP=4 (within node), PP=2 (across pipeline stages)
  Handles models > single-node memory efficiently
```

---

# PART 7: QUICK REFERENCE SUMMARY TABLE

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    LLM INFERENCE OPTIMIZATION — MASTER REFERENCE                     │
├─────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Technique               │ What It Solves            │ Key Metric                       │
├─────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ KV Cache (basic)        │ Avoid recomputing K,V     │ ~N× speedup over no cache        │
│ MQA                     │ Reduce KV cache size      │ num_heads× KV reduction          │
│ GQA                     │ Balance MHA vs MQA        │ num_heads/groups× KV reduction   │
│ Paged Attention (vLLM)  │ KV memory fragmentation   │ 14-24× throughput vs HuggingFace │
│ Continuous Batching     │ GPU idle between requests │ 2-4× throughput vs static batch  │
│ Chunked Prefill         │ Long prompt latency spikes│ Reduces p99 TTFT by 30-50%       │
│ Flash Attention 2       │ O(N²) attention memory    │ 2-4× speedup, O(N) memory        │
│ Flash Attention 3       │ H100 utilization           │ ~75% H100 peak FLOPS             │
│ INT8 (LLM.int8())       │ FP16 memory/bandwidth     │ 2× less memory, 1.5-2× speedup   │
│ GPTQ (INT4)             │ GPU memory for large LLM  │ 4× less memory, slight quality ↓ │
│ AWQ                     │ INT4 with better quality  │ Similar to GPTQ, easier calib.   │
│ SmoothQuant (W8A8)      │ Activation outliers       │ Enables full INT8, 2× speedup    │
│ Speculative Decoding    │ Single-user throughput    │ 2-4× speedup, exact distribution │
│ Medusa Heads            │ Spec decode without 2nd model│ 2-2.5× speedup, single model  │
│ Tensor Parallelism      │ Model too large for 1 GPU │ Scales to N GPUs linearly (TP)   │
│ Pipeline Parallelism    │ Multi-node large model    │ Scales across nodes (PP)         │
│ TensorRT-LLM           │ All of the above (NVIDIA) │ 2-5× vs PyTorch naive            │
│ vLLM                    │ All above (open source)   │ 14-24× vs HuggingFace            │
└─────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## 🎯 INTERVIEW Q&A — EXPECT THESE FROM Alex

### Q1: "Walk me through why Flash Attention is faster. What's the bottleneck it solves?"

> "The bottleneck in standard attention isn't compute — it's memory bandwidth. When we compute `QK^T`, we produce an N×N attention matrix that must be written to GPU HBM memory and read back for softmax, then written again, then read back for the V multiply. For N=2048, this is 8MB per head per layer — and with 80 layers × 64 heads, it's tens of gigabytes of HBM traffic per forward pass.
>
> Flash Attention avoids materializing this matrix at all. It tiles Q, K, V into blocks that fit in on-chip SRAM (which is ~9.5× faster bandwidth than HBM), and uses the online softmax trick — maintaining a running max and normalization factor — to compute exact attention incrementally without storing intermediate results.
>
> The result: O(N) HBM memory usage instead of O(N²), and 2-4× wall-clock speedup because the arithmetic intensity increases from ~1 FLOP/byte to the GPU's sweet spot. Flash Attention 2 then adds better parallelism across sequence length and fewer non-matmul operations — getting to ~75% of A100 theoretical throughput."

---

### Q2: "How does Paged Attention differ from standard KV cache, and why does it matter for serving?"

> "Standard KV cache pre-allocates a contiguous block of max sequence length for each request. Because sequences finish at different lengths, you get internal fragmentation — every request wastes the slots from its actual length to max length. More critically, you can't support copy-on-write for parallel sampling, because the pages aren't tracked independently.
>
> Paged Attention borrows the OS virtual memory model. Physical GPU memory is divided into fixed-size pages (say, 16 tokens each). Each request has a block table — a logical-to-physical page mapping. Pages are allocated on demand, one at a time, so there's no pre-allocation waste. When a request finishes, its pages are immediately freed and available for new requests.
>
> This enables three critical features for production: first, near-zero memory fragmentation — GPU utilization goes from 40-60% to ~80%. Second, copy-on-write: four beam search candidates can share physical pages for the prompt and only diverge when generating different tokens. Third, continuous batching — because memory is managed at page granularity, new requests can be admitted immediately when a page frees up, rather than waiting for a full sequence slot."

---

### Q3: "Explain speculative decoding — what's the acceptance criterion and why does it produce identical output to the target model?"

> "The bottleneck in single-user LLM inference is that we load all 140GB of a 70B model's weights for each token — but we only generate one token, so the compute-to-memory ratio is ~1 FLOP/byte versus the GPU's optimal 100+. Speculative decoding fixes this by verifying multiple tokens in a single target model forward pass.
>
> The algorithm: a small draft model generates K tokens speculatively with probabilities q(x_k). The large target model does ONE forward pass over the K draft tokens, yielding target probabilities p(x_k). For each draft token k, we accept it with probability min(1, p(x_k)/q(x_k)). If the target assigns higher probability than the draft, we always accept. If the target assigns lower probability, we accept probabilistically to correct for the discrepancy.
>
> The key theoretical result: this acceptance-rejection sampling scheme produces tokens that are exactly distributed according to p — the target model's distribution. Intuition: when we do accept a token sampled from q, we've reweighed to match p. When we reject, we sample from the residual distribution max(0, p - q), normalized. The two cases together recover p exactly.
>
> For Flipkart's use case, where SLAP generates structured product recommendations with fairly predictable patterns, acceptance rates can be 80%+ — giving us 2.5-3× throughput improvement at identical quality."

---

### Q4: "If your 70B LLM response latency is too high, what levers do you pull and in what order?"

```
SYSTEMATIC OPTIMIZATION LADDER:

1. First: PROFILE — Identify the bottleneck
   Is it prefill-bound (long prompts) or decode-bound (long generation)?
   Memory-bound or compute-bound?

2. KV CACHE: Ensure it's implemented (most frameworks do this by default)
   Verify no unnecessary KV evictions for your use case

3. ATTENTION EFFICIENCY: Use Flash Attention 2 (or 3 on H100)
   Free speedup if not already using it

4. BATCHING: Switch to continuous batching (vLLM / TRT-LLM)
   Immediate 2-4× throughput for multi-user scenarios

5. QUANTIZATION: Profile quality tolerance
   INT8 (LLM.int8() or SmoothQuant): near-lossless, 2× speedup
   INT4 (GPTQ or AWQ): moderate quality loss, 4× less memory

6. ARCHITECTURE: GQA/MQA to reduce KV cache size
   (requires model modification or using a model already trained with GQA)

7. SPECULATIVE DECODING: If single-user latency is critical
   Find a good draft model from same family (10-15% size of target)

8. HARDWARE: Scale out with tensor parallelism
   Multiple GPUs with NVLink for single-request latency

9. DISTILLATION: If quality budget allows, use smaller model
   LLaMA-2-13B often reaches LLaMA-2-70B quality on specific domains after fine-tuning

AT FLIPKART (350M users):
  Priority: Throughput per GPU-hour (cost) > single-request latency
  Best combo: vLLM + GQA + INT8 quantization + continuous batching
  = ~5-8× cost reduction vs naive HuggingFace serving with minimal quality impact
```

---

---

# PART 8: REASONING IN LLMs — HOW MODELS "THINK"

> Why it matters for Flipkart: SLAP agent plans multi-step shopping decisions.
> Fraud investigator reasons over evidence chains. Seller AI decomposes business problems.
> Understanding HOW reasoning emerges — and fails — is critical for building reliable agentic systems.

---

## 🧩 TOPIC 9: WHAT IS "REASONING" IN AN LLM?

### 9.1 The Fundamental Question

```
Standard LLM call:
  Input:  "What is 17 × 23?"
  Output: "391"   ← Direct answer, single forward pass

Does the model "reason"? Or just pattern-match?

EVIDENCE FOR PATTERN MATCHING:
  "What is 17 × 24?" → might give wrong answer (not in training patterns)
  Change surface form: "17 * 23 = ?" → different accuracy than "17 × 23 = ?"
  → Model memorized surface patterns, not arithmetic procedure

EVIDENCE FOR REASONING:
  With chain-of-thought: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391" → correct
  Models can generalize to novel multi-step problems when shown HOW to break them down
  → The approach to thinking matters, not just the final answer

CURRENT UNDERSTANDING (2024):
  Transformers perform "System 1" thinking by default (fast, pattern-matching)
  Deliberate prompting / extended compute → approximate "System 2" (slow, deliberate)
  Reasoning models (o1, DeepSeek-R1) systematize this extended computation
```

### 9.2 System 1 vs System 2 Thinking (Kahneman Framework Applied to LLMs)

```
  SYSTEM 1 (Fast Thinking — standard LLM generation):
  ├── Single forward pass per token
  ├── Pattern matching from training distribution
  ├── Works well for: common facts, simple QA, standard phrases
  └── Fails on: novel multi-step problems, logical chains, math

  SYSTEM 2 (Slow Thinking — reasoning LLMs):
  ├── Multiple forward passes; many tokens of "thinking" before answering
  ├── Systematic decomposition, backtracking, self-correction
  ├── Works well for: math, code, logical reasoning, complex planning
  └── Cost: 10-100× more tokens → 10-100× more compute and latency

  GPT-3.5/4: Mostly System 1 with some System 2 via prompting
  o1/o3, DeepSeek-R1: Trained System 2 — extended thinking is learned behavior
```

---

## 🔗 TOPIC 10: CHAIN-OF-THOUGHT (CoT) — THE FOUNDATION

### 10.1 What CoT Is and Why It Works

> Wei et al., 2022 (Google) — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

```
STANDARD PROMPTING:
  Q: "Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many?"
  A: "11"   ← LLM jumps to answer; often wrong on harder variants

CHAIN-OF-THOUGHT PROMPTING:
  Q: "Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many?"
  A: "Roger starts with 5 balls. He buys 2 × 3 = 6 more balls.
      In total: 5 + 6 = 11 balls."

KEY: By WRITING OUT intermediate steps, the model is forced to:
  1. Allocate separate tokens to each reasoning step
  2. Condition each new step on previously correct steps
  3. Catch contradictions earlier in the chain
```

### 10.2 WHY CoT Works — The Mechanistic Explanation

```
THE SERIAL COMPUTATION HYPOTHESIS:
  A transformer has fixed depth (e.g., 96 layers)
  For a single token, the model can do at most 96 "operations"
  Complex reasoning requires MORE than 96 sequential operations

  With CoT: Each generated token gets its own 96-layer computation path
  A 100-step chain-of-thought → 96 × 100 = 9600 effective "operations"
  → Exponentially more representational power for complex problems

  Formally: Problems solvable in sequential time T can be solved with O(T) tokens
            even if no single forward pass can solve them directly

THE COMMUNICATION CHANNEL HYPOTHESIS:
  Attention allows arbitrary information routing between positions
  Intermediate reasoning steps written to context act as "working memory"
  Each new step can attend to and build on ALL previous steps
  → The context becomes external scratchpad memory
```

### 10.3 CoT Variants

#### Zero-Shot CoT (Kojima et al., 2022)

```
MAGIC PHRASE: "Let's think step by step."

Q: "If there are 3 cars and each has 4 wheels, how many wheels total?"
A: "Let's think step by step. There are 3 cars. Each car has 4 wheels.
    Total wheels = 3 × 4 = 12. The answer is 12."

WHY IT WORKS: "Let's think step by step" activates a reasoning "mode"
              in the model's learned behavior — it learned this pattern
              from training data where problem-solving explanations followed this phrase

PERFORMANCE UPLIFT: +20-30% on math benchmarks vs direct answering
```

#### Few-Shot CoT (Wei et al., 2022)

```
Provide 4-8 examples of (question, step-by-step reasoning, answer) in the prompt
Model learns the STYLE of reasoning to apply to new questions

TRADEOFF: Uses precious context window for examples
          vs zero-shot CoT which uses just one phrase

WHEN TO USE WHICH:
  Zero-shot CoT: General reasoning, out-of-domain problems, limited context
  Few-shot CoT:  Specific domain (math, code, logic), more consistent format needed
```

#### Self-Consistency CoT (Wang et al., 2023)

```
IDEA: Single CoT path can make reasoning errors → sample MULTIPLE paths, majority vote

ALGORITHM:
  1. Sample K reasoning chains: CoT_1, CoT_2, ..., CoT_K  (temperature > 0)
  2. Each chain produces a final answer: a_1, a_2, ..., a_K
  3. Return the majority answer: argmax_a count(a_i == a)

INTUITION: Different reasoning paths that reach the same answer
           are more likely to be correct than any single path
           Errors in reasoning tend to be path-specific;
           correct answers are more "attractors" in the answer space

PERFORMANCE: +5-15% on top of standard CoT on math benchmarks
COST: K× more tokens generated (typically K=10-40)
```

---

## 🌲 TOPIC 11: TREE-OF-THOUGHT (ToT) — SYSTEMATIC SEARCH

> Yao et al., 2023 — "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"

### 11.1 Motivation: CoT is Linear, Problems are Non-Linear

```
CHAIN-OF-THOUGHT LIMITATION:
  CoT generates ONE linear sequence of thoughts:
  Thought_1 → Thought_2 → Thought_3 → Answer

  If Thought_2 is wrong:
  ├── No backtracking → error propagates
  └── Different valid paths from Thought_1 never explored

  Real problem-solving: explore multiple approaches, backtrack from dead ends

TREE-OF-THOUGHT:
  At each step: generate MULTIPLE candidate thoughts (branching)
  Evaluate each thought (is this a good direction?)
  Search the tree: BFS or DFS or MCTS over the thought space
  Backtrack when stuck → pursue better branches
```

### 11.2 ToT Architecture

```
COMPONENTS:
┌─────────────────────────────────────────────────────────────────┐
│ 1. THOUGHT GENERATOR                                            │
│    Given current state S_t → generate B candidate thoughts     │
│    Thought = intermediate reasoning step, partial solution      │
│    Methods: "sample" (temperature) or "propose" (few-shot)      │
│                                                                 │
│ 2. STATE EVALUATOR                                              │
│    Given state S_t → score quality of this thought path         │
│    Methods:                                                     │
│    ├── Value function: LLM scores each state 1-10              │
│    │   Prompt: "Rate this partial solution on a 1-10 scale..."  │
│    └── Vote function: Generate K answers, count majority votes  │
│                                                                 │
│ 3. SEARCH ALGORITHM                                             │
│    BFS: Explore all thoughts at depth d before depth d+1       │
│    DFS: Pursue one branch to completion; backtrack if stuck     │
│    MCTS: Balance exploration vs exploitation (see below)        │
└─────────────────────────────────────────────────────────────────┘

EXAMPLE (Game of 24 — make 24 from 4 numbers):
  Input: [4, 9, 10, 13]
  Thought 1a: 13 - 9 = 4  → remaining: {4, 4, 10}
  Thought 1b: 10 - 4 = 6  → remaining: {6, 9, 13}
  Thought 1c: 13 - 10 = 3 → remaining: {3, 4, 9}

  Evaluate: which path is more promising for making 24?
  Thought 1a → {4,4,10}: 4×4+10=26 close, 4×(4+10)=56 too big... evaluate: 7/10
  Thought 1c → {3,4,9}:  3×4+9=21, 9-3=6×4=24! YES → evaluate: 9/10

  Pursue Thought 1c:
  Thought 2a: 9 - 3 = 6 → remaining: {4, 6}
  Thought 2b: 4 × 3 = 12 → remaining: {9, 12}
  ...

GPT-4 + ToT: 74% success on Game of 24 vs 4% with standard prompting
```

### 11.3 Monte Carlo Tree Search (MCTS) for LLM Reasoning

```
MCTS adapted for LLM reasoning (used in AlphaCode 2, some reasoning models):

STATE: Current partial reasoning chain + problem
ACTION: Generate next thought/step
REWARD: 1 if reasoning leads to correct answer, 0 otherwise

MCTS LOOP:
┌────────── SELECTION ──────────────────────────────────────────┐
│ From root, traverse tree using UCB1 formula:                  │
│   UCB1(s) = Q(s)/N(s) + c × √(ln N(parent) / N(s))          │
│   Q(s) = total reward from state s, N(s) = visit count        │
│   c = exploration constant (tradeoff exploit vs explore)      │
│ Select leaf with highest UCB1                                  │
└───────────────────────────────────────────────────────────────┘
     ↓
┌────────── EXPANSION ──────────────────────────────────────────┐
│ At selected leaf: generate K new thoughts (actions)           │
│ Add them as children of the leaf node                         │
└───────────────────────────────────────────────────────────────┘
     ↓
┌────────── SIMULATION (Rollout) ──────────────────────────────┐
│ From expanded node: quickly simulate to end                   │
│ Use greedy/low-temp LLM sampling to reach terminal state      │
│ Check if final answer is correct → reward = 1 or 0            │
└───────────────────────────────────────────────────────────────┘
     ↓
┌────────── BACKPROPAGATION ───────────────────────────────────┐
│ Update Q(s) and N(s) for ALL nodes on the path to root       │
│ Winning paths get higher Q → visited more in future SELECTION │
└───────────────────────────────────────────────────────────────┘

REPEAT for T iterations → return highest-Q path as final answer

ADVANTAGE over simple ToT:
  Naturally balances exploration (trying new branches) vs
  exploitation (following known-good branches)
  Asymptotically optimal search strategy
```

---

## 🎯 TOPIC 12: PROCESS REWARD MODELS (PRMs) — STEP-BY-STEP VERIFICATION

> Lightman et al., 2023 (OpenAI) — "Let's Verify Step by Step"

### 12.1 Outcome vs Process Reward Models

```
OUTCOME REWARD MODEL (ORM):
  Input:  Problem + complete solution
  Output: Correct / Incorrect (binary)
  Used in: RLHF for final answer quality

PROBLEM with ORM:
  "The answer is 42" → ORM judges 42 correct or wrong
  But the REASONING PATH that led to 42 might be flawed
  Model could get right answer for wrong reasons → doesn't generalize

PROCESS REWARD MODEL (PRM):
  Input:  Problem + EACH STEP of reasoning (judged one at a time)
  Output: Correct / Incorrect for EACH INTERMEDIATE STEP
  Trains the model to produce correct reasoning, not just correct answers

  Q: "What is 17 × 23?"
  Step 1: "17 × 23 = 17 × (20 + 3)"          → PRM: CORRECT ✓
  Step 2: "= 17 × 20 + 17 × 3"               → PRM: CORRECT ✓
  Step 3: "= 340 + 61"                        → PRM: WRONG ✗ (17×3=51, not 61)
  Step 4: "= 401"                             → PRM: WRONG ✗ (consequence of step 3)

  With ORM: Final answer 401 → WRONG (model penalized, but unclear which step failed)
  With PRM: Step 3 → WRONG (model knows EXACTLY where the error occurred)
```

### 12.2 Training a PRM

```
DATA COLLECTION (the hard part):
  Need human labels on EACH step of EACH solution
  OpenAI PRM800K dataset: 800K step-level labels from human annotators
  Process: Show annotator partial solution → "Is this step correct? Yes/No/Neutral"

MODEL ARCHITECTURE:
  Base: Fine-tuned LLM (same architecture as policy model)
  Input: Problem + reasoning steps up to step k
  Output: P(step_k is correct | problem, steps_1..k)

  Token-level prediction: Place step separator tokens [STEP] between steps
  Predict correctness at each [STEP] token position
  Final hidden state at [STEP] token → linear layer → binary classification

TRAINING:
  L = -Σ_k [y_k log P(correct_k) + (1-y_k) log(1-P(correct_k))]
  y_k = 1 if step k is correct, 0 otherwise
  Train with BCE loss over all steps of all solutions
```

### 12.3 Using PRMs at Inference — Best-of-N with PRM

```
BEST-OF-N DECODING WITH PRM:
  1. Generate N candidate complete solutions (temperature > 0)
  2. Score each solution using PRM:
     score(solution) = product of step-level correctness probabilities
                     = Π_k P(step_k correct)
     OR: min score across all steps (most conservative)
  3. Return the solution with the highest PRM score

WHY BETTER THAN MAJORITY VOTE:
  Majority vote: count how many solutions reach same final answer
  PRM: evaluate the QUALITY of reasoning in each solution
  PRM catches: correct-answer-wrong-reasoning vs correct-answer-correct-reasoning
  → More reliable, especially for complex multi-step proofs

PERFORMANCE (Math benchmarks):
  GPT-4 greedy: ~42% on MATH dataset
  GPT-4, Best-of-100 with ORM: ~60%
  GPT-4, Best-of-100 with PRM: ~68%
  → PRM verification is significantly more effective than outcome-based selection
```

---

## 🔬 TOPIC 13: OpenAI o1/o3 AND DeepSeek-R1 — TRAINED REASONING

### 13.1 The o1 Paradigm — "Think Before You Answer"

> OpenAI, September 2024 — o1-preview; January 2025 — o1 full; 2025 — o3

**Core Insight:** Don't generate long CoT at inference time through clever prompting — **train the model to reason** during a private "thinking" phase before producing the final answer.

```
o1 INFERENCE FLOW:
  User:  "Prove that √2 is irrational"
           ↓
  [THINKING PHASE — invisible to user, charged as input tokens]:
  <think>
  Let me assume for contradiction that √2 = p/q where p,q are coprime integers.
  Then 2 = p²/q², so p² = 2q².
  This means p² is even, so p must be even (since odd² is odd).
  Let p = 2m. Then (2m)² = 2q², → 4m² = 2q² → q² = 2m².
  This means q² is even, so q is also even.
  But if both p and q are even, they share factor 2 — contradiction with coprimality.
  Therefore, √2 cannot be expressed as p/q → irrational. ✓
  </think>
           ↓
  [FINAL ANSWER — visible to user]:
  "Proof by contradiction: Assume √2 = p/q in lowest terms (gcd(p,q)=1).
   Then p² = 2q², implying p is even (p=2m). Substituting: q² = 2m², so q is even.
   But this contradicts gcd(p,q)=1. Therefore √2 is irrational. ∎"

WHAT'S IN THE THINKING PHASE:
  - Draft reasoning steps (often exploratory, not polished)
  - Backtracking: "No wait, that's wrong... let me reconsider..."
  - Multiple approaches tried and abandoned
  - Self-verification: "Does this answer check out? Let me verify..."
  - The model is "learning" during inference via the extended context
```

### 13.2 How o1 Was Trained

```
o1's training is NOT published in full — these are inferred from context + partial disclosures:

STEP 1: COLD START DATA
  Collect human expert demonstrations of step-by-step long reasoning
  Format: <think>...</think><answer>...</answer>
  Domains: Math, code, logic, science problems with verified answers

STEP 2: REINFORCEMENT LEARNING ON REASONING CHAINS
  Policy: LLM generating reasoning chains
  Reward: Outcome-based (is the final answer correct?)
  Optional: PRM reward for intermediate steps

  Train with RL (PPO or REINFORCE):
  → Model learns to generate reasoning chains that lead to correct answers
  → No need to label WHICH reasoning steps are good — just whether answer is right
  → Model discovers effective reasoning strategies through trial and error

STEP 3: SCALING TEST-TIME COMPUTE
  Key insight: More thinking tokens → better answers (compute-scaling at inference)
  Train model to USE MORE compute when problems are harder
  o1 uses "adaptive thinking budget" — harder problems get more thinking tokens

VERIFIED FACT (from OpenAI):
  o1 performance scales with:
  1. Training compute (standard)
  2. TEST-TIME compute (new!) — more thinking = better answer, up to a point
  This is the key claim: spending more compute at inference systematically improves reasoning
```

### 13.3 DeepSeek-R1 — Open-Source Reasoning (January 2025)

> DeepSeek-AI — "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"

```
KEY CONTRIBUTION: Full training recipe published (unlike o1)

TRAINING STAGES:

STAGE 0: Cold Start
  Fine-tune base model on small set (~thousands) of long CoT examples
  Establishes basic format: <think>...</think><answer>...</answer>
  Without this: RL often produces unintelligible reasoning

STAGE 1: GRPO (Group Relative Policy Optimization) RL Training
  For each problem:
  1. Sample G=8 responses from current policy
  2. Score each response: r_i = 1 if answer correct, 0 if wrong
  3. Baseline: mean reward of the group: r̄ = (1/G) Σ r_i
  4. Policy gradient update:
     L = -(1/G) Σ_i [min(r_i - r̄) × ratio, clip(ratio, 1-ε, 1+ε) × (r_i - r̄)]
     where ratio = π_θ(response_i) / π_old(response_i)

  GRPO vs PPO:
  PPO: needs separate value function (critic model)
  GRPO: uses group average as baseline → no critic needed → simpler + cheaper

STAGE 2: Rejection Sampling + SFT
  Run current model on many problems
  Keep only solutions that are: (a) correct AND (b) have readable reasoning
  Fine-tune on this high-quality filtered dataset
  → Makes reasoning more readable/coherent while maintaining correctness

STAGE 3: RL for All Reward Types (helpfulness, harmlessness + reasoning)
  Multi-objective RL:
  ├── Reasoning reward: PRM step-level correctness
  ├── Format reward: Is the output format correct? (<think> tag present?)
  └── Helpfulness reward: Human preference model (harmless, helpful)

EMERGENT BEHAVIOR (discovered during training, not engineered):
  The model spontaneously learned:
  ├── Self-verification: "Wait, let me check my work..."
  ├── Backtracking: "That approach doesn't work. Let me try..."
  ├── Reflection: "I made an error in step 3. The correct calculation is..."
  └── Exploration: "There are two approaches. Let me try approach 1 first..."

  THESE WERE NOT IN THE TRAINING DATA — RL discovered them as reward-maximizing strategies!
```

### 13.4 Reasoning Scaling Laws — Test-Time Compute

```
TRADITIONAL SCALING (Kaplan 2020):
  Performance improves with: training compute C, model size N, data size D
  But: fixed cost after training

TEST-TIME COMPUTE SCALING (Snell et al., 2024; OpenAI o1):
  Performance ALSO improves with compute spent at INFERENCE TIME
  More thinking tokens → better accuracy on hard problems

THE COMPUTE-OPTIMAL QUESTION:
  Given a fixed inference compute budget B, what's the best strategy?
  Option A: One large model, fast answer
  Option B: One smaller model, much more thinking tokens

  Finding: For math/reasoning tasks, Option B often wins
  A 3B model with 256 thinking tokens can match an 8B model with 32 thinking tokens
  "Small model + more thinking" is often more efficient than "bigger model, less thinking"

SCALING LAW FOR TEST-TIME COMPUTE:
  accuracy ≈ a × log(thinking_tokens) + b   (approximate log-linear relationship)
  Returns diminish but remain positive over 4 orders of magnitude of thinking budget
```

### 13.5 Reasoning Failure Modes and Mitigations

```
FAILURE MODE 1: OVERTHINKING (o1 sometimes reasons too long on easy problems)
  Model uses 1000 thinking tokens for "What is 2+2?"
  Mitigation: Adaptive compute — train special token "thinking budget" signal
              Easy problems → short think; hard problems → long think

FAILURE MODE 2: REASONING COLLAPSE
  Under reward pressure, model "games" the reward:
  Generate plausible-looking reasoning that isn't actually valid
  Final answer is correct (reward = 1) but reasoning is post-hoc rationalization

  Detection: PRM catches this — invalid intermediate steps scored low
  Mitigation: Dense PRM rewards; adversarial verification

FAILURE MODE 3: SYCOPHANTIC REASONING
  User hints at wrong answer → model's reasoning steers toward that answer
  "I think the answer might be 42... let me reason..."
  → Reasoning confirms 42 even if wrong

  Mitigation: Train on examples where user hints are wrong;
              reward for disagreeing with user when correct

FAILURE MODE 4: FORMAT DRIFT IN LONG CHAINS
  Very long thinking chains → model loses track of original problem
  Reasoning becomes tangential or circular

  Mitigation: Hierarchical reasoning (summarize and re-anchor periodically);
              Attention sink mechanism for initial problem tokens

FAILURE MODE 5: HALLUCINATED REASONING
  Model generates confident-sounding intermediate reasoning steps
  that are factually wrong, leading to wrong conclusion
  Example: "Einstein won the Nobel Prize for Relativity in 1925"
           → Uses this wrong fact as premise → wrong conclusion

  Mitigation: Ground reasoning in retrieved facts (RAG before reasoning);
              PRM trained to catch factual errors in steps
```

---

## 🔄 TOPIC 14: REACT — REASONING + ACTING (Agentic Reasoning)

> Yao et al., 2022 — "ReAct: Synergizing Reasoning and Acting in Language Models"
> The paper behind most production agentic AI systems including your own work at Chubb

### 14.1 The Core Pattern

```
PROBLEM: Pure CoT reasons but doesn't ACT (can't look things up, run code, check databases)
          Pure action selection (tool-calling) acts but doesn't reason about WHY

REACT: Interleave THOUGHT (reasoning) and ACTION (tool use) in a loop

FORMAT:
  Thought: [Reasoning about what to do next, why, and what we expect]
  Action:  [Tool call with parameters]
  Observation: [Result returned by the tool]
  Thought: [Reasoning about what the observation means and what to do next]
  Action:  [Next tool call]
  ...
  Final Answer: [Synthesis of all observations into a response]

EXAMPLE (Flipkart Fraud Investigation Agent):
  Thought: This claim involves a surgery 5 days before the injury date. I need to
           check if this pattern appears in known fraud cases.
  Action: search_fraud_database(query="surgery before injury document order anomaly")
  Observation: Found 3 similar cases: Case #4521, #3890, #5102 — all confirmed fraud.

  Thought: The pattern matches confirmed fraud. Now I should check the attorney's track record.
  Action: lookup_attorney(attorney_id="ATT-987321")
  Observation: Attorney filed 18 claims in 2024; 14 were disputed; 9 confirmed fraud.

  Thought: High-risk attorney with 50% confirmed fraud rate. Combined with the
           temporal anomaly, this claim has very high indicators.
  Action: generate_risk_report(case_id="CLM-2024-9871", risk_level="HIGH",
           evidence=["procedure_before_injury", "high_risk_attorney"])
  Observation: Report generated successfully.

  Final Answer: Case CLM-2024-9871 assessed HIGH RISK.
               Key indicators: (1) Surgery date precedes injury date by 5 days,
               (2) Attorney ATT-987321 has 50% confirmed fraud rate across 18 claims.
               Escalate to senior investigator.
```

### 14.2 Why ReAct Works Better Than Pure CoT or Pure Tool-Use

```
PURE COT:
  Can reason about what tools to call but can't actually call them
  Hallucinating the tool output → building reasoning on false premises
  "I'll check the database... it probably says X" → X is fabricated

PURE TOOL-USE (No reasoning):
  Call tool → get result → call next tool → get result → answer
  No reasoning about WHY to call which tool
  Brittle: fails on novel situations not in the action template
  Can't handle unexpected tool outputs gracefully

REACT (Combined):
  Thought before action: reasons about WHY this tool call is appropriate
  Observation after action: updates understanding based on actual data
  Thought before next action: adapts plan based on what was learned
  → Self-correcting: if tool returns unexpected result, thought step reacts
  → Generalizes: novel situations handled through reasoning, not templates

ABLATION RESULTS (from ReAct paper):
  Task: HotpotQA (multi-hop QA)
  Pure CoT: 29.4% success
  Pure Action: 25.7% success
  ReAct: 35.1% success  (+20% over pure CoT)
```

### 14.3 Advanced ReAct: Plan-and-Execute vs Pure ReAct

```
PURE REACT (Interleaved Thought-Action):
  ✓ Reactive to observations — can adapt mid-execution
  ✗ Short planning horizon — next action based only on recent observations
  ✗ Prone to losing track of global goal in long chains

PLAN-AND-EXECUTE (HuggingGPT, LangChain Agents):
  Step 1: PLAN — Generate complete task plan upfront
    "To answer this: (1) Search fraud DB, (2) Check attorney, (3) Generate report"
  Step 2: EXECUTE — Execute each plan step with ReAct loop
  Step 3: REPLAN (if execute step fails) — update remaining plan based on observations

  ✓ Clear global structure — model holds complete task context
  ✓ Better for predictable, multi-step workflows
  ✗ Less adaptive if early steps produce unexpected results

HYBRID (recommended for production):
  Plan at task level → ReAct at subtask level
  Strategic planning + tactical flexibility
```

---

## 📊 REASONING SUMMARY TABLE

```
┌────────────────────────────┬──────────────────┬─────────────────┬────────────────────┐
│ Technique                  │ Key Idea          │ Best For        │ Cost               │
├────────────────────────────┼──────────────────┼─────────────────┼────────────────────┤
│ Standard LLM (System 1)    │ Direct answer     │ Simple Q&A      │ 1× tokens          │
│ Zero-Shot CoT              │ "Think step by    │ General reason  │ 2-3× tokens        │
│                            │  step"            │                 │                    │
│ Few-Shot CoT               │ Examples of       │ Domain-specific │ 3-5× tokens        │
│                            │ step-by-step      │ reasoning       │                    │
│ Self-Consistency           │ Majority vote     │ Math, logic     │ 10-40× tokens      │
│                            │ across K chains   │                 │                    │
│ Tree-of-Thought            │ Branch + eval     │ Search, games,  │ 10-100× tokens     │
│                            │ + backtrack       │ planning        │                    │
│ Best-of-N + PRM            │ PRM scores each   │ Math proofs     │ N× tokens          │
│                            │ step              │                 │                    │
│ o1 / DeepSeek-R1          │ RL-trained        │ Hard math,      │ 10-1000× tokens    │
│ (Reasoning Models)         │ long thinking     │ code, science   │ (adaptive budget)  │
│ ReAct                      │ Thought+Action    │ Agentic tasks   │ Variable           │
│                            │ interleaved       │ tool use        │                    │
│ MCTS + LLM                 │ UCB1 search       │ Novel search    │ Very high          │
│                            │ over thought tree │ problems        │                    │
└────────────────────────────┴──────────────────┴─────────────────┴────────────────────┘
```

---

## 🎯 INTERVIEW QUESTIONS ON REASONING

### Q: "How does o1 differ from GPT-4 with chain-of-thought prompting?"

> "GPT-4 with CoT prompting adds reasoning at inference time via the prompt — it's System 1 with a forced linear trace. o1 has *trained* reasoning: through RL, it learned to generate long exploratory 'thinking' before answering, including backtracking, self-correction, and trying multiple approaches — behaviors that emerged spontaneously as reward-maximizing strategies. Crucially, o1 scales with *test-time compute* — spending more thinking tokens systematically improves accuracy on hard problems. The thinking is also private (not shown to the user), trained to be exploratory. GPT-4+CoT is a prompt trick; o1's reasoning is a learned behavior."

### Q: "How would you use reasoning models in the Flipkart fraud system?"

> "I'd deploy a two-tier approach. For real-time transaction screening (< 100ms budget), I use a fast classifier — no reasoning, just pattern matching. For complex investigation cases flagged by the classifier, I'd deploy a reasoning-capable agent using the ReAct pattern: Thought → Action (lookup case history, check attorney, query knowledge graph) → Observation → next Thought. The reasoning trace becomes the audit log — every decision backed by evidence citations. For the most ambiguous cases, I'd use Best-of-N sampling with a PRM trained on investigator labels, selecting the reasoning chain that is most step-by-step correct rather than just the most confident final answer. This prevents the system from reaching correct fraud verdicts through flawed reasoning — which matters for regulatory compliance and court defensibility."

---

*End of LLM Inference Optimization + Reasoning Deep Dive*

