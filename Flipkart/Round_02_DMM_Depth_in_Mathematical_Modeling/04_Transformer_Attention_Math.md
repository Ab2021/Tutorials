# 🤖 Transformer & Attention Math — DMM Round Deep Dive
### PhD-Level Questions You'll Face in the Alex Rivera / Research Director Interview

---

## SECTION 1: FULL TRANSFORMER ARCHITECTURE

```
INPUT TEXT: "The claim was filed on Jan 15"
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                EMBEDDING LAYER                      │
│  Token Embedding + Positional Encoding              │
│  x_i = E_token(token_i) + E_pos(position_i)        │
│  Output: sequence of d_model-dimensional vectors   │
└─────────────────────────────────────────────────────┘
       │
       ▼  (× N transformer blocks)
┌─────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK                      │
│                                                     │
│  1. MULTI-HEAD SELF-ATTENTION                       │
│     Q = XW_Q, K = XW_K, V = XW_V                  │
│     Attention = softmax(QK^T/√d_k)V                │
│     Output + residual connection → LayerNorm        │
│                                                     │
│  2. FEED-FORWARD NETWORK                            │
│     FFN(x) = max(0, xW₁ + b₁)W₂ + b₂              │
│     Two linear layers with ReLU/GELU activation     │
│     Output + residual connection → LayerNorm        │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              TASK-SPECIFIC HEAD                     │
│  BERT: [CLS] token → classification head           │
│  GPT: Last token → next token prediction head      │
│  T5: Encoder-decoder → sequence-to-sequence        │
└─────────────────────────────────────────────────────┘
```

---

## SECTION 2: ATTENTION MECHANISM — EVERY DETAIL

### Why Scaled Dot-Product?

For query $q \in \mathbb{R}^{d_k}$ and key $k \in \mathbb{R}^{d_k}$:
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

If $q_i, k_i \sim \mathcal{N}(0,1)$, then $q \cdot k \sim \mathcal{N}(0, d_k)$ (variance grows linearly with $d_k$).

**Problem:** Large $d_k$ → large dot products → softmax saturates (outputs near-one-hot) → near-zero gradients for non-maximum positions → training slows down severely.

**Fix:** Scale by $\frac{1}{\sqrt{d_k}}$ → variance of scaled dot product = 1 → softmax stays in gradient-friendly regime.

### Multi-Head Attention — WHY Multiple Heads?

Single-head attention computes one weighted average of values. With multiple heads:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{MultiHead} = \text{Concat}(\text{head}_1, ..., \text{head}_H)W^O$$

Each head uses **different learned projection matrices** $(W_i^Q, W_i^K, W_i^V)$.

**Empirical discovery:** Different heads learn different types of relationships:
- Head 1: Direct syntactic dependencies (subject-verb)
- Head 2: Coreference resolution ("it" → "the claim")
- Head 3: Long-range semantic dependencies
- Head 4: Position-based patterns (adjacent words)

**Computation:** If $d_{model} = 768$, $H = 12$ heads: each head has $d_k = d_v = 64$. Total parameters: similar to single-head with $d_k = 768$ but much richer representational capacity.

---

## SECTION 3: POSITIONAL ENCODING

### Sinusoidal Encoding (Original Transformer / BERT):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Why sinusoidal?**
1. **Unique encoding:** Each position gets a unique vector
2. **Generalization:** Model can potentially generalize to sequence lengths longer than seen in training (extrapolation)
3. **Relative positions:** $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$ — encoder can learn to attend by relative position

**Alternative: Learnable positional encoding (BERT uses this)**
Simply learn a lookup table: $PE_i \in \mathbb{R}^{d_{model}}$ for each position up to max_position (typically 512 for BERT).
- Pro: More flexible, can learn task-optimal encoding
- Con: Cannot generalize beyond max_position seen during training

**Modern alternative: RoPE (Rotary Position Embedding) — used in LLaMA, GPT-NeoX**
Encodes absolute positions using rotation matrices; enables relative position information in attention. Better extrapolation than learnable.

---

## SECTION 4: BERT vs. GPT — MATHEMATICAL DISTINCTION

| | BERT | GPT |
|---|---|---|
| Architecture | Encoder only | Decoder only |
| Attention mask | Full bidirectional | Causal (lower triangular) |
| Pre-training | MLM + NSP | Next token prediction (LM) |
| Training objective | $-\log P(w_{masked} | w_{context})$ | $-\log P(w_t | w_{<t})$ |
| Best use | Classification, NER, QA | Generation, completion |
| BERT [CLS] | Class representation of full sequence | Not used for classification |

**The causal mask in GPT:**
$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

Apply before softmax: $\exp(-\infty) = 0$ → future positions get zero attention weight → unidirectional: only past context influences current token.

---

## SECTION 5: FINE-TUNING STRATEGIES

### Full Fine-Tuning vs. Parameter-Efficient Methods

**Full Fine-Tuning:**
- Update all $N$ parameters (BERT-base: 110M, GPT-3: 175B)
- Pros: Maximum adaptation capacity
- Cons: Expensive, catastrophic forgetting risk, requires all data in memory

**LoRA (Low-Rank Adaptation) — Must Know:**
Instead of updating $W \in \mathbb{R}^{d \times d}$, learn a low-rank decomposition:
$$W_{new} = W_{frozen} + \Delta W = W_{frozen} + BA$$

Where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, rank $r \ll d$.

For BERT-base with $d=768$: Full fine-tuning updates $768 \times 768 = 590K$ parameters per matrix.
LoRA with $r=8$: Updates $768 \times 8 + 8 \times 768 = 12,288$ parameters — **48× fewer**.

**In your fraud system:** If re-training an embedding model on new fraud patterns:
- LoRA on the transformer layers: fine-tune $r=8$ rank matrices on fraud domain data
- Preserve general language understanding (frozen base weights) while adapting to fraud-specific patterns
- Prevents catastrophic forgetting of general NLP capability

**Prompt Tuning / Prefix Tuning:**
Learn soft prompt tokens; freeze all model weights. Only ≈1K parameters updated.
Best for: Very limited data, strong base model, task closely related to pre-training distribution.

---

## SECTION 6: KV CACHE AND INFERENCE OPTIMIZATION

**The Problem:** Autoregressive generation requires re-computing K and V matrices for all previous tokens at each new generation step. For a 2048-token sequence: $O(n^2)$ compute per token.

**KV Cache:**
Store Key and Value matrices for all previously generated tokens. At each new step, only compute for the new token, then append to cache.

$$\text{Cache at step } t: \{(K_1, V_1), ..., (K_t, V_t)\}$$

New step: compute only $(K_{t+1}, V_{t+1})$, concatenate with cache.

**Memory cost:** KV cache size = $2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_length \times \text{dtype\_size}$ bytes.

For GPT-4 (estimated): 96 layers, 128 heads, d_head=128, fp16: ~3GB per 1000 tokens.

**At scale (production serving):** This is why vLLM uses **paged attention** — divides KV cache into fixed-size pages, manages memory like OS virtual memory. Enables much higher throughput by sharing KV cache across requests that share common prefixes.

---

## SECTION 7: RAG MATH — WHAT YOU BUILT

### Retrieval Quality: The Math Behind FAISS

FAISS uses **HNSW (Hierarchical Navigable Small World)** for Approximate Nearest Neighbor (ANN) search.

Graph construction:
- Insert each vector as a node
- Connect to $M$ nearest neighbors in each of $L$ layers
- Higher layers: fewer nodes, longer connections (for fast rough navigation)
- Lower layers: all nodes, shorter connections (for precise local search)

Search: Start at top layer, greedily descend to nearest neighbor at each layer.
Time complexity: $O(\log N)$ vs. exact search $O(N \cdot d)$ — massive speedup at scale.

**Recall vs. Speed tradeoff:**
- Parameter `ef_search` controls how many candidates are explored at each step
- Higher `ef_search` → better recall, slower search
- Typical: 95% recall at 2ms latency for 1M vectors

### Why Not Exact Search In Production?

At 1M fraud patterns with 768-dim embeddings:
- Exact search: 1M × 768 float32 = 3GB to scan × n_queries per second = infeasible
- FAISS HNSW: sub-linear search, ~2ms per query, 95% recall — acceptable for fraud use case

---

## SECTION 8: UNUSUAL DEPTH QUESTIONS (PhD-Level)

**Q: Why does layer normalization perform better than batch normalization in transformers?**

Batch Norm normalizes across the batch dimension: $\hat{x} = \frac{x - \mu_{batch}}{\sigma_{batch}}$
- Problem 1: Performance degrades with small batch sizes (NLP often needs seq-level batches — effective batch size can be small)
- Problem 2: For sequences of different lengths — batch statistics are polluted by padding tokens
- Problem 3: At inference with batch_size=1, batch statistics are unstable

Layer Norm normalizes across the feature dimension for each sample independently: $\hat{x}_i = \frac{x_i - \mu_{features_i}}{\sigma_{features_i}}$
- Independent of other samples in batch → works perfectly with batch_size=1
- Works with variable-length sequences (each position normalized independently)
- Pre-norm (before attention) or post-norm (after) — modern transformers often use pre-norm for training stability

---

**Q: What is catastrophic forgetting and how did you prevent it in your domain fine-tuning?**

Catastrophic forgetting: When fine-tuning a pre-trained model on domain-specific data, the model forgets general language knowledge encoded in pre-training, as new gradients overwrite useful representations.

Signs: Model fine-tuned on claims data can no longer correctly parse general English sentences; general benchmark performance drops.

Prevention strategies:
1. **Low learning rate:** 1e-5 to 5e-5 (very small compared to pre-training: 1e-4 to 3e-4)
2. **Layer-wise learning rate decay (LLRD):** Lower layers get smaller LR (they contain general syntax); higher layers get larger LR (they contain task-specific semantics)
3. **Elastic Weight Consolidation (EWC):** Add regularization that penalizes changes to parameters that were important for original pre-training tasks
4. **LoRA:** Only update low-rank delta matrices; frozen base weights retain all pre-training knowledge

In your fraud embedding model: Used LoRA + low LR (2e-5) on the contrastive fine-tuning step. Validated by checking embedding quality on a general NLP benchmark didn't degrade.

---

*See companion files: 07_Deep_Dive_RNN_LSTM_Attention_Transformers_BERT_GPT.md (in root), 08_LLM_Inference_Optimization_Deep_Dive.md*
