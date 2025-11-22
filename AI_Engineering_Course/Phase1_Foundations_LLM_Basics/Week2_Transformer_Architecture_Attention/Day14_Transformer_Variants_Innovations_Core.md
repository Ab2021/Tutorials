# Day 14: Transformer Variants & Innovations
## Core Concepts & Theory

### The Quest for Efficiency

The standard Transformer has a fundamental bottleneck: **Quadratic Complexity**.
- Attention mechanism: $O(N^2)$ time and memory.
- For $N=1024$, $N^2 \approx 1M$.
- For $N=100,000$, $N^2 \approx 10B$.
This makes processing very long documents (books, codebases, DNA) impossible with standard attention.

### 1. Sparse Attention (Longformer, BigBird)

**Core Idea:** Not every token needs to attend to every other token.
- **Local Attention:** Tokens attend only to their neighbors (sliding window). $O(N \cdot W)$.
- **Global Attention:** A few special tokens (e.g., `[CLS]`) attend to everything. $O(N)$.
- **Random Attention:** (BigBird) Add random connections to approximate full attention.

**Result:** Linear complexity $O(N)$.
**Trade-off:** Can miss complex long-range dependencies if not carefully tuned.

### 2. Linear Attention (Linformer, Performer)

**Core Idea:** Approximate the Softmax attention matrix using kernel methods.
Standard: $Attention(Q, K, V) = softmax(QK^T)V$
Linear: Decompose $softmax(QK^T)$ into $\phi(Q)\phi(K)^T$.
Then: $(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^TV)$.
- $\phi(K)^TV$ is $d \times d$.
- Multiplication is $N \times d^2$.
- **Complexity:** $O(N)$.

**Trade-off:** Approximation error. Often performs worse than standard attention on "hard" tasks requiring precise retrieval.

### 3. Flash Attention (The Modern Standard)

**Core Idea:** It's not about changing the math; it's about **IO-Awareness**.
GPUs have fast compute (H100: 1000 TFLOPS) but slow memory (HBM: 3 TB/s).
Standard attention reads/writes large $N \times N$ matrices to HBM.
**Flash Attention:**
- Tiling: Compute attention block-by-block in fast SRAM (L1 cache).
- Recomputation: Don't store the huge attention matrix for backward pass; recompute it on the fly.
- **Result:** Exact attention (no approximation) but 2-4x faster and linear memory $O(N)$ (for activation storage).

### 4. Mixture of Experts (MoE)

**Core Idea:** Scale model size (parameters) without scaling compute cost.
Replace the dense Feed-Forward Network (FFN) with a sparse layer of "Experts".
- **Router:** Decides which expert(s) process each token.
- **Experts:** $E$ parallel FFNs. Only top-k (usually k=2) are activated.

**Benefits:**
- Massive parameter count (e.g., GPT-4, Mixtral 8x7B).
- Fast inference (only use fraction of params).
- **Trade-off:** Training stability, VRAM usage (need to load all experts).

### 5. Grouped Query Attention (GQA)

**Core Idea:** Optimize inference speed and KV cache size.
- **MHA (Multi-Head):** $H$ query heads, $H$ key/value heads. High memory.
- **MQA (Multi-Query):** $H$ query heads, 1 key/value head. Fast, but lower quality.
- **GQA (Grouped-Query):** $H$ query heads, $G$ key/value heads (e.g., 8). Sweet spot.
Used in LLaMA-2/3, Mistral.

### Summary of Variants

| Variant | Complexity | Key Feature | Used In |
| :--- | :--- | :--- | :--- |
| **Standard** | $O(N^2)$ | Exact, robust | BERT, GPT-2 |
| **Longformer** | $O(N)$ | Sparse/Local | Long docs |
| **Linear** | $O(N)$ | Kernel Approx | Research |
| **Flash Attn** | $O(N^2)$* | IO-Optimized | **All Modern LLMs** |
| **MoE** | $O(1)$** | Sparse FFN | GPT-4, Mixtral |

*\*Flash Attn is mathematically $O(N^2)$ but practically linear memory for training due to not storing the matrix.*
*\*\*MoE is $O(1)$ relative to parameter count increase.*

### Next Steps
In the Deep Dive, we will explore the mechanics of Flash Attention and Mixture of Experts in detail.
