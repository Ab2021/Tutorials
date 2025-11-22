# Day 14: Transformer Variants & Innovations
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Flash Attention: IO-Awareness

**The Bottleneck:**
Standard Attention:
1.  Load $Q, K$ from HBM to SRAM.
2.  Compute $S = QK^T$ (Size $N \times N$). Write $S$ to HBM.
3.  Load $S$ from HBM. Compute $P = softmax(S)$. Write $P$ to HBM.
4.  Load $P, V$ from HBM. Compute $O = PV$. Write $O$ to HBM.

**Problem:** $N \times N$ matrix $S$ and $P$ are huge. Writing/Reading them to HBM is slow.

**Flash Attention Algorithm:**
Use **Tiling** (block-wise computation) and **Recomputation**.
1.  Load blocks $Q_i, K_j, V_j$ into SRAM.
2.  Compute block output $O_{ij}$ in SRAM.
3.  Update running statistics (max, sum) for Softmax normalization.
4.  Write only final output $O$ to HBM.

**Key Insight:**
We never materialize the full $N \times N$ matrix in HBM.
- **Memory:** $O(N)$ (linear).
- **Speed:** 2-4x faster (fewer HBM accesses).
- **Exact:** Mathematically identical to standard attention.

### 2. Mixture of Experts (MoE): Sparse Scaling

**Architecture:**
Replace dense FFN layer with:
$$ y = \sum_{i=1}^E G(x)_i \cdot E_i(x) $$
- $E_i(x)$: Output of Expert $i$ (a small FFN).
- $G(x)$: Gating network (Router). Returns a sparse vector (e.g., only 2 non-zeros).

**Routing Strategies:**
1.  **Top-k Gating:** $G(x) = softmax(W_g x)$. Keep top-k values, set rest to 0.
2.  **Load Balancing:** If everyone goes to Expert 1, it becomes a bottleneck.
    - **Auxiliary Loss:** Penalize if distribution of experts is unbalanced.
    - **Capacity Factor:** Drop tokens if an expert is over capacity.

**Switch Transformer (Google):** Top-1 routing.
**Mixtral 8x7B:** Top-2 routing. 8 experts total.
- Total Params: 47B.
- Active Params per token: 13B (2 experts).
- Result: Quality of 47B model, Inference speed of 13B model.

### 3. Grouped Query Attention (GQA)

**Memory Analysis of KV Cache:**
For batch size $B$, sequence length $L$, heads $H$, dim $D$:
- KV Cache Size: $2 \cdot B \cdot L \cdot H \cdot D$.
- For LLaMA-65B: $H=64, D=128$. 1k tokens $\approx$ 1GB per user.

**MQA (Multi-Query Attention):**
- Share 1 KV head across all Query heads.
- Memory reduction: $64x$.
- Quality: Significant degradation.

**GQA (Grouped-Query Attention):**
- Interpolation. Group $H$ query heads into $G$ groups.
- Share 1 KV head per group.
- Example: 64 Q-heads, 8 KV-heads (8 groups).
- Memory reduction: $8x$.
- Quality: Near-identical to full MHA.

### 4. Sliding Window Attention (Mistral)

**Mechanism:**
- Each token attends only to the last $W$ tokens (e.g., 4096).
- Theoretical Receptive Field: $L \times W$.
- **Why?** Layers stack. Layer 1 sees 4k. Layer 2 sees 4k of Layer 1's output (which saw 4k).
- Effective context grows linearly with depth.
- **Rolling Buffer Cache:** Fixed size KV cache. Oldest tokens overwritten. $O(1)$ memory inference.

### Summary of Mechanics

| Mechanism | Optimizes | Trade-off |
| :--- | :--- | :--- |
| **Flash Attention** | Speed & Memory | Implementation Complexity |
| **MoE** | Parameter Efficiency | VRAM (must load all experts) |
| **GQA** | Inference Memory | Slight Quality Drop |
| **Sliding Window** | Inference Speed | Long-range dependency (direct) |

### Code: Simple MoE Router

```python
class MoERouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: (batch, seq, d_model)
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize probs to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices
```
