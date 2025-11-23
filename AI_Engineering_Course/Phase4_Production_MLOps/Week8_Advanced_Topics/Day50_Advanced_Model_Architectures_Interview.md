# Day 50: Advanced Model Architectures
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is Mixture of Experts (MoE) and what are its benefits?

**Answer:**
- **Concept:** Multiple expert networks, router selects which experts to use per token.
- **Sparse Activation:** Only 10-20% of parameters active per token.
- **Benefits:**
  - **Scaling:** Can scale to trillions of parameters.
  - **Efficiency:** Lower inference cost than dense models.
  - **Specialization:** Each expert specializes in different patterns.
- **Examples:** Switch Transformer (1.6T params), Mixtral 8x7B.

#### Q2: How do long-context models handle sequences >100K tokens?

**Answer:**
**Sparse Attention:**
- **Longformer:** Local + global attention. O(N) complexity.
- **BigBird:** Random + window + global. O(N) complexity.

**Linear Attention:**
- **Performer:** Kernel-based approximation. O(N) complexity.

**Sliding Window:**
- **Mistral:** Attend to last 4K tokens only.

**Retrieval:**
- **Memorizing Transformers:** Retrieve from external memory.

**Trade-off:** Sparse/linear attention loses some quality vs full attention.

#### Q3: Explain Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).

**Answer:**
**MQA:**
- Q has num_heads, K/V shared across all heads (single head).
- **KV Cache:** 8x smaller (for 8 heads).
- **Speed:** 1.5-2x faster inference.
- **Accuracy:** Minimal loss (<1%).

**GQA:**
- Q has num_heads, K/V shared within groups.
- **Example:** 32 Q heads, 8 KV heads (4 groups).
- **KV Cache:** 4x smaller.
- **Accuracy:** Better than MQA, close to standard.

**When:** MQA for maximum speed, GQA for balance.

#### Q4: What is model merging and when should you use it?

**Answer:**
- **Concept:** Combine multiple fine-tuned models into one.
- **Methods:**
  - **Averaging:** `(model1 + model2) / 2`
  - **Task Arithmetic:** `base + α(model1 - base) + β(model2 - base)`
  - **Model Soup:** Average checkpoints from same training run.
- **Benefits:** Multi-task model without multi-task training.
- **When:** Have multiple task-specific models, want single model.

#### Q5: What are the challenges with MoE models in production?

**Answer:**
**Load Balancing:**
- Some experts get overused, others underused.
- **Solution:** Add load balancing loss to router.

**Memory:**
- All experts must fit in memory (even if not all active).
- **Solution:** Expert parallelism across GPUs.

**Routing Overhead:**
- Router adds latency.
- **Solution:** Optimize router network.

**Training Instability:**
- Router can collapse (always select same experts).
- **Solution:** Auxiliary losses, careful initialization.

---

### Production Challenges

#### Challenge 1: MoE Load Imbalance

**Scenario:** MoE model with 8 experts. Expert 0 gets 80% of tokens, others get 2-3% each.
**Root Cause:** Router learned to prefer one expert.
**Solution:**
- **Load Balancing Loss:** Penalize imbalanced routing.
  ```python
  load_balance_loss = coefficient * variance(expert_usage)
  ```
- **Auxiliary Loss:** Encourage uniform distribution.
- **Expert Dropout:** Randomly drop experts during training.

#### Challenge 2: Long-Context OOM

**Scenario:** Trying to process 100K token sequence. GPU runs out of memory.
**Root Cause:** O(N²) attention memory.
**Solution:**
- **Sparse Attention:** Use Longformer (O(N) memory).
- **Chunking:** Process in chunks of 4K tokens.
- **Gradient Checkpointing:** Recompute activations instead of storing.
- **Larger GPU:** Upgrade to A100 80GB or H100.

#### Challenge 3: Model Merging Quality Drop

**Scenario:** Merged math and code models. Quality dropped 20% on both tasks.
**Root Cause:** Negative interference between tasks.
**Solution:**
- **Tune Alphas:** Try different merge coefficients (α=0.3, β=0.7 instead of 0.5, 0.5).
- **Task Vectors:** Use task arithmetic instead of simple averaging.
- **Selective Merging:** Only merge certain layers (e.g., FFN, not attention).
- **Fine-tune After Merge:** Fine-tune merged model on both tasks.

#### Challenge 4: Sparse Attention Quality Loss

**Scenario:** Switched to Longformer for long context. Quality dropped 10%.
**Root Cause:** Sparse attention misses some long-range dependencies.
**Solution:**
- **Hybrid:** Use sparse for most layers, full attention for last few layers.
- **Global Attention:** Mark important tokens (e.g., [CLS]) for global attention.
- **Increase Window:** Use larger window size (1024 instead of 512).
- **Accept Trade-off:** 10% quality loss might be acceptable for 10x longer context.

#### Challenge 5: EWC Catastrophic Forgetting

**Scenario:** Used EWC for continual learning. Still forgot 30% of old task.
**Root Cause:** Fisher Information doesn't capture all important parameters.
**Solution:**
- **Increase Lambda:** Higher EWC penalty (1000 → 10000).
- **Memory Replay:** Store 10% of old task examples, replay during new task training.
- **Progressive Networks:** Add new parameters for new task instead of modifying old.
- **Multi-Task Training:** Train on both tasks simultaneously.

### Summary Checklist for Production
- [ ] **MoE:** Use for **scaling to >100B parameters** efficiently.
- [ ] **Long Context:** Use **sparse attention** for >8K tokens.
- [ ] **MQA/GQA:** Use for **1.5-2x faster inference**.
- [ ] **Model Merging:** Use **task arithmetic** for multi-task models.
- [ ] **Load Balancing:** Add **load balancing loss** for MoE.
- [ ] **Monitor:** Track **expert usage**, **memory**, **quality**.
