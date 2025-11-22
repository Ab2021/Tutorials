# Day 18: Model Parallelism
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the difference between Data Parallelism (DP) and Model Parallelism (MP).

**Answer:**
- **Data Parallelism:** The *entire model* fits on one GPU. We replicate the model $N$ times. Each GPU processes a different slice of the batch. We sync gradients (All-Reduce) at the end.
- **Model Parallelism:** The model is *too big* for one GPU. We split the model itself across GPUs.
    - **Pipeline Parallelism:** Split layers (GPU1: Layers 1-10, GPU2: Layers 11-20).
    - **Tensor Parallelism:** Split matrices (GPU1: Left half of W, GPU2: Right half of W).

#### Q2: Why is Tensor Parallelism usually restricted to a single node (e.g., 8 GPUs)?

**Answer:**
- **Communication Volume:** TP requires an All-Reduce operation *in every forward and backward pass of every layer*.
- **Bandwidth:** NVLink (intra-node) offers ~600-900 GB/s. Ethernet/InfiniBand (inter-node) offers ~50-100 GB/s.
- **Latency:** Inter-node latency is too high for the frequency of synchronization required by TP.
- **Conclusion:** Use TP within a node, and Pipeline/Data Parallelism across nodes.

#### Q3: What is the "Bubble" in Pipeline Parallelism, and how do we minimize it?

**Answer:**
- **Bubble:** The time when a GPU is idle waiting for data from the previous stage or gradients from the next stage.
- **Cause:** Sequential dependency of layers.
- **Solution:** **Micro-batching.** Split the global batch into many small chunks. Inject them into the pipeline rapidly.
    - **GPipe:** Flush pipeline after all micro-batches.
    - **1F1B:** Interleave forward and backward passes to keep GPUs busy and reduce memory.

#### Q4: How does Megatron-LM optimize the communication in the Transformer block?

**Answer:**
- A naive implementation would require 2 All-Reduces for the Attention block and 2 for the MLP block (4 total).
- Megatron-LM uses a specific split strategy (Column Parallel then Row Parallel).
- $Y = (X A_{col}) B_{row}$.
- The output of $X A_{col}$ is split across GPUs. $B_{row}$ expects split input.
- Therefore, no synchronization is needed *between* the two linear layers.
- **Result:** Only 2 All-Reduces per Transformer block (1 for Attention, 1 for MLP).

#### Q5: What is Sequence Parallelism?

**Answer:**
- **Problem:** In standard TP, the activation memory scales with sequence length $L$. For very long sequences (100k+), activations don't fit.
- **Solution:** Split the sequence dimension $L$ across GPUs.
- **Mechanism:** In Self-Attention, use **Ring Attention**. Pass Key/Value blocks in a ring topology so each GPU can compute attention for its local query tokens against all key tokens without storing full K/V matrices.

---

### Production Challenges

#### Challenge 1: Training a 175B Model (GPT-3 scale)

**Scenario:** You have 1000 A100s. How do you configure the parallelism?
**Analysis:**
1.  **Model Size:** 175B params $\approx$ 350GB (FP16).
2.  **GPU Memory:** 80GB.
3.  **Strategy:**
    - **Tensor Parallel (TP):** Size 8. (Fit layers on 1 node). Model becomes $350/8 \approx 44$GB/GPU. Fits!
    - **Pipeline Parallel (PP):** Size 16. (Split depth across 16 nodes).
    - **Data Parallel (DP):** Remaining GPUs. $1000 / (8 \times 16) \approx 7$ replicas.
**Result:** 3D Parallelism (TP=8, PP=16, DP=7).

#### Challenge 2: Slow Training due to Inter-Node Communication

**Scenario:** Training is slow. Profiler shows huge time spent in `ncclAllReduce`.
**Diagnosis:**
- You might be doing **Tensor Parallelism across nodes**.
- Check your topology. Ensure TP groups are within the same physical server.
- Check InfiniBand health. One slow cable slows down the entire cluster.

#### Challenge 3: Uneven GPU Memory Usage (OOM on GPU 0)

**Scenario:** In Pipeline Parallelism, GPU 0 runs out of memory while GPU 7 is empty.
**Root Cause:**
- **Embedding Layer:** GPU 0 holds the huge token embedding matrix ($V \times D$).
- **Activation Stashing:** GPU 0 stays alive the longest (waiting for backward pass of all micro-batches).
**Solution:**
- **Scatter Embeddings:** Don't put full embedding on GPU 0. Use TP for embeddings.
- **1F1B Schedule:** Aggressively free activations.
- **Balance Layers:** Give GPU 0 fewer transformer layers to compensate for embedding memory.

#### Challenge 4: Debugging Silent Accuracy Drop

**Scenario:** Loss decreases but accuracy is garbage.
**Root Cause:**
- **Synchronization Bug:** A tensor was split but not gathered correctly.
- **RNG Seed:** Dropout masks must be synchronized across TP replicas (or carefully handled). If GPU 1 drops index 5 and GPU 2 drops index 6, the reconstruction is invalid.
**Solution:**
- Use `torch.cuda.manual_seed` carefully.
- In Megatron-LM, use `get_cuda_rng_tracker` to manage seeds for model parallel regions vs data parallel regions.

### Summary Checklist for Production
- [ ] **Topology:** TP=8 (Intra-node), PP (Inter-node), DP (Inter-node).
- [ ] **Micro-batch:** Tune size to hide pipeline bubbles.
- [ ] **Network:** Ensure fast InfiniBand/RoCE for inter-node.
- [ ] **Framework:** Use DeepSpeed or Megatron-LM (don't write MP from scratch).
