# Day 13: Transformer Training & Optimization
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is "Warmup" needed for Transformer training?

**Answer:**
Transformers, especially with Post-LayerNorm, have very unstable gradients at initialization.
- **Variance:** The variance of the output grows with depth. Early gradients can be massive.
- **Optimization Landscape:** The loss landscape is very rugged initially. A large learning rate can propel parameters into a "bad valley" from which they cannot recover (divergence).
- **Warmup:** Starting with a near-zero learning rate and linearly increasing it allows the model to slowly align its parameters and reduce gradient variance before taking larger steps.

#### Q2: Explain the difference between FP16 and BF16. Why is BF16 preferred for LLMs?

**Answer:**
- **FP16 (Half Precision):** 1 sign bit, 5 exponent bits, 10 mantissa bits.
    - *Issue:* Small dynamic range. Gradients often underflow (become 0) or overflow (become Inf). Requires **Loss Scaling**.
- **BF16 (Brain Float):** 1 sign bit, 8 exponent bits, 7 mantissa bits.
    - *Advantage:* Same exponent bits (dynamic range) as FP32.
    - *Trade-off:* Lower precision (mantissa), but neural networks are robust to noise.
    - *Benefit:* **No Loss Scaling needed.** Training is much more stable and easier to debug.

#### Q3: What is Gradient Checkpointing, and what is the trade-off?

**Answer:**
- **Concept:** Instead of storing all intermediate activations ($O(N)$ memory) for the backward pass, we only store a few "checkpoints" (e.g., every $\sqrt{N}$ layers).
- **Mechanism:** During backward pass, when gradients are needed for a layer between checkpoints, we **re-run the forward pass** from the nearest checkpoint to generate the activations on the fly.
- **Trade-off:** Reduces memory usage from $O(N)$ to $O(\sqrt{N})$ (allowing larger batches/models) but increases computation time by roughly 20-30% (due to re-forwarding).

#### Q4: How does AdamW differ from Adam, and why does it matter?

**Answer:**
- **Adam:** Adds L2 regularization to the loss function: $Loss = Loss_{data} + \lambda ||W||^2$. The gradient of this term is added to the moving averages ($m_t, v_t$).
- **AdamW:** Decouples weight decay. It applies the decay directly to the weights during the update step: $W_{t+1} = W_t - \eta (\dots) - \eta \lambda W_t$.
- **Why:** In adaptive optimizers like Adam, the L2 regularization in the gradient gets scaled by the adaptive learning rate ($1/\sqrt{v_t}$), leading to uneven decay. AdamW ensures consistent decay across all parameters, leading to better generalization.

#### Q5: You have a model that fits in GPU memory, but the batch size is too small (e.g., 2) for stable training. What do you do?

**Answer:**
**Gradient Accumulation.**
Run the forward and backward pass for $K$ micro-batches (e.g., 16 steps of size 2) without updating the weights. Accumulate the gradients. Then, perform one optimizer step.
Effective Batch Size = $2 \times 16 = 32$.
This mimics training with a large batch size without the memory cost.

---

### Production Challenges

#### Challenge 1: The "Loss Spike" Phenomenon

**Scenario:** Training is going well, loss is decreasing. Suddenly, at step 10,000, the loss spikes to a huge value or NaN, and the model collapses.
**Root Causes:**
1.  **Bad Data:** A single corrupted example or extremely long sequence caused gradients to explode.
2.  **Optimizer State Corruption:** A massive gradient update corrupted the variance estimates ($v_t$) in Adam.
**Solutions:**
- **Checkpointing:** Always save frequent checkpoints. Roll back to the last good one.
- **Data Cleaning:** Filter out extremely long documents or garbage text.
- **Gradient Clipping:** Ensure it's enabled (usually 1.0).
- **Skip Batch:** If a batch causes NaN, skip the update and move to the next.

#### Challenge 2: Training is Slow (Low GPU Utilization)

**Scenario:** You have A100 GPUs, but `nvidia-smi` shows only 40% utilization.
**Root Causes:**
1.  **Data Loading Bottleneck:** CPU cannot preprocess/tokenize data fast enough to feed the GPU.
2.  **Small Batch Size:** GPU kernels are not saturated.
3.  **Communication Overhead:** In multi-GPU training, GPUs are waiting for gradients to sync.
**Solutions:**
- **Dataloader:** Increase `num_workers`, use pre-tokenized datasets (Arrow/Parquet format).
- **Batch Size:** Increase micro-batch size to the limit of memory.
- **Flash Attention:** Use fused kernels to reduce memory bandwidth pressure.

#### Challenge 3: OOM (Out of Memory) despite small model

**Scenario:** A 1B model should fit on 24GB VRAM, but OOMs immediately.
**Root Causes:**
1.  **Optimizer States:** Adam stores 2 states per parameter (FP32). This triples the memory usage.
2.  **Fragmentation:** PyTorch memory allocator fragmentation.
**Solutions:**
- **Precision:** Use Mixed Precision (FP16/BF16).
- **Optimizer:** Use **8-bit Adam** (bitsandbytes) or **CPU Offloading** (DeepSpeed).
- **Gradient Checkpointing:** Enable it.

#### Challenge 4: Divergence with Post-LN Architecture

**Scenario:** Training a BERT-style (Post-LN) model from scratch, loss doesn't go down.
**Root Cause:** Post-LN is sensitive to initialization and warmup. Gradients can vanish or explode in deep layers early on.
**Solution:**
- **Switch to Pre-LN:** Much more stable.
- **Increase Warmup:** Extend warmup steps.
- **Lower LR:** Reduce max learning rate.

### Summary Checklist for Production
- [ ] **Data:** Pre-tokenized, shuffled, filtered.
- [ ] **Precision:** BF16 if supported, else FP16 with scaler.
- [ ] **Optimizer:** AdamW, $\beta_2=0.95$, Weight Decay=0.1.
- [ ] **Memory:** Gradient Checkpointing + Accumulation if needed.
- [ ] **Monitoring:** Log Loss, Grad Norm, LR, and GPU Util.
