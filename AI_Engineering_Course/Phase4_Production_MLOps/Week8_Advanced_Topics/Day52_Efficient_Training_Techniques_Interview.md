# Day 52: Efficient Training Techniques
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is mixed precision training and what are its benefits?

**Answer:**
- **Concept:** Use FP16 for most operations, FP32 for critical ones (loss scaling, optimizer).
- **Benefits:**
  - **Speed:** 2-3x faster on modern GPUs (Tensor Cores).
  - **Memory:** 2x reduction (FP16 uses half the memory).
  - **Accuracy:** Minimal loss with proper gradient scaling.
- **Implementation:** PyTorch AMP (Automatic Mixed Precision).

#### Q2: Explain gradient accumulation and when to use it.

**Answer:**
- **Problem:** Batch size limited by GPU memory.
- **Solution:** Accumulate gradients over multiple mini-batches before updating weights.
- **Effective Batch Size:** `mini_batch_size × accumulation_steps`
- **Example:** Mini-batch=8, accumulation=4 → Effective batch=32.
- **When:** Want large batch size but GPU memory is limited.

#### Q3: What is the difference between DDP and FSDP?

**Answer:**
**DDP (Distributed Data Parallel):**
- Replicate entire model on each GPU.
- **Memory:** Each GPU holds full model.
- **Use:** Model fits on single GPU.

**FSDP (Fully Sharded Data Parallel):**
- Shard model parameters across GPUs.
- **Memory:** Each GPU holds 1/N of model (N=num GPUs).
- **Use:** Model doesn't fit on single GPU.

**Example:** 70B model, 8 GPUs → FSDP stores ~9B params per GPU.

#### Q4: How does LoRA reduce training memory?

**Answer:**
- **Concept:** Fine-tune only low-rank matrices (A, B) instead of full weights.
- **Parameters:** `rank × (in_dim + out_dim)` vs `in_dim × out_dim`
- **Example:** rank=8, dim=4096 → 65K params vs 16M params (250x reduction).
- **Memory:** Only train 0.1-1% of parameters.
- **Quality:** 90-95% of full fine-tuning performance.

#### Q5: What is ZeRO and how does it work?

**Answer:**
**ZeRO (Zero Redundancy Optimizer):**
- **ZeRO-1:** Partition optimizer states → 4x memory reduction.
- **ZeRO-2:** Partition optimizer states + gradients → 8x reduction.
- **ZeRO-3:** Partition optimizer states + gradients + parameters → Linear scaling with GPUs.

**Example:** 64 GPUs, ZeRO-3 → Each GPU stores 1/64 of model.

---

### Production Challenges

#### Challenge 1: Mixed Precision Overflow

**Scenario:** Loss becomes NaN during mixed precision training.
**Root Cause:** Gradient overflow/underflow in FP16.
**Solution:**
- **Gradient Scaling:** Use GradScaler (automatic in PyTorch AMP).
- **Loss Scaling:** Scale loss by 2^16 before backward, unscale before optimizer step.
- **Gradient Clipping:** Clip gradients to prevent overflow.
- **Check Scaling:** If scaler.get_scale() keeps decreasing, investigate model/data.

#### Challenge 2: DDP Hanging

**Scenario:** DDP training hangs indefinitely.
**Root Cause:** Uneven workload across GPUs (some GPUs finish early, wait forever).
**Solution:**
- **Balanced Data:** Ensure all GPUs get same number of batches.
- **Drop Last:** Set `drop_last=True` in DataLoader.
- **Timeout:** Set `timeout` in `init_process_group`.
- **Debug:** Use `NCCL_DEBUG=INFO` to see communication logs.

#### Challenge 3: FSDP OOM Despite Sharding

**Scenario:** FSDP still runs out of memory.
**Root Cause:** Activations not sharded, or batch size too large.
**Solution:**
- **Gradient Checkpointing:** Reduce activation memory.
- **Smaller Batch:** Reduce batch size per GPU.
- **CPU Offload:** Offload optimizer states to CPU.
- **Activation Checkpointing:** Use `activation_checkpointing` in FSDP.

#### Challenge 4: LoRA Quality Drop

**Scenario:** LoRA fine-tuning gives 20% lower accuracy than full fine-tuning.
**Root Cause:** Rank too low or wrong layers selected.
**Solution:**
- **Increase Rank:** Try rank=16 or 32 instead of 8.
- **More Layers:** Apply LoRA to more layers (all attention layers, not just Q/V).
- **Longer Training:** Train for more epochs (LoRA may need more iterations).
- **Learning Rate:** Try higher learning rate (1e-3 instead of 1e-4).

#### Challenge 5: Gradient Accumulation Instability

**Scenario:** Training unstable with gradient accumulation (loss spikes).
**Root Cause:** Effective batch size too large, or gradients not normalized properly.
**Solution:**
- **Normalize Loss:** Divide loss by `accumulation_steps` before backward.
- **Gradient Clipping:** Clip gradients after accumulation.
- **Smaller Accumulation:** Reduce accumulation steps (4 → 2).
- **Learning Rate:** Reduce learning rate proportionally to effective batch size.

### Summary Checklist for Production
- [ ] **Mixed Precision:** Use **AMP** for 2-3x speedup.
- [ ] **Gradient Accumulation:** Use for **large effective batch sizes**.
- [ ] **DDP:** Use for **multi-GPU training** (model fits on 1 GPU).
- [ ] **FSDP:** Use for **very large models** (>10B params).
- [ ] **Gradient Checkpointing:** Use for **2-4x memory reduction**.
- [ ] **LoRA:** Use for **efficient fine-tuning** (0.1-1% params).
- [ ] **Monitor:** Track **GPU memory**, **throughput**, **loss stability**.
