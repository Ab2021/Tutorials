# Day 58: Distributed Training at Scale
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between DDP and FSDP?

**Answer:**
**DDP (Distributed Data Parallel):**
- Replicate entire model on each GPU
- Synchronize gradients via all-reduce
- **Memory:** Each GPU holds full model
- **Use:** Model fits on single GPU

**FSDP (Fully Sharded Data Parallel):**
- Shard parameters across GPUs (ZeRO-3)
- All-gather before forward/backward
- **Memory:** Each GPU holds 1/N of model
- **Use:** Model doesn't fit on single GPU

**Example:** 70B model, 8 GPUs → FSDP stores ~9B params per GPU.

#### Q2: Explain tensor parallelism and when to use it.

**Answer:**
- **Concept:** Split individual layers across GPUs
- **Column-parallel:** Split output dimension (Q, K, V projections)
- **Row-parallel:** Split input dimension (output projection)
- **Communication:** All-reduce after row-parallel layers

**When to use:**
- Very large layers (e.g., 20K hidden dim)
- Combine with pipeline parallelism for >100B models
- **Example:** Megatron-LM uses 8-way tensor parallelism

#### Q3: What is 3D parallelism?

**Answer:**
**Combine three types:**
1. **Data Parallelism:** Split batches across GPUs
2. **Pipeline Parallelism:** Split layers across GPUs
3. **Tensor Parallelism:** Split individual layers across GPUs

**Example (GPT-3):**
- 64-way data parallelism
- 8-way pipeline parallelism
- 8-way tensor parallelism
- **Total:** 64 × 8 × 8 = 4,096 GPUs

#### Q4: How does ZeRO reduce memory usage?

**Answer:**
**ZeRO Stages:**
- **ZeRO-1:** Partition optimizer states → 4x reduction
- **ZeRO-2:** Partition optimizer states + gradients → 8x reduction
- **ZeRO-3:** Partition optimizer states + gradients + parameters → Linear scaling

**Example (70B model, 64 GPUs):**
- Without ZeRO: 280GB per GPU (doesn't fit)
- With ZeRO-3: ~4GB per GPU (fits easily)

#### Q5: What is the communication overhead in distributed training?

**Answer:**
**DDP:** All-reduce gradients every step
- **Bandwidth:** 2 × model_size per step
- **Latency:** O(log N) with ring-allreduce

**FSDP:** All-gather params + reduce-scatter gradients
- **Bandwidth:** 3 × model_size per step
- **Latency:** Higher than DDP

**Mitigation:**
- Gradient compression (top-k sparsification)
- Overlap communication with computation
- Use high-bandwidth interconnect (NVLink, InfiniBand)

---

### Production Challenges

#### Challenge 1: DDP Hanging

**Scenario:** DDP training hangs indefinitely.
**Root Cause:** Uneven workload (some GPUs finish early, wait forever).
**Solution:**
- **Balanced Data:** Ensure all GPUs get same number of batches.
- **Drop Last:** Set `drop_last=True` in DataLoader.
- **Timeout:** Set `timeout=1800` in `init_process_group`.
- **Debug:** Use `NCCL_DEBUG=INFO` to see communication logs.

#### Challenge 2: FSDP OOM Despite Sharding

**Scenario:** FSDP still runs out of memory.
**Root Cause:** Activations not sharded, or batch size too large.
**Solution:**
- **Gradient Checkpointing:** Reduce activation memory.
- **Smaller Batch:** Reduce batch size per GPU (32 → 16).
- **CPU Offload:** Offload optimizer states to CPU.
- **Mixed Precision:** Use FP16 or BF16.

#### Challenge 3: Pipeline Parallelism Low GPU Utilization

**Scenario:** GPUs idle 50% of the time with pipeline parallelism.
**Root Cause:** Not enough micro-batches in pipeline.
**Solution:**
- **More Micro-batches:** Increase from 4 to 16.
- **Smaller Micro-batches:** Reduce micro-batch size.
- **Interleaved Pipeline:** Use GPipe or PipeDream schedule.
- **Trade-off:** More micro-batches = more memory for activations.

#### Challenge 4: Tensor Parallelism Communication Bottleneck

**Scenario:** Tensor parallelism slower than expected due to communication.
**Root Cause:** All-reduce overhead too high.
**Solution:**
- **Reduce TP Degree:** Use 4-way instead of 8-way.
- **Better Interconnect:** Use NVLink or InfiniBand.
- **Overlap Communication:** Fuse all-reduce with computation.
- **Sequence Parallelism:** Shard sequence dimension to reduce communication.

#### Challenge 5: 3D Parallelism Configuration

**Scenario:** Don't know how to configure 3D parallelism for 512 GPUs.
**Root Cause:** Many possible configurations.
**Solution:**
- **Start Simple:** DP=64, PP=8, TP=1 (512 GPUs).
- **Add TP if needed:** DP=64, PP=4, TP=2 (512 GPUs).
- **Profile:** Measure throughput for each configuration.
- **Rule of Thumb:** Maximize DP, minimize TP (TP has most communication).

### Summary Checklist for Production
- [ ] **DDP:** Use for **models that fit on 1 GPU**.
- [ ] **FSDP:** Use for **models >10B parameters**.
- [ ] **Pipeline Parallelism:** Use for **>100B models** with tensor parallelism.
- [ ] **Tensor Parallelism:** Use **4-8 way** for very large layers.
- [ ] **3D Parallelism:** Combine for **1000+ GPUs**.
- [ ] **Communication:** Use **NVLink/InfiniBand**, overlap with computation.
- [ ] **Monitor:** Track **GPU utilization**, **communication time**.
