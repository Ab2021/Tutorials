# Day 58: Distributed Training at Scale
## Core Concepts & Theory

### Distributed Training Fundamentals

**Goal:** Train models larger than single GPU memory

**Parallelism Types:**
- Data Parallelism
- Model Parallelism
- Pipeline Parallelism
- Tensor Parallelism

### 1. Data Parallelism (DP)

**Concept:** Replicate model across GPUs, split data

**Process:**
```
1. Each GPU has full model copy
2. Each GPU processes different batch
3. Gradients synchronized across GPUs
4. All GPUs update with averaged gradients
```

**Benefits:**
- Simple to implement
- Linear scaling (ideally)

**Limitations:**
- Model must fit on single GPU
- Communication overhead

### 2. Distributed Data Parallel (DDP)

**Improvement over DP:**
- More efficient gradient synchronization
- Uses NCCL for GPU communication
- Ring-AllReduce algorithm

**Speedup:**
- **2 GPUs:** 1.9x
- **4 GPUs:** 3.7x
- **8 GPUs:** 7.2x

### 3. Model Parallelism

**Concept:** Split model across GPUs

**Vertical Split:**
```
GPU 0: Layers 1-12
GPU 1: Layers 13-24
GPU 2: Layers 25-36
```

**Benefits:**
- Can train models larger than single GPU

**Limitations:**
- Sequential execution (GPU idle time)
- Communication overhead

### 4. Pipeline Parallelism

**Concept:** Split model + pipeline micro-batches

**Process:**
```
Micro-batch 1: GPU 0 → GPU 1 → GPU 2
Micro-batch 2:         GPU 0 → GPU 1 → GPU 2
Micro-batch 3:                 GPU 0 → GPU 1
```

**Benefits:**
- Better GPU utilization than naive model parallelism
- Reduces idle time

**GPipe:**
- Gradient accumulation across micro-batches
- Synchronous pipeline

### 5. Tensor Parallelism

**Concept:** Split individual layers across GPUs

**Example (Attention):**
```
Q, K, V = split_across_gpus(input)
Attention_GPU0 = Attention(Q0, K0, V0)
Attention_GPU1 = Attention(Q1, K1, V1)
Output = concat(Attention_GPU0, Attention_GPU1)
```

**Megatron-LM:**
- Tensor parallelism for Transformers
- Column-parallel and row-parallel layers

### 6. 3D Parallelism

**Concept:** Combine data + pipeline + tensor parallelism

**Example (GPT-3):**
- **Data Parallelism:** 64-way
- **Pipeline Parallelism:** 8-way
- **Tensor Parallelism:** 8-way
- **Total:** 64 × 8 × 8 = 4,096 GPUs

### 7. ZeRO (Zero Redundancy Optimizer)

**Problem:** Optimizer states consume 12 bytes per parameter

**ZeRO Stages:**

**ZeRO-1:** Partition optimizer states
- **Memory:** 4x reduction

**ZeRO-2:** Partition optimizer states + gradients
- **Memory:** 8x reduction

**ZeRO-3:** Partition optimizer states + gradients + parameters
- **Memory:** Linear scaling with GPUs

**Example:**
- **70B model on 64 GPUs:** Each GPU stores ~1B parameters

### 8. FSDP (Fully Sharded Data Parallel)

**PyTorch's ZeRO-3:**
- Shard parameters across GPUs
- All-gather before forward/backward
- Reduce-scatter after backward

**Benefits:**
- Simpler API than DeepSpeed
- Native PyTorch integration

### 9. Communication Optimization

**Gradient Compression:**
- Quantize gradients to INT8
- **Benefit:** 4x less communication

**Gradient Accumulation:**
- Accumulate gradients over multiple steps
- **Benefit:** Reduce communication frequency

**Overlap Communication:**
- Compute next layer while communicating gradients
- **Benefit:** Hide communication latency

### 10. Real-World Examples

**GPT-3 Training:**
- **4,096 A100 GPUs**
- **3D Parallelism:** 64 (data) × 8 (pipeline) × 8 (tensor)
- **Training time:** Weeks

**LLaMA 70B:**
- **2,048 A100 GPUs**
- **FSDP** for parameter sharding
- **Training time:** 21 days

**Megatron-Turing NLG (530B):**
- **4,480 A100 GPUs**
- **8-way tensor, 35-way pipeline, 16-way data**
- **Training time:** Months

### Summary

**Parallelism Strategies:**
- **Data Parallelism:** Model fits on 1 GPU
- **Model/Pipeline Parallelism:** Model doesn't fit on 1 GPU
- **Tensor Parallelism:** Very large layers
- **3D Parallelism:** Combine all three for maximum scale
- **ZeRO/FSDP:** Shard optimizer states + parameters

**Best Practices:**
- Start with DDP (simplest)
- Add FSDP if model doesn't fit
- Add pipeline/tensor for >100B models
- Optimize communication (compression, overlap)

### Next Steps
In the Deep Dive, we will implement DDP, FSDP, pipeline parallelism, and 3D parallelism with complete code examples.
