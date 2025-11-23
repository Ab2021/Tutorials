# Day 52: Efficient Training Techniques
## Core Concepts & Theory

### Training Efficiency Challenges

**Problems:**
- **Memory:** Large models don't fit in GPU memory
- **Speed:** Training takes weeks/months
- **Cost:** Millions of dollars in compute

**Solutions:**
- Mixed precision training
- Gradient accumulation
- Distributed training
- Memory-efficient optimizers
- Gradient checkpointing

### 1. Mixed Precision Training

**Concept:** Use FP16 for most operations, FP32 for critical ones

**Benefits:**
- **Speed:** 2-3x faster on modern GPUs
- **Memory:** 2x reduction
- **Accuracy:** Minimal loss with proper scaling

**Implementation:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    # Forward in FP16
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Gradient Accumulation

**Problem:** Batch size limited by GPU memory

**Solution:** Accumulate gradients over multiple mini-batches

**Effective Batch Size:** `mini_batch_size × accumulation_steps`

**Example:**
```python
accumulation_steps = 4
effective_batch_size = 32  # 8 × 4

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Distributed Training

**Data Parallelism:**
- Replicate model across GPUs
- Split data across GPUs
- Synchronize gradients

**Model Parallelism:**
- Split model across GPUs
- Each GPU handles part of model

**Pipeline Parallelism:**
- Split model into stages
- Each stage on different GPU
- Process micro-batches in pipeline

### 4. Memory-Efficient Optimizers

**AdamW Memory:** `2 × model_params` (momentum + variance)

**Adafactor:**
- Factorized second moments
- **Memory:** `O(√d)` instead of `O(d)`
- **Use Case:** Very large models

**8-bit Adam:**
- Quantize optimizer states to INT8
- **Memory:** 4x reduction
- **Accuracy:** Minimal loss

### 5. Gradient Checkpointing

**Problem:** Activations consume memory

**Solution:** Recompute activations during backward pass

**Trade-off:**
- **Memory:** 2-4x reduction
- **Speed:** 20-30% slower

**When to Use:** Model doesn't fit in memory

### 6. Flash Attention

**Standard Attention Memory:** O(N²)

**Flash Attention:**
- Tiled computation
- **Memory:** O(N)
- **Speed:** 2-3x faster

### 7. DeepSpeed Optimizations

**ZeRO (Zero Redundancy Optimizer):**

**ZeRO-1:** Partition optimizer states
- **Memory:** 4x reduction

**ZeRO-2:** Partition optimizer states + gradients
- **Memory:** 8x reduction

**ZeRO-3:** Partition optimizer states + gradients + parameters
- **Memory:** 64x reduction (for 64 GPUs)

### 8. FSDP (Fully Sharded Data Parallel)

**Concept:** Shard model parameters across GPUs

**Benefits:**
- **Memory:** Linear scaling with GPUs
- **Speed:** Efficient communication

**Use Case:** Train models larger than single GPU

### 9. LoRA (Low-Rank Adaptation)

**Concept:** Fine-tune only low-rank matrices

**Parameters:** `0.1-1%` of full model

**Memory:** 10-100x reduction

**Quality:** 90-95% of full fine-tuning

### 10. Quantization-Aware Training

**Concept:** Train with quantization in mind

**Benefits:**
- Model learns to be robust to quantization
- Better accuracy than post-training quantization

**Use Case:** Deploy quantized model

### Real-World Examples

**GPT-3 Training:**
- **Hardware:** 10,000 V100 GPUs
- **Time:** Weeks
- **Cost:** ~$5M
- **Techniques:** Model parallelism, mixed precision

**LLaMA Training:**
- **Hardware:** 2,048 A100 GPUs
- **Time:** 21 days (65B model)
- **Techniques:** FSDP, Flash Attention

### Summary

**Efficiency Techniques:**
- **Mixed Precision:** 2-3x speedup (easy win)
- **Gradient Accumulation:** Simulate large batches
- **Distributed Training:** Scale to multiple GPUs
- **ZeRO/FSDP:** Train models larger than single GPU
- **Gradient Checkpointing:** 2-4x memory reduction
- **LoRA:** 10-100x fewer parameters to train

### Next Steps
In the Deep Dive, we will implement these techniques with complete code examples and benchmarks.
