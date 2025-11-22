# Day 13: Transformer Training & Optimization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Gradient Accumulation

**The Problem:** Large batch sizes (e.g., 512, 1024) are crucial for stable Transformer training, but they don't fit in GPU memory.
**The Solution:** Simulate a large batch by accumulating gradients over multiple smaller "micro-batches" before updating weights.

**Mechanism:**
1.  Forward pass with micro-batch (e.g., size 32).
2.  Compute loss.
3.  **Scale loss:** `loss = loss / accumulation_steps`. This is critical to keep the magnitude of the gradients correct (averaging).
4.  Backward pass (`loss.backward()`). Gradients accumulate in `.grad` attributes.
5.  Repeat for `accumulation_steps`.
6.  `optimizer.step()` to update weights.
7.  `optimizer.zero_grad()` to clear gradients.

**Code:**
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch['input_ids'])
    loss = criterion(outputs.logits, batch['labels'])
    
    # Scale loss
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Automatic Mixed Precision (AMP)

**The Concept:** Use 16-bit floating point (FP16 or BF16) for most operations to save memory and compute, while keeping master weights in FP32 for stability.

**FP16 vs BF16:**
- **FP16:** 5 bits exponent, 10 bits mantissa. Prone to underflow/overflow. Requires **Gradient Scaling**.
- **BF16 (Brain Float):** 8 bits exponent (same as FP32), 7 bits mantissa. Less precision but same dynamic range as FP32. **No gradient scaling needed.** Preferred for LLMs.

**Gradient Scaling (for FP16):**
1.  Multiply loss by a large factor (e.g., $2^{16}$).
2.  Backward pass (gradients are now large enough to not underflow).
3.  Unscale gradients (divide by factor).
4.  If `Inf` or `NaN` found, skip update and decrease scale factor.
5.  Else, update weights.

**Code (PyTorch AMP):**
```python
scaler = torch.cuda.amp.GradScaler() # Only for FP16

with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = model(input)
    loss = criterion(outputs, labels)

# Scale loss and backward
scaler.scale(loss).backward()

# Unscale and update
scaler.step(optimizer)
scaler.update()
```

### 3. Gradient Checkpointing (Activation Checkpointing)

**The Trade-off:** Trade compute for memory.
**Standard Backprop:** Store all intermediate activations during forward pass to use in backward pass. Memory $O(N)$.
**Checkpointing:** Do not store intermediate activations. During backward pass, **recompute** them from the nearest checkpoint.
**Result:** Memory drops to $O(\sqrt{N})$, allowing much deeper models or larger batches. Cost is ~20-30% extra compute.

**Implementation:**
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Instead of: x = self.layer(x)
    x = checkpoint(self.layer, x)
    return x
```

### 4. ZeRO (Zero Redundancy Optimizer)

**The Problem:** In Data Parallelism, every GPU holds a full copy of the Model, Gradients, and Optimizer States. This is redundant.
**The Solution:** Shard these states across GPUs.

- **ZeRO Stage 1:** Shard Optimizer States (32-bit master weights, momentum). ~4x memory savings.
- **ZeRO Stage 2:** Shard Gradients. ~2x additional savings.
- **ZeRO Stage 3:** Shard Model Parameters. Memory scales linearly with number of GPUs.

**DeepSpeed / FSDP (Fully Sharded Data Parallel):**
PyTorch's FSDP implements ZeRO-3.

### 5. The "Curse" of Batch Normalization in Transformers

Transformers use **Layer Normalization**, not Batch Normalization.
**Why?**
1.  **NLP Statistics:** In NLP, batch statistics (mean/var of words across sentences) fluctuate wildly and aren't representative.
2.  **Sequence Length:** Batch Norm is hard to apply when sequences have different lengths (padding issues).
3.  **Layer Norm:** Normalizes across the feature dimension for a *single* token. Independent of batch size and other tokens.

### 6. Learning Rate Warmup: The "Why"

**Variance of Gradients:**
At initialization, layers are random. The output of the Transformer is a sum of many random variables (residual path).
The gradient variance through these layers can be very high.
A large step size early on can push the model into a region where gradients explode or vanish.
**Warmup** allows the model to learn "direction" with small steps, stabilizing the variance of the gradients before accelerating.
**Pre-LN** (Pre-Layer Norm) architecture significantly reduces the need for warmup compared to Post-LN, but warmup is still best practice.

### Summary Table

| Technique | Saves Memory? | Saves Compute? | Complexity |
| :--- | :--- | :--- | :--- |
| **Gradient Accumulation** | Yes | No | Low |
| **Mixed Precision (AMP)** | Yes (2x) | Yes (Tensor Cores) | Medium |
| **Checkpointing** | Yes (Significant) | No (Adds cost) | Low |
| **ZeRO / FSDP** | Yes (Massive) | No (Comm overhead) | High |
