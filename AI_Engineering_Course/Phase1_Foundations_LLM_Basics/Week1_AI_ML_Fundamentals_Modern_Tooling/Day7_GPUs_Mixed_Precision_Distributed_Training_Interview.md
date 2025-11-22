# Day 7: GPUs, Mixed Precision &  Distributed Training
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the difference between DataParallel and DistributedDataParallel. When would you use each?

**Answer:**

**DataParallel (DP):**

Single-process, multi-thread model parallelism.

```python
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

**How it works:**
1. **Forward**: Replica models on all GPUs, split batch across GPUs
2. **Backward**: Compute gradients on each GPU
3. **Gather**: Gather gradients to GPU 0
4. **Update**: Update model on GPU 0
5. **Broadcast**: Broadcast updated model to all GPUs

**Problems:**
- GPU 0 bottleneck (gathers all gradients, broadcasts all params)
- Single-process → Python GIL contention
- Inefficient GPU utilization

**DistributedDataParallel (DDP):**

Multi-process, each GPU has own process.

```python
# Each process
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

How it works:**
1. **Forward**: Each process independently
2. **Backward**: Compute gradients independently
3. **AllReduce**: Average gradients across all GPUs (efficient ring algorithm)
4. **Update**: Each process updates its model copy (identical)

**Advantages:**
- No GPU bottleneck (AllReduce is symmetric)
- Multi-process → No GIL
- Gradient overlapping with computation
- Faster!

**Performance Comparison:**

```
Model: ResNet-50, Batch: 256, 4× V100 GPUs

DataParallel:
- Time per epoch: 180s
- GPU 0 util: 95%, GPU 1-3 util: 65%

DistributedDataParallel:
- Time per epoch: 100s (1.8× faster!)
- All GPUs util: 90%
```

**When to Use:**

| Use Case | Choice |
|----------|--------|
| Single machine, < 4 GPUs, quick experiment | DataParallel (simpler code) |
| Single machine, ≥ 4 GPUs | DistributedDataParallel |
| Multiple machines | DistributedDataParallel (only option) |
| Production training | Always DDP |

**Interview Follow-up:**
*Q: How do you launch DDP training?*

**A:**
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py

# Or torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

---

#### Q2: Your model fits in GPU memory for batch_size=1 but you need batch_size=32 for training. What strategies can you use?

**Answer:**

**Problem:**

```
Model: 10GB
Batch=1: 2GB activations → Total 12GB (fits in 16GB GPU)
Batch=32: 64GB activations → Total 74GB (doesn't fit!)
```

**Solutions:**

**1. Gradient Accumulation**

Simulate larger batch by accumulating gradients:

```python
# Effective batch = 32, actual batch = 1
accumulation_steps = 32

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    # Forward with batch=1
    output = model(batch)  # Only 2GB activations
    loss = criterion(output, target) / accumulation_steps
    
    loss.backward()  # Gradients accumulate
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Pros:**
- Works on any GPU
- Mathematically equivalent to large batch

**Cons:**
- 32× slower (32 forwards/backwards per update)

**2. Gradient Checkpointing**

Trade compute for memory:

```python
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpointing(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# Saves ~50% activation memory
# Can now fit batch=2 instead of batch=1
```

**Combined with grad accumulation:**
```
Batch=2 × 16 accumulation steps = Effective batch=32
Only 16× slower (vs 32×)
```

**3. Mixed Precision (BF16/FP16)**

```python
with autocast(dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

# Activations in BF16 → 2× memory reduction
# Can fit batch=2
```

**4. Reduce Sequence Length (if applicable)**

```python
# For NLP: Use shorter sequences during training
max_seq_len = 512  # Instead of 2048

# Activations ∝ seq_len
# 4× reduction in seq_len = 4× less memory
# Can fit larger batch
```

**5. Multi-GPU (if available)**

```python
# DDP across 4 GPUs
# Each GPU: batch=8 → Total effective batch=32
model = DDP(model, device_ids=[local_rank])
```

**6. CPU Offloading (last resort)**

```python
from deepspeed. import DeepSpeed

# Offload optimizer states to CPU
config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}

model_engine = DeepSpeed.initialize(model=model, config=config)
```

**Recommended Strategy:**

```
1. Try BF16/FP16 first (2× memory, minimal cost)
2. Add gradient checkpointing (another 1.5-2× memory)
3. Use gradient accumulation for remaining gap
4. If multi-GPU available, use DDP

Example:
- Original: batch=1 (74GB doesn't fit)
- BF16: batch=2 (37GB fits!)
- + Checkpointing: batch=4 (28GB fits!)
- + Grad accum (8 steps): Effective batch=32 ✓
```

---

#### Q3: Explain FSDP (Fully Sharded Data Parallel). How does it differ from standard DDP and when should you use it?

**Answer:**

**DDP (DistributedDataParallel):**

Each GPU holds:
- Full model parameters
- Full gradients
- Full optimizer states

```
4 GPUs, 10B param model:

Each GPU:
- Parameters: 20GB (FP16)
- Gradients: 20GB
- Optimizer (Adam): 40GB (FP32 states)
Total per GPU: 80GB
```

**Redundancy:** Same data replicated 4 times!

**FSDP (ZeRO-3):**

Shard everything across GPUs.

```
4 GPUs, 10B param model:

Each GPU:
- Parameters: 5GB (1/4 of model)
- Gradients: 5GB (1/4)
- Optimizer: 10GB (1/4)
Total per GPU: 20GB (4× reduction!)
```

**How Forward Works:**

```python
# Layer i forward on GPU 0
def forward_layer_i(x):
    # Step 1: AllGather parameters from all GPUs
    local_shard = layer_i_params[rank]  # 1/4 of layer params
    full_params = all_gather([shard_0, shard_1, shard_2, shard_3])
    
    # Step 2: Compute forward
    output = compute(x, full_params)
    
    # Step 3: Free full_params (save memory!)
    del full_params
    
    return output
```

**Trade-off:**

```
DDP:
- Communication: 2× model size per step (gradient AllReduce)
- Memory: 1× model per GPU (redundant)

FSDP:
- Communication: 3× model size per step (AllGather×2 + ReduceScatter)
- Memory: 1/N model per GPU (sharded)
```

**When to Use:**

| Model Size | Strategy | Reasoning |
|-----------|----------|-----------|
| < 10GB | DDP | Fits in memory, DDP is simpler/faster |
| 10-40GB | DDP or FSDP | Either works, DDP slightly faster |
| 40-100GB | FSDP | Won't fit in single GPU with DDP |
| > 100GB | FSDP + CPU offload | Essential for very large models |

**Code Comparison:**

```python
# DDP
model = DDP(model, device_ids=[local_rank])

# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
)
```

**Interview Follow-up:**
*Q: FSDP has more communication. Why use it?*

**A:**
- Modern interconnects (NVLink, InfiniBand) are fast
- Communication overhead < 20%
- Memory savings enable training models that wouldn't fit otherwise
- Can use saved memory for larger batch → faster convergence → Nets win!

---

#### Q4: You're training with BF16 and notice some layers have very small gradients that might underflow. How do you handle this?

**Answer:**

**Problem:**

BF16 has limited precision (~2 decimal digits).

```
Gradient = 0.00012
In BF16: Rounded to 0.000122 or 0.000115 (precision loss)
```

Over many steps, precision loss accumulates → Suboptimal training.

**Check if It's Actually a Problem:**

```python
# Monitor gradient magnitudes
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm()
        grad_max = param.grad.data.abs().max()
        grad_min = param.grad.data[param.grad.data != 0].abs().min()
        
        print(f"{name}:")
        print(f"  Norm: {grad_norm:.2e}")
        print(f"  Max: {grad_max:.2e}")
        print(f"  Min (non-zero): {grad_min:.2e}")
```

If min gradients < 1e-4: Potential precision issues.

**Solutions:**

**1. Selective Mixed Precision**

Keep sensitive layers in FP32:

```python
class ModelWithSelectivePrecision(nn.Module):
    def __init__(self):
        self.embedding = nn.Embedding(...).to(torch.float32)  # FP32
        self.transformer = Transformer()  # Will be BF16
        self.output = nn.Linear(...).to(torch.float32)  # FP32
    
    def forward(self, x):
        # Embedding in FP32
        x = self.embedding(x)
        
        # Transformer in BF16
        with autocast(dtype=torch.bfloat16):
            x = self.transformer(x.to(torch.bfloat16))
        
        # Output in FP32
        x = self.output(x.to(torch.float32))
        return x
```

**2. Gradient Accumulation in FP32**

Accumulate gradients in FP32 even if model is BF16:

```python
# Inside autocast backward
with autocast(dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

# Backward WITHOUT autocast (gradients in FP32!)
loss.backward()  # Gradients stored in FP32
```

PyTorch master weights handle this automatically!

**3. Layer-wise Learning Rates**

Give layers with small gradients higher learning rates:

```python
optimizer = AdamW([
    {'params': model.embedding.parameters(), 'lr': 1e-3},
    {'params': model.transformer.parameters(), 'lr': 5e-4},
    {'params': model.output.parameters(), 'lr': 2e-3}  # 2× higher!
])
```

**4. Check if BF16 is Appropriate**

For some tasks, FP32 might be necessary:

```python
# Scientific computing, very small gradients
# Use FP32 throughout (slower but accurate)
```

**5. Verify Training is Still Effective**

```python
# Compare FP32 vs BF16 training
# Small validation loss difference (< 1%) → BF16 fine
# Large difference (> 5%) → Need FP32 or selective precision
```

**Production Recommendation:**

```python
# Default: BF16 for transformer layers
# Keep embeddings and final layers in FP32
# Monitor gradient norms
# If issues: Selectively increase precision
```

**Real Example (GPT-3):**

- Most layers: BF16
- Embedding layer: FP32
- Final projection: FP32
- Reason: Gradients in embeddings/output are typically smaller

---

#### Q5: You have 8× A100 GPUs (80GB each). Your model is 60GB. Design a training strategy to maximize throughput.

**Answer:**

**Analysis:**

- Model: 60GB per GPU (parameters + optimizer states + activations)
- Available: 80GB per GPU
- Headroom: 20GB per GPU

**Goal:** Maximize tokens/second processed.

**Strategy:**

**Option 1: DDP (Replicate Model on All GPUs)**

```python
model = DDP(model, device_ids=[local_rank])

# Each GPU:
# - Model: 60GB
# - Batch: Maximize to use remaining 20GB
# - Effective batch: 8 × per_gpu_batch
```

**Per-GPU batch size:**
```
20GB / (seq_len × hidden_dim × 2 bytes) tokens

Assuming seq_len=2048, hidden_dim=4096:
20GB / (2048 × 4096 × 2) ≈ 1.2M elements
≈ batch_size=4 sequences
```

**Effective batch:**
```
8 GPUs × 4 sequences = 32 sequences
= 32 × 2048 tokens = 65K tokens per step
```

**Option 2: FSDP (Shard Model Across GPUs)**

```python
model = FSDP(model, sharding_strategy="FULL_SHARD")

# Each GPU:
# - Model: 60GB / 8 = 7.5GB (sharded)
# - Available for batch: 80 - 7.5 = 72.5GB!
```

**Per-GPU batch size:**
```
72.5GB / (2048 × 4096 × 2) ≈ 4.4M elements
≈ batch_size=14 sequences
```

**Effective batch:**
```
8 GPUs × 14 sequences = 112 sequences
= 112 × 2048 = 229K tokens per step (3.5× more!)
```

**Trade-off:**

```
DDP:
- Throughput: 65K tokens/step
- Communication: Less (only gradients)
- Simpler code

FSDP:
- Throughput: 229K tokens/step (3.5× better!)
- Communication: More (AllGather params each layer)
- On A100 with NVLink: Communication overhead ~15%

Net: FSDP wins! (3.5× / 1.15 = 3× effective speedup)
```

**Recommended Configuration:**

```python
# FSDP with BF16, gradient checkpointing
model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Overlap comm
    activation_checkpointing=True  # Further memory savings
)

# Optuna hyperparams
per_gpu_batch = 16      # Push to limit (with checkpointing)
gradient_accum = 4      # Larger effective batch
effective_batch = 8 × 16 × 4 = 512 sequences = 1M tokens

# Learning rate scaled with batch size
lr = base_lr × sqrt(effective_batch / base_batch)
```

**Expected Throughput:**

```
Per-GPU: ~2000 tokens/sec
Total: 8 × 2000 = 16,000 tokens/sec

Training 1T tokens: 1T / 16K tokens/sec ≈ 17 hours
```

---

### Production Challenges

**Challenge: OOM After Several Hours of Stable Training**

**Scenario:**
- Train for 5 hours normally
- Suddenly OOM (Out Of Memory) error
- No code changes

**Root Causes:**

1. **Gradient Accumulation:**
   - Some graphs not freed properly
   - Memory leak in custom ops

2. **Dynamic Shapes:**
   - Variable sequence lengths
   - Rare long sequence causes OOM

3. **Fragmentation:**
   - PyTorch memory allocator fragmentation
   - Especially with mixed precision

**Solutions:**
```python
# Set max seq length strictly
max_seq_len = 2048

# Empty cache periodically
if step % 1000 == 0:
    torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.95, device=0)
```

---

### Key Takeaways for Interviews

1. **Understand trade-offs**: DDP vs FSDP (memory vs communication)
2. **Mixed precision benefits**: 2× speedup, minimal accuracy loss
3. **Scaling strategies**: Gradient accumulation, checkpointing, sharding
4. **Debugging skills**: OOM, gradient issues, performance bottlenecks
5. **Production awareness**: Monitor memory, throughput, handle edge cases
