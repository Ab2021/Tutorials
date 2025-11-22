# Day 7: GPUs, Mixed Precision & Distributed Training  
## Core Concepts & Theory

### GPU Architecture Fundamentals

**Why GPUs for Deep Learning?**

CPUs: Optimized for sequential tasks (few powerful cores)
GPUs: Optimized for parallel tasks (thousands of smaller cores)

Deep learning: Massive matrix mult iplications → Perfect for GPUs!

**GPU Memory Hierarchy:**

```
Registers (fastest, smallest)
    ↓
Shared Memory / L1 Cache
    ↓
L2 Cache
    ↓
Global Memory (VRAM) (slowest, largest)
    ↓
CPU RAM (via PCIe)
```

**Typical GPU (A100):**
- CUDA Cores: 6912
- Tensor Cores: 432 (specialized for matrix multiply)
- Memory: 40GB or 80GB HBM2
- Memory Bandwidth: 1.5-2 TB/s
- FP32 Performance: 19.5 TFLOPS
- TF32 Performance: 156 TFLOPS (Tensor Cores)

### CUDA Programming Model

**Kernel Launch:**

```python
# PyTorch abstracts CUDA but understanding helps
# Conceptual CUDA kernel for matrix multiplication

@cuda.jit
def matmul_kernel(A, B, C):
    # Each thread computes one element of C
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# Launch with blocks and threads
threads_per_block = (16, 16)
blocks_per_grid = (
    (C.shape[0] + 15) // 16,
    (C.shape[1] + 15) // 16
)

matmul_kernel[blocks_per_grid, threads_per_block](A, B, C)
```

**In PyTorch:**

```python
# PyTorch handles all CUDA details
C = torch.matmul(A, B)  # Automatically uses optimized CUDA kernels
```

### Mixed Precision Training

**Precision Types:**

**FP32 (Float32):**
- 1 sign + 8 exponent + 23 mantissa = 32 bits
- Range: ±10^±38
- Precision: ~7 decimal digits
- Standard for decades

**FP16 (Float16):**
- 1 sign + 5 exponent + 10 mantissa = 16 bits  
- Range: ±65,504
- Precision: ~3 decimal digits
- **2× faster**, **2× less memory**

**BF16 (BFloat16):**
- 1 sign + 8 exponent + 7 mantissa = 16 bits
- Range: Same as FP32 (±10^±38)
- Precision: ~2 decimal digits
- **No overflow/underflow issues!**

**Comparison:**

| Type | Bits | Range | Precision | Speed | Memory |
|------|------|-------|-----------|-------|--------|
| FP32 | 32 | ±10^38 | 7 digits | 1× | 1× |
| FP16 | 16 | ±65K | 3 digits | 2× | 0.5× |
| BF16 | 16 | ±10^38 | 2 digits | 2× | 0.5× |
| INT8 | 8 | ±127 | Integer | 4× | 0.25× |

### Automatic Mixed Precision (AMP)

**PyTorch AMP:**

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward in FP16/BF16
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # Backward with gradient scaling (FP16 only)
    scaler.scale(loss).backward()
    
    # Unscale gradients, clip, step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Auto's Working:**

1. **Whitelist Operations** (use FP16/BF16):
   - Matrix multiplications (@, matmul)
   - Convolutions
   - Linear layers

2. **Blacklist Operations** (keep FP32):
   - Loss functions
   - Softmax
   - Log, exp

3. **Automatic Casting:**
   ```python
   # Inside autocast context
   x_fp16 = x.half()  # Input to matmul
   output_fp16 = torch.matmul(x_fp16, weight_fp16)
   output_fp32 = softmax(output_fp16.float())  # Softmax in FP32
   ```

**Gradient Scaling (FP16 only):**

```python
# Problem: Small gradients underflow in FP16
grad = 1e-7  # Underflows to 0 in FP16!

# Solution: Scale loss before backward
loss_scaled = loss * 2^16
loss_scaled.backward()  # Gradients also scaled

# Unscale before optimizer step
grad_unscaled = grad_scaled / 2^16
```

**Dynamic Scaling:**

```python
class GradScaler:
    def __init__(self):
        self.scale = 2^16
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
    
    def update(self):
        # If gradients had inf/nan: decrease scale
        # Else: try to increase scale periodically
        if self.found_inf:
            self.scale *= self.backoff_factor
        else:
            self.scale *= self.growth_factor  # Every N successful steps
```

### Distributed Training

**Data Parallelism:**

Replicate model on multiple GPUs, split data across them.

```python
# Simple DataParallel (single-process)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Forward: Splits batch across GPUs
# Backward: Gradients gathered and averaged
# Update: Weights synchronized
```

**DistributedDataParallel (DDP):**

Multi-process, better performance.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl', init_method='env://')

# Create model and move to local GPU
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()  # Gradients automatically all-reduced!
    optimizer.step()
```

**How DDP Works:**

1. **Forward**: Each GPU processes its shard independently
2. **Backward**: Gradients computed locally
3. **AllReduce**: Gradients averaged across all GPUs
   ```
   GPU 0: grad_0
   GPU 1: grad_1
   GPU 2: grad_2
   GPU 3: grad_3
   
   AllReduce:
   grad_avg = (grad_0 + grad_1 + grad_2 + grad_3) / 4
   
   All GPUs now have grad_avg
   ```
4. **Update**: Each GPU updates its model copy (identical!)

**Gradient AllReduce Optimization:**

Instead of waiting for all gradients, overlap communication with computation:

```
Layer N backward → AllReduce layer N → Layer N-1 backward → AllReduce layer N-1 → ...
              (overlap)                               (overlap)
```

This is automatic in DDP!

### Model Parallelism

For models too large for single GPU.

**Tensor Parallelism:**

Split individual layers across GPUs.

```python
# Example: Split linear layer
# W: (d_out, d_in) - too large!

# Split column-wise across 2 GPUs
W_0 = W[:, :d_in//2]  # GPU 0
W_1 = W[:, d_in//2:]  # GPU 1

# Forward
y_0 = x @ W_0.T  # GPU 0
y_1 = x @ W_1.T  # GPU 1

# AllGather results
y = torch.cat([y_0, y_1], dim=-1)
```

**Pipeline Parallelism:**

Split model layers across GPUs.

```python
# Model: Layer1 → Layer2 → Layer3 → Layer4

GPU 0: Layer1, Layer2
GPU 1: Layer3, Layer4

# Forward:
x = input
x = gpu0.forward(x)  # Layers 1-2
x = x.to(gpu1)
x = gpu1.forward(x)  # Layers 3-4
output = x

# Problem: GPU 1 idle during GPU 0 forward!
```

**Solution: Pipeline with Micro-batches:**

```
Time  GPU 0          GPU 1
 1    Batch1_F
 2    Batch2_F       Batch1_F
 3    Batch3_F       Batch2_F
 4    Batch1_B       Batch3_F
 5    Batch2_B       Batch1_B
 6    Batch3_B       Batch2_B

F = Forward, B = Backward
```

Both GPUs utilized!

### Fully Sharded Data Parallel (FSDP)

Combines data + model parallelism.

**ZeRO (Zero Redundancy Optimizer):**

**Problem with DDP:**
```
4 GPUs, each holds:
- Full model parameters (10GB)
- Full gradients (10GB)
- Full optimizer states (20GB for Adam)

Total: 40GB × 4 = 160GB redundant!
```

**ZeRO Solution:**

Shard optimizer states, gradients, and parameters.

**ZeRO Stage 1: Shard Optimizer States**
```
Each GPU holds 1/4 of optimizer states
Saves: 15GB per GPU × 4 = 60GB saved
```

**ZeRO Stage 2: Shard Gradients**
```
Each GPU holds 1/4 of gradients
Additional savings: 7.5GB per GPU
```

**ZeRO Stage 3 (FSDP): Shard Parameters**
```
Each GPU holds 1/4 of parameters
Only gathers full parameters when needed
Maximum savings!
```

**PyTorch FSDP:**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyModel()
model = FSDP(
    model,
    #sharding_strategy="FULL_SHARD",  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
)

# Training loop same as DDP!
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)  # Parameters gathered as needed
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Parameters sharded again after
```

**Memory Savings:**

```
Without FSDP (per GPU):
- Parameters: 10GB
- Gradients: 10GB
- Optimizer: 20GB
Total: 40GB per GPU

With FSDP (4 GPUs):
- Parameters: 2.5GB (sharded)
- Gradients: 2.5GB (sharded)
- Optimizer: 5GB (sharded)
Total: 10GB per GPU (4× reduction!)
```

### Gradient Checkpointing

Trade compute for memory.

**Problem:**
```
Forward: Must store all intermediate activations for backward
Memory ∝ num_layers × batch_size × seq_len × hidden_dim
```

**Solution:**
```
Don't store activations, recompute them during backward!
```

**Implementation:**

```python
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # Normal forward
        return self.layer(x)

class TransformerWithCheckpointing(nn.Module):
    def __init__(self, num_layers):
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            # Checkpoint: Don't store activations
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Trade-off:**

```
Memory saved: 30-50% (don't store activations)
Compute overhead: 20-30% (recompute during backward)

Net: Can train larger models or longer sequences!
```

### Summary: Scaling Training

**Scale Types:**

1. **Within-GPU**: Mixed precision, gradient accumulation
2. **Multi-GPU (single node)**: DDP
3. **Multi-Node**: DDP or FSDP
4. **Huge Models**: FSDP + mixed precision + gradient checkpointing

**Strategy Chart:**

| Model Size | Strategy |
|-----------|----------|
| < 1GB | Single GPU, FP32 |
| 1-10GB | Single GPU, mixed precision |
| 10-40GB | DDP (2-4 GPUs), mixed precision |
| 40-100GB | FSDP (4-8 GPUs), BF16 |
| 100GB-1TB | FSDP (8+ GPUs), BF16, gradient checkpointing |
| > 1TB | Tensor parallelism + pipeline + FSDP |

**Modern LLM Training (LLaMA-70B):**

```python
# FSDP + BF16 + gradient checkpointing
model = TransformerWithCheckpointing(num_layers=80)
model = FSDP(
    model,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    sharding_strategy="FULL_SHARD"
)

# Trained on 64× A100 GPUs (multi-node)
# Effective batch size: 4M tokens via gradient accumulation
```

This enables training 70B parameter models that wouldn't fit on any single GPU!
