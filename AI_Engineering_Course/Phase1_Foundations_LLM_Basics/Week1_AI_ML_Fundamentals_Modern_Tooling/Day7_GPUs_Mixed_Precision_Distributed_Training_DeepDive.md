# Day 7: GPUs, Mixed Precision & Distributed Training
## Deep Dive - Internal Mechanics & Advanced Reasoning

### CUDA Memory Model: Deep Dive

**GPU Memory Hierarchy Performance:**

```
Registers: ~1 cycle latency, ~20 TB/s bandwidth
Shared Memory / L1: ~28 cycles, ~14 TB/s
L2 Cache: ~200 cycles, ~3 TB/s
Global Memory (HBM): ~400 cycles, ~1.5 TB/s
Host Memory (RAM): ~10,000+ cycles (via PCIe)
```

**Why Matrix Multiplication is Fast on GPUs:**

**Naive CPU Implementation:**

```python
# Sequential, cache-inefficient
for i in range(M):
    for  j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j]  # Memory access pattern poor!

Time: O(M×N×K) with sequential execution
```

**GPU with Tiling:**

```python
# Conceptual CUDA kernel with shared memory tiling
TILE_SIZE = 16

@cuda.jit
def matmul_tiled(A, B, C):
    # Shared memory for tile (fast!)
    tile_A = cuda.shared.array((TILE_SIZE, TILE_SIZE), float32)
    tile_B = cuda.shared.array((TILE_SIZE, TILE_SIZE), float32)
    
    row = cuda.blockIdx.x * TILE_SIZE + cuda.threadIdx.x
    col = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
    
    tmp = 0.0
    
    # Loop over tiles
    for m in range(0, K, TILE_SIZE):
        # Load tile from global to shared memory
        tile_A[cuda.threadIdx.x, cuda.threadIdx.y] = A[row, m + cuda.threadIdx.y]
        tile_B[cuda.threadIdx.x, cuda.threadIdx.y] = B[m + cuda.threadIdx.x, col]
        cuda.syncthreads()  # Wait for all threads to load
        
        # Compute using shared memory (fast!)
        for k in range(TILE_SIZE):
            tmp += tile_A[cuda.threadIdx.x, k] * tile_B[k, cuda.threadIdx.y]
        cuda.syncthreads()  # Wait before loading next tile
    
    C[row, col] = tmp
```

**Key Optimizations:**

1. **Tiling**: Load data once to shared memory, reuse many times
2. **Coalesced Memory Access**: Adjacent threads access adjacent memory (full bandwidth)
3. **Parallelism**: Thousands of threads working simultaneously

**Tensor Cores (A100):**

Specialized hardware for (matmul):

```
# Single Tensor Core operation (1 clock cycle):
D = A @ B + C

Where A, B, C, D are 16×16 matrices (FP16/BF16)
```

**Performance:**

```
Without Tensor Cores (CUDA Cores):
- FP32 matmul: 19.5 TFLOPS

With Tensor Cores:
- TF32 (FP32 emulation): 156 TFLOPS (8× faster!)
- BF16: 312 TFLOPS (16× faster!)
```

### Mixed Precision: Numerical Stability Analysis

**FP16 Accumulation Problem:**

```python
# Summing many small numbers in FP16
values = [1e-4] * 100000  # 100K elements

# FP16 accumulation
sum_fp16 = torch.tensor(0.0, dtype=torch.float16)
for v in values:
    sum_fp16 += v

print(sum_fp16)  # Expected: 10.0, Actual: ~8.5 (precision loss!)

# FP32 accumulation
sum_fp32 = torch.tensor(0.0, dtype=torch.float32)
for v in values:
    sum_fp32 += v

print(sum_fp32)  # Expected: 10.0, Actual: 10.0 (correct!)
```

**Why?**

FP16 has 10-bit mantissa → precision ~0.001.

After accumulating ~1000 values, accumulated sum ≈ 0.1.

Adding 1e-4 to 0.1:
```
0.1 + 0.0001 ≈ 0.1001

In FP16 with 3 decimal digits precision:
0.1001 → 0.100 (rounded, 0.0001 lost!)
```

**Solution in PyTorch:**

```python
# Automatic upcasting for accumulation
with autocast():
    x_fp16 = x.half()
    y_fp16 = torch.sum(x_fp16)  # Internally uses FP32 accumulator!
    # Returns FP16 result but computed in FP32
```

**Loss Scaling Mathematics:**

**Problem:**

Gradients often in range [1e-7, 1e-3].

FP16 min representable: 6e-8.

Gradients < 6e-8 → 0 (underflow!).

**Solution:**

Scale loss by S (e.g., S = 2^16 = 65536).

```
Scaled loss = S × L
Scaled gradients = S × ∇L

Range: [S × 1e-7, S × 1e-3] = [6e-3, 65.5]

No underflow!
```

**After backward, unscale:**

```
True gradients = Scaled gradients / S
```

**Dynamic Scaling Algorithm:**

```python
class DynamicGradScaler:
    def __init__(self, init_scale=2**16):
        self.scale = init_scale
        self.growth_interval = 2000
        self.backoff_factor = 0.5
        self.growth_factor = 2.0
        self.growth_tracker = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def update_scale(self, found_inf_or_nan):
        if found_inf_or_nan:
            # Overflow detected: reduce scale
            self.scale *= self.backoff_factor
            self.growth_tracker = 0
            return "skip_step"  # Skip optimizer step
        else:
            # Successful step: try to grow scale
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_tracker = 0
            return "success"
```

**Why This Works:**

- Start with large scale (minimize underflow)
- If overflow (inf/nan): reduce scale, skip step
- If many successful steps: increase scale (maximize precision)
- Converges to optimal scale automatically!

### BFloat16: Why It's Better

**Comparison:**

```
FP32: s1 e8 m23 | Range: ±3.4×10^38 | Precision: ~7 digits
FP16: s1 e5 m10 | Range: ±65,504   | Precision: ~3 digits
BF16: s1 e8 m7  | Range: ±3.4×10^38 | Precision: ~2 digits
```

**Key Insight:**

BF16 sacrifices precision for range (same as FP32).

**Why This Matters:**

**Scenario: Large Logits**

```python
logits = torch.tensor([100.0, 95.0, 90.0], dtype=torch.float16)

# FP16: Max value is 65,504
# 100 is fine, but what if logits grow during training?

logits_large = torch.tensor([70000.0, 69000.0, 68000.0])  # Overflow in FP16!
```

BF16: Can represent up to 3.4×10^38 (no overflow!).

**Gradient Dynamics:**

```python
# Early training: Large gradients (0.1 to 10)
# Late training: Small gradients (1e-4 to 1e-2)

# FP16: Risk overflow early, underflow late
# BF16: Same range as FP32, no issues!
```

**Conversion Cost:**

FP32 ↔  BF16: Simple truncation/zero-extension (cheap!)

```
FP32: seeeeeee emmmmmmmmmmmmmmmmmmmmmmm
BF16: seeeeeee emmmmmm
              └─────┘ Just drop these bits!
```

FP32 ↔ FP16: Complex (adjust exponent, expensive).

### DistributedDataParallel (DDP): Communication Analysis

**AllReduce Operation:**

**Ring AllReduce (Efficient):**

```
4 GPUs, N elements per GPU

Step 1: Each GPU sends chunk to next GPU
GPU 0 → GPU 1
GPU 1 → GPU 2
GPU 2 → GPU 3
GPU 3 → GPU 0

Step 2-4: Rotate and accumulate

Step 5-7: Distribute final result

Total communication: 2(N) per GPU
Bandwidth efficient!
```

**Latency:**

```
Latency = α + β × N

α = latency (overhead)
β = 1 / bandwidth
N = data size
```

For small N: α dominates (latency-bound).
For large N: β×N dominates (bandwidth-bound).

**DDP Bucketing:**

Problem: Thousands of small parameters → high α overhead.

Solution: Bucket gradients (group into larger chunks).

```python
# DDP automatically buckets gradients
model = DDP(
    model,
    bucket_cap_mb=25  # 25 MB buckets
)

# Groups gradients into ~25MB chunks before AllReduce
# Reduces number of AllReduce calls
```

**Gradient Overlapping:**

```
Traditional:
├─ Forward all layers
├─ Backward all layers
└─ AllReduce all gradients  ← Idle during all reduce!

DDP with Overlapping:
├─ Forward all layers
├─ Backward layer N        ├─ AllReduce layer N grads
├─ Backward layer N-1      ├─ AllReduce layer N-1 grads
└─ ...

Backward and AllReduce happen simultaneously!
```

**Performance Impact:**

```
Without overlapping: 100ms backward + 50ms AllReduce = 150ms
With overlapping: max(100ms, 50ms) = 100ms (33% faster!)
```

### FSDP: Hierarchical Sharding

**ZeRO-3 (FSDP) Algorithm:**

**Forward Pass:**

```python
# Layer i forward
# Each GPU holds 1/N of layer i parameters

def forward_layer_i(x, rank, world_size):
    # Step 1: AllGather parameters
    param_shard = model.layer_i.weight_shard[rank]  # Local shard
    all_params = all_gather(param_shard, world_size)  # Gather from all GPUs
    
    # Step 2: Compute forward
    output = torch.matmul(x, all_params.T)
    
    # Step 3: Free AllGathered parameters (save memory!)
    del all_params
    
    return output
```

**Backward Pass:**

```python
def backward_layer_i(grad_output, rank, world_size):
    # Step 1: AllGather parameters again (need for backward)
    param_shard = model.layer_i.weight_shard[rank]
    all_params = all_gather(param_shard, world_size)
    
    # Step 2: Compute local gradient
    grad_local = compute_gradient(grad_output, all_params)
    
    # Step 3: ReduceScatter gradient (each GPU keeps 1/N)
    grad_shard = reduce_scatter(grad_local, world_size)
    
    # Step 4: Free AllGathered parameters
    del all_params
    
    # Step 5: Update local shard
    optimizer.step_on_shard(grad_shard)
```

**Communication Volume:**

Per layer forward + backward:
```
AllGather params: N bytes
AllGather params (backward): N bytes
ReduceScatter grads: N bytes

Total: 3N bytes (vs 2N for DDP)
```

**But**: Memory savings enable larger models!

**Memory Breakdown:**

```
Model: 10B params × 2 bytes (BF16) = 20GB

DDP (4 GPUs):
- Each GPU: 20GB params + 20GB grads + 40GB optimizer = 80GB

FSDP (4 GPUs):
- Each GPU: 5GB params + 5GB grads + 10GB optimizer = 20GB
```

4× memory reduction!

### Gradient Checkpointing: Compute-Memory Trade-off

**Activation Memory:**

```
Transformer layer:
Input: (batch, seq_len, d_model)
After attention: (batch, seq_len, d_model)
After FFN: (batch, seq_len, d_model)

Must store both for backward!

For GPT-3 (96 layers):
Activations: 96 × batch × seq_len × d_model × 2 bytes
            = 96 × 32 × 2048 × 12288 × 2
            ≈ 150GB!
```

**Checkpointing:**

```python
# Standard (store all activations)
def forward_layer(x):
    attn_out = attention(x)  # Store for backward
    ffn_out = ffn(attn_out)  # Store for backward
    return ffn_out

# With checkpointing (don't store)
def forward_layer_checkpoint(x):
    def custom_forward(x_inner):
        attn_out = attention(x_inner)
        ffn_out = ffn(attn_out)
        return ffn_out
    
    # Only store input x, not intermediate activations
    return torch.utils.checkpoint.checkpoint(custom_forward, x)
```

**Backward:**

```python
# Recompute forward to get activations!
def backward_with_recompute():
    # Re-run forward (extra compute)
    attn_out = attention(x)  # Recomputed
    ffn_out = ffn(attn_out)  # Recomputed
    
    # Now can compute gradients
    grad backward pass
```

**Trade-off Quantification:**

```
Without checkpointing:
- Memory: 150GB
- Compute: 1× forward + 1× backward = 2× layer compute

With checkpointing:
- Memory: ~50GB (3× reduction)
- Compute: 2× forward + 1× backward = 3× layer compute (50% overhead)

Net: Can train 3× larger model with 50% more time!
```

### Summary: Scaling Modern LLMs

**LLaMA-70B Training Configuration:**

```python
# Model
model = LLaMATransformer(
    num_layers=80,
    d_model=8192,
    num_heads=64
)

# FSDP wrapping
model = FSDP(
    model,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    sharding_strategy="FULL_SHARD",  # ZeRO-3
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Overlap comm
    cpu_offload=None  # Keep all on GPU
)

# Gradient checkpointing
model.enable_gradient_checkpointing()

# Hardware
# 64 GPUs (8 nodes × 8× A100 80GB)

# Effective batch size
# Global batch: 4M tokens
# Per-GPU batch: 64K tokens
# Gradient accumulation: 8 steps
# Per-GPU per-step: 8K tokens
```

**

Performance:**

```
Throughput: ~50K tokens/sec across all GPUs
Training time: ~21 days for 1.4T tokens
Cost: ~$2M in GPU time
```

This is state-of-the-art LLM training (2024-2025)!
