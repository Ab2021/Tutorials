# 10. Distributed Training — Scaling to Multiple GPUs

## Table of Contents
- [Why Distributed Training?](#why-distributed-training)
- [Data Parallelism (DP)](#data-parallelism-dp)
- [Distributed Data Parallelism (DDP)](#distributed-data-parallelism-ddp)
- [Model Parallelism](#model-parallelism)
- [Tensor Parallelism](#tensor-parallelism)
- [Pipeline Parallelism](#pipeline-parallelism)
- [FSDP (Fully Sharded Data Parallel)](#fsdp-fully-sharded-data-parallel)
- [DeepSpeed: ZeRO Optimization](#deepspeed-zero-optimization)
- [HuggingFace Accelerate](#huggingface-accelerate)
- [When to Use Each Strategy](#when-to-use-each-strategy)

---

## Why Distributed Training?

```
Single GPU limitation:
  Gemma-2B with QLoRA: 6 GB VRAM → fits on 1 consumer GPU ✅
  Llama-70B with QLoRA: ~40 GB → needs A100 or multi-GPU ❌ on single

Speed limitation:
  Training Gemma-2B on 50K examples: ~4 hours on 1 GPU
  Same training on 4 GPUs: ~1 hour (near-linear speedup)
```

---

## Data Parallelism (DP)

### Simplest Multi-GPU Strategy

```
GPU 0                GPU 1                GPU 2                GPU 3
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Full Model   │    │ Full Model   │    │ Full Model   │    │ Full Model   │
│   Copy       │    │   Copy       │    │   Copy       │    │   Copy       │
│              │    │              │    │              │    │              │
│ Batch 1/4    │    │ Batch 2/4    │    │ Batch 3/4    │    │ Batch 4/4    │
│ → gradient₁  │    │ → gradient₂  │    │ → gradient₃  │    │ → gradient₄  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │                   │
       └───────────────────┼───────────────────┘                   │
                           │                                       │
                    ┌──────┴───────────────────────────────────────┘
                    │
              Average gradients → update ALL copies
```

### How It Works

```
1. Each GPU gets a FULL copy of the model
2. Training batch is SPLIT across GPUs
3. Each GPU computes gradients on its sub-batch
4. Gradients are AVERAGED across all GPUs (all-reduce)
5. All GPUs update their models identically

Effective batch size = per_gpu_batch × num_gpus
  4 GPUs × batch_size_4 = effective batch_size 16
```

### Limitations

```
❌ Each GPU needs enough memory for the FULL model
   (model weights + optimizer states + activations)
❌ Communication bottleneck: gradient all-reduce across GPUs
❌ GPU 0 is often a bottleneck (gathers/scatters data) in nn.DataParallel
```

---

## Distributed Data Parallelism (DDP)

### The Improvement Over DP

```
DP (DataParallel):
  GPU 0 gathers all gradients, averages, then broadcasts
  GPU 0 is the bottleneck → other GPUs wait

DDP (DistributedDataParallel):
  ALL GPUs participate equally in all-reduce
  No single bottleneck → much faster

DDP uses "Ring All-Reduce":
  GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
  
  Each GPU sends/receives a chunk of gradients simultaneously
  After n-1 rounds, all GPUs have the averaged gradients

Speedup: DDP is typically 10-20% faster than DP for 4+ GPUs
```

---

## Model Parallelism

### When the Model Doesn't Fit on One GPU

```
Gemma-7B in fp16: ~14 GB (model only)
Add optimizer states: ~42 GB
Add activations: ~55+ GB

Even an A100 (80 GB) struggles!

Solution: Split the model across multiple GPUs.
```

### Naive Model Parallelism

```
GPU 0              GPU 1              GPU 2
┌──────────┐      ┌──────────┐      ┌──────────┐
│ Layers   │      │ Layers   │      │ Layers   │
│  1-6     │ ──→  │  7-12    │ ──→  │ 13-18   │
│          │      │          │      │          │
│ Input    │      │ Hidden   │      │ Output   │
│ here     │      │ states   │      │ here     │
└──────────┘      └──────────┘      └──────────┘

Problem: "Bubble" inefficiency:
  GPU 0 computes layers 1-6   │ GPU 1 IDLE │ GPU 2 IDLE
  GPU 0 IDLE │ GPU 1 computes 7-12  │ GPU 2 IDLE
  GPU 0 IDLE │ GPU 1 IDLE │ GPU 2 computes 13-18

Only one GPU active at a time! 67% wasted compute.
```

---

## Tensor Parallelism

### Split Individual Layers Across GPUs

```
Instead of assigning whole layers to GPUs,
split EACH LAYER across GPUs:

Linear layer W (2048 × 2048):
  GPU 0 gets W[:, :1024]    (first half of columns)
  GPU 1 gets W[:, 1024:]    (second half of columns)

Each GPU computes HALF the output → combine with all-reduce.

Advantage: All GPUs work on EVERY layer simultaneously.
  No bubble inefficiency!

Disadvantage: Requires all-reduce communication WITHIN each layer.
  Only practical when GPUs are connected by fast interconnect (NVLink).
```

---

## Pipeline Parallelism

### Micro-Batching to Fill the Pipeline

```
Split the batch into micro-batches (MB):

Time →
GPU 0: [MB1 fwd]─[MB2 fwd]─[MB3 fwd]─[MB4 fwd]─[MB1 bwd]─[MB2 bwd]─...
GPU 1: ──────────[MB1 fwd]─[MB2 fwd]─[MB3 fwd]─[MB4 fwd]─[MB1 bwd]─...
GPU 2: ────────────────────[MB1 fwd]─[MB2 fwd]─[MB3 fwd]─[MB4 fwd]─...

While GPU 0 processes MB2 forward, GPU 1 processes MB1 forward.
The pipeline stays mostly full!

Bubble = (P-1)/M × 100% wasted time
  P = number of pipeline stages (GPUs)
  M = number of micro-batches
  
  4 GPUs, 16 micro-batches: (4-1)/16 = 18.75% bubble → acceptable
```

---

## FSDP (Fully Sharded Data Parallel)

### The Memory Problem with DDP

```
DDP: Every GPU stores:
  - Full model weights (4 GB)
  - Full optimizer states (16 GB)
  - Full gradients (4 GB)
  Total per GPU: 24 GB

With 4 GPUs: 4 × 24 GB = 96 GB total for just one model!
  Redundant: storing 4 copies of everything.
```

### FSDP: Shard Everything

```
FSDP: Each GPU stores ONLY 1/N of everything:

GPU 0           GPU 1           GPU 2           GPU 3
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ W[0:25%] │    │ W[25:50%]│    │ W[50:75%]│    │ W[75:100%]│
│ O[0:25%] │    │ O[25:50%]│    │ O[50:75%]│    │ O[75:100%]│
│ G[0:25%] │    │ G[25:50%]│    │ G[50:75%]│    │ G[75:100%]│
└─────────┘    └─────────┘    └─────────┘    └─────────┘

Memory per GPU: 24 GB / 4 = 6 GB
Total: 24 GB (no redundancy!)

When computing: temporarily all-gather needed weights → compute → discard.
```

---

## DeepSpeed: ZeRO Optimization

### Three Levels of Sharding

```
ZeRO Stage 1: Shard optimizer states only
  Memory reduction: ~4× for optimizer states
  Communication: Same as DDP

ZeRO Stage 2: Shard optimizer states + gradients
  Memory reduction: ~8× for optimizer + gradients
  Communication: Slightly more than DDP

ZeRO Stage 3: Shard optimizer states + gradients + model weights
  Memory reduction: Linear with number of GPUs
  Communication: Most communication
  Equivalent to FSDP
```

### ZeRO-Offload

```
Even with 1 GPU, offload optimizer states to CPU:

GPU Memory:                    CPU Memory:
┌──────────────┐              ┌──────────────────┐
│ Model weights│              │ Optimizer states  │
│ Activations  │              │ (m, v for Adam)   │
│ Gradients    │              │                   │
└──────────────┘              └──────────────────┘

Steps:
  1. Forward + backward on GPU (compute gradients)
  2. Send gradients to CPU
  3. Update optimizer states on CPU
  4. Send updated weights to GPU

Slower (CPU ↔ GPU transfer) but allows training MUCH larger models
on limited GPU memory.
```

---

## HuggingFace Accelerate

### Simplifying Distributed Training

```python
# Without Accelerate (raw PyTorch DDP):
import torch.distributed as dist
dist.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
sampler = DistributedSampler(dataset)
# ... lots of boilerplate

# With Accelerate:
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
# That's it! The same code runs on 1 GPU, 4 GPUs, or mixed precision.
```

---

## When to Use Each Strategy

```
1 GPU, model fits:
  → No parallelism needed (our case!)
  Just use QLoRA on a single GPU.

1 GPU, model doesn't fit:
  → ZeRO-Offload (DeepSpeed) or QLoRA
  Offload optimizer states to CPU.

2-4 GPUs, model fits on each:
  → DDP (DistributedDataParallel)
  Simple, effective, near-linear speedup.

2-4 GPUs, model doesn't fit on one:
  → FSDP or DeepSpeed ZeRO Stage 3
  Shard model across GPUs.

8+ GPUs (data center):
  → FSDP + Tensor Parallelism
  Combine techniques for maximum efficiency.

Our Project: 1 GPU + QLoRA = sufficient!
  QLoRA reduces memory enough to fit on a single consumer GPU.
  No distributed training needed for Gemma-2B.
```
