# 12. GPU Architecture & CUDA — How Hardware Powers Deep Learning

## Table of Contents
- [CPU vs GPU: Why GPUs Win at ML](#cpu-vs-gpu-why-gpus-win-at-ml)
- [GPU Architecture Deep Dive](#gpu-architecture-deep-dive)
- [Memory Hierarchy](#memory-hierarchy)
- [Tensor Cores](#tensor-cores)
- [CUDA Programming Model](#cuda-programming-model)
- [How PyTorch Uses CUDA](#how-pytorch-uses-cuda)
- [GPU Memory Management](#gpu-memory-management)
- [Benchmarking and Profiling](#benchmarking-and-profiling)
- [Choosing a GPU for Fine-Tuning](#choosing-a-gpu-for-fine-tuning)

---

## CPU vs GPU: Why GPUs Win at ML

### The Fundamental Difference

```
CPU (Central Processing Unit):
  Few powerful cores (8-64), optimized for SEQUENTIAL tasks.
  Each core: complex logic, branch prediction, large cache.
  Good at: if/else decisions, operating system tasks, single-thread speed.

GPU (Graphics Processing Unit):
  MANY simple cores (thousands), optimized for PARALLEL tasks.
  Each core: simple arithmetic, no complex logic, small cache.
  Good at: doing the SAME operation on MANY data points simultaneously.

Matrix multiplication: y = W × x
  CPU:  for each output element: compute dot product (sequential)
       → 1 core × many operations = slow for large matrices
  
  GPU:  EACH output element computed by a DIFFERENT core (parallel)
       → 4096 cores × 1 operation each = massively faster

Example for a 2048×2048 matrix multiply:
  CPU (8 cores):   ~4 million operations per core = 0.5 seconds
  GPU (4096 cores): ~1000 operations per core = 0.001 seconds (500× faster!)
```

### Visual Comparison

```
CPU (8 cores):
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ Big  │ │ Big  │ │ Big  │ │ Big  │
  │ Core │ │ Core │ │ Core │ │ Core │
  └──────┘ └──────┘ └──────┘ └──────┘
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ Big  │ │ Big  │ │ Big  │ │ Big  │
  │ Core │ │ Core │ │ Core │ │ Core │
  └──────┘ └──────┘ └──────┘ └──────┘
  = 8 powerful workers

GPU (10,752 cores on RTX 4090):
  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐
  │·││·││·││·││·││·││·││·││·││·││·││·││·││·││·││·││·││·││·││·│
  └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘
  (... 10,752 small cores total ...)
  = thousands of simple workers
```

---

## GPU Architecture Deep Dive

### NVIDIA GPU Hierarchy

```
GPU
 └── GPC (Graphics Processing Cluster) × N
      └── TPC (Texture Processing Cluster) × M
           └── SM (Streaming Multiprocessor) × 2
                └── CUDA Cores × 128 (or more)
                └── Tensor Cores × 4
                └── Shared Memory (128 KB)
                └── L1 Cache (128 KB)
                └── Register File (256 KB)

RTX 4090 (Ada Lovelace):
  16 GPCs × 8 TPCs × 2 SMs = 128 SMs
  128 SMs × 128 CUDA cores = 16,384 CUDA cores
  128 SMs × 4 Tensor Cores = 512 Tensor Cores

A100 (Ampere, data center):
  8 GPCs × 14 SMs = 108 SMs
  108 SMs × 64 CUDA cores = 6,912 CUDA cores
  108 SMs × 4 Tensor Cores = 432 Tensor Cores
```

### Streaming Multiprocessor (SM)

```
The SM is the fundamental compute unit:

┌─────────────────────────────────────────────┐
│              Streaming Multiprocessor (SM)    │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │         Warp Schedulers (×4)           │  │
│  │  Schedule groups of 32 threads (warps) │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌──────────────┐  ┌──────────────────────┐  │
│  │  CUDA Cores  │  │   Tensor Cores (×4)  │  │
│  │    (×128)    │  │   Matrix ops: 4×4    │  │
│  │  FP32, INT32 │  │   FP16, BF16, INT8   │  │
│  └──────────────┘  └──────────────────────┘  │
│                                              │
│  ┌──────────────────────────────────────────┐│
│  │  Shared Memory / L1 Cache (128 KB)       ││
│  │  Fast on-chip memory, shared by threads  ││
│  └──────────────────────────────────────────┘│
│                                              │
│  ┌──────────────────────────────────────────┐│
│  │  Register File (256 KB per SM)           ││
│  │  Fastest memory, per-thread              ││
│  └──────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

---

## Memory Hierarchy

```
Speed & Size Trade-off:

Type          │ Size       │ Bandwidth    │ Latency  │ Access
──────────────┼────────────┼──────────────┼──────────┼─────────
Registers     │ 256 KB/SM  │ ~20 TB/s     │ 0 cycles │ Per thread
Shared Memory │ 128 KB/SM  │ ~15 TB/s     │ ~20 cyc  │ Per SM (block)
L1 Cache      │ 128 KB/SM  │ ~12 TB/s     │ ~30 cyc  │ Per SM
L2 Cache      │ 6-96 MB    │ ~4 TB/s      │ ~200 cyc │ Global
HBM (VRAM)    │ 16-80 GB   │ 1-3.35 TB/s  │ ~400 cyc │ Global (main)
System RAM    │ 64-512 GB  │ ~100 GB/s    │ ~10K cyc │ CPU memory
Disk (SSD)    │ 1-8 TB     │ ~7 GB/s      │ ~100K c  │ Storage

Key insight for ML:
  HBM bandwidth is the BOTTLENECK for most operations.
  
  Matrix multiply: compute-bound (Tensor Cores fast enough)
  Attention softmax: MEMORY-BOUND (reads/writes lots of data)
  Activation functions: MEMORY-BOUND (simple compute, lots of data)
  
  FlashAttention is revolutionary because it moves attention
  from HBM to SRAM (shared memory), reducing memory traffic.
```

### VRAM (HBM) — Where Your Model Lives

```
On an RTX 4090 (24 GB VRAM):
  Gemma-2B fp16:           ~4 GB  (model weights)
  Gemma-2B 4-bit (QLoRA):  ~1.5 GB (model weights)
  LoRA adapters:            ~5 MB  (in fp16)
  Optimizer states:         ~10 MB (8-bit for LoRA)
  Activations/Gradients:    ~2-4 GB (depends on batch size/seq length)
  KV Cache:                 ~0.5-2 GB (during inference)
  CUDA context:             ~0.5 GB (overhead)
  ──────────────────────────────────
  Total (training):         ~6-8 GB of 24 GB  ← fits!
  
On an RTX 3060 (12 GB):
  Same breakdown → ~6-8 GB  ← fits (barely!)
```

---

## Tensor Cores

### What Are Tensor Cores?

```
CUDA Cores: General-purpose floating-point arithmetic.
  One multiply-add per cycle: a × b + c

Tensor Cores: SPECIALIZED matrix multiply hardware.
  One 4×4 matrix multiply per cycle: D = A × B + C
  
  4×4 = 64 multiply-adds in ONE cycle!
  8-16× faster than CUDA cores for matrix operations.
```

### Supported Data Types

```
Generation     │ FP16  │ BF16  │ TF32  │ INT8  │ FP8   │ Peak TFLOPS
───────────────┼───────┼───────┼───────┼───────┼───────┼───────────
Volta (V100)   │ ✅    │ ❌    │ ❌    │ ✅    │ ❌    │ 125
Ampere (A100)  │ ✅    │ ✅    │ ✅    │ ✅    │ ❌    │ 312
Ada (RTX 4090) │ ✅    │ ✅    │ ✅    │ ✅    │ ✅    │ 660
Hopper (H100)  │ ✅    │ ✅    │ ✅    │ ✅    │ ✅    │ 990

Our training uses BF16 compute on top of 4-bit quantized weights.
The Tensor Cores handle the matrix multiplications in BF16.
```

### How 4-bit QLoRA Uses Tensor Cores

```
1. Base weights stored in 4-bit (NF4) in VRAM
2. For each forward pass:
   a. Dequantize a block: 4-bit → BF16 (fast, on-the-fly)
   b. Transfer to Tensor Core registers
   c. Tensor Core multiplies: output = input × weight_bf16
   d. Result in BF16 → accumulate
   e. Discard dequantized weights (only needed briefly)

The dequantization overhead is small compared to the matrix multiply.
Net effect: ~2× slower than pure BF16, but 4× less VRAM!
```

---

## CUDA Programming Model

### Threads, Blocks, Grids

```
CUDA organizes computation hierarchically:

Grid (the entire computation)
└── Block (executed on one SM)
     └── Thread (individual unit of work)

Example: Matrix addition C = A + B for 1024×1024 matrices

Grid:
  ┌────────┬────────┬────────┬────────┐
  │Block(0)│Block(1)│Block(2)│Block(3)│   ← 32 blocks
  ├────────┼────────┼────────┼────────┤      in this example
  │Block(4)│Block(5)│Block(6)│Block(7)│
  ├────────┼────────┼────────┼────────┤
  │  ...   │  ...   │  ...   │  ...   │
  └────────┴────────┴────────┴────────┘

Each block has 256 threads:
  Thread 0: C[0] = A[0] + B[0]
  Thread 1: C[1] = A[1] + B[1]
  ...
  Thread 255: C[255] = A[255] + B[255]

Total threads: 32 blocks × 256 threads = 8192 (one per element?)
Wait — 1024×1024 = 1M elements, so we need more blocks!
Grid: (1024/16) × (1024/16) = 64 × 64 = 4096 blocks
```

### Warps: The Execution Unit

```
Threads within a block are grouped into WARPS of 32 threads.
All 32 threads in a warp execute the SAME instruction simultaneously.

This is SIMT (Single Instruction, Multiple Threads):
  Warp 0, step 1: ALL 32 threads compute a = input[tid]
  Warp 0, step 2: ALL 32 threads compute b = weight[tid]  
  Warp 0, step 3: ALL 32 threads compute c = a * b

If threads DIVERGE (different if/else branches):
  Both branches execute, but threads in the "wrong" branch do nothing.
  This wastes compute → avoid divergent warps!
```

---

## How PyTorch Uses CUDA

```python
import torch

# Move data to GPU
x = torch.randn(1, 2048, device="cuda")     # Allocated in GPU VRAM
W = torch.randn(2048, 2048, device="cuda")   # Also in GPU VRAM

# Matrix multiply → automatically uses CUDA kernels
y = x @ W  # PyTorch dispatches to cuBLAS (NVIDIA's optimized BLAS library)
           # cuBLAS uses Tensor Cores if available and dtype is fp16/bf16

# Reduction → CUDA kernel
loss = y.mean()  # Custom CUDA kernel for reduction

# Backward → CUDA kernels for gradient computation
loss.backward()  # All gradient computations happen on GPU

# When you call torch.compile():
#   PyTorch generates CUSTOM FUSED CUDA kernels via Triton
#   These can be faster than standard cuBLAS for specific patterns
```

---

## GPU Memory Management

### Memory Allocation in PyTorch

```python
import torch

# PyTorch uses a CACHING memory allocator:
# 1. Request GPU memory from CUDA driver (expensive, ~1ms)
# 2. Cache the allocation for reuse
# 3. Future tensors reuse cached memory (free, ~1μs)

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")
#                    ↑ allocated by PyTorch's cache
#                    Some may be unused but reserved for future use

# Force clear cache (rarely needed)
torch.cuda.empty_cache()
# Returns cached memory to CUDA driver
# Does NOT free actively used tensors

# Peak memory (useful for debugging OOM)
print(f"Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Why OOM Happens

```
Scenario: 24 GB GPU

Model weights (4-bit):    1.5 GB
LoRA weights (fp16):      0.005 GB
CUDA context:             0.5 GB
Cached allocations:       1.0 GB
Forward activations:      4.0 GB  ← scales with batch size and seq_len!
Backward gradients:       4.0 GB
Optimizer states:          0.01 GB
────────────────────────────────
Total:                    11.0 GB → fits in 24 GB ✅

But if batch_size = 8 and seq_len = 2048:
  Activations jump to: 8 × 2048 × 2048 × 18 layers × 2 bytes ≈ 12 GB
  Total: ~19 GB → still fits

With batch_size = 16:
  Activations: ~24 GB
  Total: ~31 GB → OOM! ❌
  
  Solution: Reduce batch_size or enable gradient checkpointing
  (recompute activations during backward instead of storing)
```

---

## Benchmarking and Profiling

```python
# Time a CUDA operation
import torch
from torch.cuda import Event

start = Event(enable_timing=True)
end = Event(enable_timing=True)

start.record()
y = model(input_ids)  # Forward pass
end.record()
torch.cuda.synchronize()  # Wait for GPU to finish!
print(f"Forward pass: {start.elapsed_time(end):.1f} ms")

# IMPORTANT: torch.cuda.synchronize() is needed because
# CUDA operations are ASYNCHRONOUS.
# Python returns immediately, but the GPU is still computing.
# elapsed_time between events gives the TRUE GPU time.
```

---

## Choosing a GPU for Fine-Tuning

```
Budget GPU (QLoRA fine-tuning):
  RTX 3060 12GB:  ~$300 → Can fine-tune Gemma-2B ✅
  RTX 4060 8GB:   ~$300 → Tight for 2B, reduce seq_len
  
Mid-range (comfortable QLoRA):
  RTX 3090 24GB:  ~$700 → Gemma-2B easily, 7B possible
  RTX 4090 24GB: ~$1600 → Fast, Tensor Core FP8 support

Data center (larger models):
  A100 40GB:      ~$10K → Gemma-2B, 7B, Llama-13B
  A100 80GB:      ~$15K → Up to 70B with QLoRA
  H100 80GB:      ~$30K → Fastest, FP8 Tensor Cores

Cloud options:
  Google Colab (free): T4 16GB → Gemma-2B with QLoRA ✅
  AWS g5.xlarge: A10G 24GB → ~$1/hour
  Lambda Labs: A100 40GB → ~$1.10/hour
  RunPod: A100 80GB → ~$1.64/hour

Our recommendation: 
  Free? → Google Colab T4
  Own hardware? → RTX 3060/3090
  Need speed? → Rent A100 on cloud
```
