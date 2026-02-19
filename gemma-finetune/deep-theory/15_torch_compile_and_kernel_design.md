# 15. PyTorch Compile Internals & GPU Kernel Design

## Table of Contents
- [The Compilation Pipeline End-to-End](#the-compilation-pipeline-end-to-end)
- [TorchDynamo: Python Bytecode Capture](#torchdynamo-python-bytecode-capture)
- [FX Graph: The Intermediate Representation](#fx-graph-the-intermediate-representation)
- [AOTAutograd: Automatic Differentiation at Compile Time](#aotautograd-automatic-differentiation-at-compile-time)
- [Inductor: The Code Generator](#inductor-the-code-generator)
- [Triton Kernels: How They're Generated](#triton-kernels-how-theyre-generated)
- [Kernel Fusion Deep Dive](#kernel-fusion-deep-dive)
- [Kernel Tiling and Memory Access Patterns](#kernel-tiling-and-memory-access-patterns)
- [Writing Custom Triton Kernels](#writing-custom-triton-kernels)
- [CUDA Graphs: Replay-Based Optimization](#cuda-graphs-replay-based-optimization)
- [Operator-Level Kernel Design](#operator-level-kernel-design)
- [Profiling Compiled Kernels](#profiling-compiled-kernels)
- [The Full Picture: From Python to Silicon](#the-full-picture-from-python-to-silicon)

---

## The Compilation Pipeline End-to-End

### The Stack

```
Your Python code
       │
       ▼
┌──────────────────────┐
│   TorchDynamo        │  Step 1: Capture Python bytecode
│   (Python → FX IR)   │  Intercept at CPython frame level
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   FX Graph           │  Step 2: Intermediate representation
│   (Symbolic trace)   │  Pure computational graph, no Python
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   AOTAutograd        │  Step 3: Trace forward AND backward
│   (Grad computation) │  Create joint graph for both passes
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   FX Passes          │  Step 4: Graph-level optimizations
│   (Fusion, CSE, DCE) │  Combine ops, eliminate redundancies
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   TorchInductor      │  Step 5: Lower to backend IR
│   (Scheduling)       │  Decide which ops to fuse into kernels
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Triton / C++       │  Step 6: Code generation
│   (Kernel codegen)   │  Emit actual GPU kernel source code
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Triton Compiler    │  Step 7: Compile to PTX/CUBIN
│   (LLVM → PTX)       │  Binary code the GPU can execute
└──────────┬───────────┘
           ▼
       GPU silicon
```

### Timing Breakdown

```
For a typical Gemma-2B forward pass compilation:

Step                │ Time (approx) │ Notes
────────────────────┼───────────────┼──────────────────────
TorchDynamo trace   │ 0.5-2s        │ One-time per graph
FX graph creation   │ <0.1s         │ Fast IR construction
AOTAutograd         │ 1-3s          │ Tracing backward pass
FX passes           │ 0.5-1s        │ Fusion, optimization
Inductor scheduling │ 1-3s          │ Kernel planning
Triton codegen      │ 2-5s          │ Writing kernel code
Triton compilation  │ 10-60s        │ LLVM → PTX (slowest!)
────────────────────┼───────────────┼──────────────────────
Total               │ 15-75s        │ ← "warm-up" cost

This cost is AMORTIZED: compile once, execute millions of times.
If your training has 1000 steps, a 60s compilation adds only
0.06s per step on average.
```

---

## TorchDynamo: Python Bytecode Capture

### How Dynamo Intercepts Python

```
Python code goes through several stages before execution:

Source code (.py)
       │
       ▼
Parser → Abstract Syntax Tree (AST)
       │
       ▼
Compiler → Python Bytecode (.pyc)
       │
       ▼───────── TorchDynamo INTERCEPTS HERE
       │
       ▼
CPython VM → Execution

TorchDynamo uses CPython's PEP 523 frame evaluation API:
  - It registers a custom frame evaluation function
  - EVERY Python function frame passes through TorchDynamo
  - Dynamo examines the bytecodes and REWRITES them
  - PyTorch operations → captured into FX graph
  - Non-PyTorch operations → fall back to Python (graph break)
```

### Bytecode Example

```python
# Original Python function:
def forward(x, W, b):
    y = x @ W
    y = y + b
    return torch.relu(y)

# Python bytecodes (simplified):
LOAD_FAST    x
LOAD_FAST    W
BINARY_MATMUL          # → TorchDynamo captures: matmul(x, W)
STORE_FAST   y
LOAD_FAST    y
LOAD_FAST    b
BINARY_ADD             # → TorchDynamo captures: add(y, b)
STORE_FAST   y
LOAD_GLOBAL  torch
LOAD_ATTR    relu
LOAD_FAST    y
CALL_FUNCTION 1        # → TorchDynamo captures: relu(y)
RETURN_VALUE

# TorchDynamo sees these bytecodes and builds:
#   FX Node: matmul(x, W) → y₁
#   FX Node: add(y₁, b) → y₂
#   FX Node: relu(y₂) → output
```

### Guard System

```
After tracing, Dynamo installs GUARDS to detect when recompilation
is needed:

Guard examples:
  - type(x) == torch.Tensor         (input must be a tensor)
  - x.shape == (1, 512, 2048)       (shape must match)
  - x.dtype == torch.float16        (dtype must match)
  - x.device == cuda:0              (device must match)
  - x.requires_grad == True         (gradient tracking matches)

If any guard FAILS on a new call:
  → Dynamo re-traces with the new inputs
  → Generates a new specialized kernel
  → Caches both versions

This is why FIXED shapes are important:
  Variable shapes → frequent guard failures → frequent recompilations
  Fixed shapes → guards always pass → use cached compiled code
```

---

## FX Graph: The Intermediate Representation

### Graph Structure

```python
# For a LoRA-adapted linear layer, the FX graph looks like:

import torch.fx

# Conceptual FX graph:
graph():
    # Inputs
    %x     : Tensor[1, 512, 2048] = placeholder
    %W_base: Tensor[2048, 2048]   = get_attr(model.q_proj.weight)
    %lora_A: Tensor[2048, 16]     = get_attr(model.q_proj.lora_A)
    %lora_B: Tensor[16, 2048]     = get_attr(model.q_proj.lora_B)
    %alpha : float = 32.0
    %r     : float = 16.0
    
    # Base computation
    %base_out: Tensor = call_function(torch.nn.functional.linear, %x, %W_base)
    
    # LoRA branch
    %lora_down: Tensor = call_function(torch.matmul, %x, %lora_A)
    %lora_up  : Tensor = call_function(torch.matmul, %lora_down, %lora_B)
    %scale    : float  = call_function(operator.truediv, %alpha, %r)
    %lora_out : Tensor = call_function(operator.mul, %lora_up, %scale)
    
    # Combined output
    %output   : Tensor = call_function(operator.add, %base_out, %lora_out)
    
    return %output
```

### Graph-Level Optimizations (FX Passes)

```
COMMON SUBEXPRESSION ELIMINATION (CSE):
  Before:
    a = x @ W
    b = x @ W       ← same computation!
    c = a + b
  
  After:
    a = x @ W
    c = a + a        ← eliminated redundant compute

DEAD CODE ELIMINATION (DCE):
  Before:
    a = x @ W
    b = torch.relu(a)
    c = a + 1         ← 'c' is never used!
    return b
  
  After:
    a = x @ W
    return torch.relu(a)   ← removed 'c', inlined 'b'

CONSTANT FOLDING:
  Before:
    scale = alpha / r       ← both are constants: 32.0 / 16.0
    out = lora_up * scale
  
  After:
    out = lora_up * 2.0     ← computed at compile time

OPERATOR FUSION (the big one — handled by Inductor):
  Before: 3 separate kernel launches
    y = x @ W              kernel 1: matmul
    y = y + bias            kernel 2: add
    y = torch.relu(y)       kernel 3: relu
  
  After: 1 fused kernel
    y = fused_matmul_bias_relu(x, W, bias)  ← single launch!
```

---

## AOTAutograd: Automatic Differentiation at Compile Time

### Why Compile the Backward Pass?

```
Standard PyTorch (eager mode):
  Forward: Record operations on tape (autograd graph)
  Backward: Replay tape, compute gradients on-the-fly
  
  Problem: The backward pass is ALSO just a sequence of operations
  that could benefit from compilation and fusion!

AOTAutograd:
  1. Trace the FORWARD pass → FX graph of forward ops
  2. Use autograd rules to DERIVE the backward graph
  3. Now we have BOTH forward and backward as FX graphs
  4. Optimize BOTH graphs with Inductor
  5. Compiled forward stores minimal intermediates
  6. Compiled backward is also fused and optimized
```

### The Joint Graph

```
               FORWARD GRAPH
  Input ──→ [matmul] ──→ [add] ──→ [relu] ──→ Output
               │           │          │
               │ save for backward     │
               ▼           ▼          ▼
               BACKWARD GRAPH (derived by AOTAutograd)
  d_Output ──→ [relu_bwd] ──→ [add_bwd] ──→ [matmul_bwd] ──→ d_Input
                    │              │              │
                    ▼              ▼              ▼
                 d_nothing      d_bias         d_W, d_x

Savings from compilation:
  - Backward kernels are fused too (relu_bwd + add_bwd → one kernel)
  - Only truly needed intermediates are saved (not all of them)
  - Memory planning: allocate exact buffer sizes
```

---

## Inductor: The Code Generator

### Scheduling: Which Ops to Fuse

```
Inductor's scheduler decides HOW to group operations into kernels:

POINTWISE FUSION (element-wise ops → one kernel):
  add, mul, relu, sigmoid, tanh, GELU, etc.
  These read/write each element independently → fuse freely.
  
  Before: 5 kernel launches
    y = x * W
    y = y + bias
    y = torch.relu(y)
    y = y * 0.5
    y = y + 1.0
  
  After: 1 kernel launch
    y = fused_mul_add_relu_mul_add(x, W, bias, 0.5, 1.0)
    
  Each element: output[i] = relu(x[i]*W[i] + bias[i]) * 0.5 + 1.0

REDUCTION FUSION (ops that aggregate across dimensions):
  sum, mean, softmax, layer_norm
  These need to read MANY elements to produce one output.
  
  Inductor fuses reductions with adjacent pointwise ops:
    mean(relu(x + bias))  →  one kernel that computes relu and mean together

MATMUL EPILOGUE FUSION:
  The big matmul itself uses cuBLAS/cuDNN (highly optimized).
  But ops AFTER matmul (bias add, activation) are fused:
    y = matmul(x, W)     ← cuBLAS kernel
    z = relu(y + bias)    ← Triton pointwise kernel (fused)
  
  In some cases, matmul + bias + activation are ALL fused into
  one cuBLAS call (epilogue fusion).
```

### Inductor's Output: Triton Code

```python
# What Inductor ACTUALLY generates for ReLU + scale + add:
# (This is real generated code, simplified)

@triton.jit
def triton_poi_kernel(
    in_ptr0,      # input tensor
    in_ptr1,      # bias tensor
    out_ptr0,     # output tensor
    xnumel,       # total number of elements
    XBLOCK: tl.constexpr = 1024,  # block size
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    # Load input and bias
    x0 = tl.load(in_ptr0 + xindex, xmask)
    x1 = tl.load(in_ptr1 + (xindex % 2048), xmask)  # broadcast bias
    
    # Fused computation: relu(x + bias) * 0.5
    tmp0 = x0 + x1           # add bias
    tmp1 = tl.maximum(tmp0, 0)  # relu
    tmp2 = tmp1 * 0.5        # scale
    
    # Store output
    tl.store(out_ptr0 + xindex, tmp2, xmask)

# This ONE kernel replaces THREE separate CUDA kernel launches!
# Memory traffic: read input+bias ONCE, write output ONCE
# Without fusion: read/write intermediates = 3× more memory traffic
```

---

## Triton Kernels: How They're Generated

### The Triton Programming Model

```
Triton is a BLOCK-LEVEL programming model:

CUDA: You program individual THREADS
  thread_id = blockIdx.x * blockDim.x + threadIdx.x
  output[thread_id] = relu(input[thread_id] + bias[thread_id])

Triton: You program BLOCKS of data
  block_start = tl.program_id(0) * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  x = tl.load(input_ptr + offsets)
  b = tl.load(bias_ptr + offsets)
  output = tl.maximum(x + b, 0)  # Operates on entire block at once!
  tl.store(output_ptr + offsets, output)

Why block-level is easier:
  CUDA: Manually manage threads, warps, shared memory, synchronization
  Triton: Just say "load this block, compute, store" — compiler handles rest
```

### Triton Compilation Pipeline

```
Triton Python code (@triton.jit)
       │
       ▼
┌──────────────────────┐
│  Triton Frontend     │  Parse Python → Triton AST
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Triton IR           │  Block-level operations
│  (MLIR-based)        │  tl.load, tl.store, tl.dot, etc.
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Optimization Passes │  Memory coalescing
│                      │  Shared memory allocation
│                      │  Instruction scheduling
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  LLVM IR             │  Standard compiler IR
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  PTX Assembly        │  NVIDIA GPU assembly language
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  CUBIN               │  Binary executable for the GPU
└──────────────────────┘
```

### Auto-Tuning

```
Triton's compiler auto-tunes kernel parameters:

Tunable parameters:
  BLOCK_SIZE:         128, 256, 512, 1024, 2048
  num_warps:          2, 4, 8
  num_stages:         1, 2, 3, 4 (pipelining depth)
  
For each combination:
  1. Compile the kernel
  2. Run it 10 times on sample data
  3. Measure average execution time
  4. Pick the fastest configuration

Example auto-tune results for our fused kernel:
  BLOCK=128,  warps=2, stages=1:  0.42 ms
  BLOCK=256,  warps=4, stages=2:  0.31 ms  
  BLOCK=512,  warps=4, stages=2:  0.28 ms  ← FASTEST
  BLOCK=1024, warps=8, stages=3:  0.30 ms
  BLOCK=2048, warps=8, stages=4:  0.35 ms

Selected: BLOCK=512, warps=4, stages=2

This tuning happens during compilation (adds to warm-up time)
but the selected config is cached for future runs.
```

---

## Kernel Fusion Deep Dive

### Why Fusion Matters: Memory Bandwidth

```
GPU computation is often MEMORY-BOUND, not compute-bound:

RTX 4090 specs:
  Compute: 82.6 TFLOPS (FP32)
  Memory bandwidth: 1 TB/s

For a simple element-wise operation (ReLU on 2048 elements):
  Compute: 2048 comparisons → ~0 time (trivial)
  Memory: Read 2048 × 4 bytes + Write 2048 × 4 bytes = 16 KB
  At 1 TB/s: 16 KB / 1 TB/s = 0.016 μs

  The kernel spends 99%+ of its time WAITING for memory!

Without fusion (3 separate operations):
  Op 1 (add):   Read x, Read b → Write y₁     = 3 × 16KB = 48 KB
  Op 2 (relu):  Read y₁        → Write y₂     = 2 × 16KB = 32 KB
  Op 3 (scale): Read y₂        → Write y₃     = 2 × 16KB = 32 KB
  Total memory traffic: 112 KB
  Plus: 3 kernel launch overheads (~5 μs each)

With fusion (1 kernel):
  Read x, Read b → compute add+relu+scale → Write y₃ = 3 × 16KB = 48 KB
  Total memory traffic: 48 KB  (57% less!)
  Plus: 1 kernel launch overhead (~5 μs)

Speedup from fusion: ~2.3× for this example
For larger tensors, the speedup is even greater.
```

### Fusion Categories

```
1. HORIZONTAL FUSION (fuse parallel independent ops)
   Before:
     a = relu(x)         # kernel 1
     b = sigmoid(y)      # kernel 2 (independent of kernel 1)
   After:
     a, b = fused_relu_sigmoid(x, y)  # 1 kernel, processes both

2. VERTICAL FUSION (fuse sequential dependent ops)  ← Most common
   Before:
     y = x + bias        # kernel 1
     z = relu(y)          # kernel 2 (depends on kernel 1)
   After:
     z = fused_add_relu(x, bias)  # 1 kernel

3. REDUCTION FUSION (fuse reduction with pointwise)
   Before:
     y = x * scale        # kernel 1 (pointwise)
     z = y.sum(dim=-1)    # kernel 2 (reduction)
   After:
     z = fused_mul_sum(x, scale, dim=-1)  # 1 kernel

4. MATMUL EPILOGUE FUSION (fuse ops after matmul)
   Before:
     y = x @ W            # kernel 1 (cuBLAS GEMM)
     y = y + bias          # kernel 2
     y = gelu(y)           # kernel 3
   After:
     y = fused_gemm_bias_gelu(x, W, bias)  # cuBLAS with fused epilogue
```

---

## Kernel Tiling and Memory Access Patterns

### Why Tiling?

```
GPU memory hierarchy (from guide #12):
  SRAM (shared memory): 128 KB, ~20 TB/s bandwidth
  HBM (VRAM):          24 GB, ~1 TB/s bandwidth

Ratio: SRAM is 20× faster but 200,000× smaller!

Tiling strategy:
  1. Load a TILE of data from HBM into SRAM
  2. Compute on the tile (many operations, staying in fast SRAM)
  3. Store results back to HBM
  4. Repeat for next tile

Without tiling:
  Every operation reads/writes HBM → 1 TB/s

With tiling:
  Data stays in SRAM for multiple operations → 20 TB/s
  Only load/store to HBM once per tile
```

### Matrix Multiply Tiling

```
Computing C = A × B where A is (M,K), B is (K,N):

Naive (no tiling):
  For each output element C[i,j]:
    Load row A[i,:] from HBM    (K reads)
    Load col B[:,j] from HBM    (K reads)
    Compute dot product          (K multiply-adds)
    Store C[i,j] to HBM         (1 write)
  Total reads: M×N×2K from HBM → VERY slow

Tiled:
  Split A into (M/BM) × (K/BK) tiles
  Split B into (K/BK) × (N/BN) tiles
  
  For each tile pair:
    Load A_tile (BM×BK) into SRAM         (BM×BK reads)
    Load B_tile (BK×BN) into SRAM         (BK×BN reads)
    Compute C_tile += A_tile × B_tile     (BM×BN×BK ops, all in SRAM!)
  
  Total HBM reads: (M×K + K×N) / reuse_factor
  Much less memory traffic!

Triton auto-tunes the tile sizes (BM, BK, BN) for your GPU.
```

### Memory Coalescing

```
GPU memory is accessed in 128-byte transactions.
If threads in a warp access ADJACENT memory → 1 transaction (coalesced)
If threads access SCATTERED memory → up to 32 transactions (uncoalesced)

Coalesced (good):
  Thread 0 reads mem[0]
  Thread 1 reads mem[1]
  Thread 2 reads mem[2]
  ...
  Thread 31 reads mem[31]
  → 1 memory transaction (128 bytes = 32 × 4 bytes)

Uncoalesced (bad):
  Thread 0 reads mem[0]
  Thread 1 reads mem[1000]
  Thread 2 reads mem[2000]
  ...
  → 32 separate transactions (32× slower!)

Triton handles coalescing automatically through its block-based model:
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  data = tl.load(ptr + offsets)   # Contiguous → coalesced!
```

---

## Writing Custom Triton Kernels

### Example: Fused LoRA Forward

```python
import triton
import triton.language as tl
import torch

@triton.jit
def fused_lora_forward_kernel(
    # Pointers to tensors
    base_out_ptr,    # Output of base linear layer
    x_ptr,           # Input tensor
    lora_A_ptr,      # LoRA A matrix (d × r)
    lora_B_ptr,      # LoRA B matrix (r × d)
    output_ptr,      # Combined output
    # Dimensions
    M,               # Batch × sequence length
    N,               # Output dimension (2048)
    R,               # LoRA rank (16)
    # Scaling
    scaling: tl.constexpr,    # alpha / r
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_R: tl.constexpr = 16,
):
    """
    Fused kernel: output = base_out + (x @ lora_A @ lora_B) * scaling
    
    Instead of:
      1. kernel: lora_down = x @ lora_A        (matmul)
      2. kernel: lora_up = lora_down @ lora_B   (matmul)
      3. kernel: scaled = lora_up * scaling     (pointwise)
      4. kernel: output = base_out + scaled     (pointwise)
    
    We fuse steps 3 and 4 into the epilogue of step 2.
    Steps 1 and 2 could also be fused but matmul fusion is complex.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output block indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load base output (from cuBLAS matmul)
    base_ptrs = base_out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_out = tl.load(base_ptrs, mask=mask, other=0.0)
    
    # Compute LoRA: accumulate x @ lora_A @ lora_B in tiles
    # This is a simplified version; real implementation tiles over R
    lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, R, BLOCK_R):
        offs_r = k + tl.arange(0, BLOCK_R)
        
        # Load x @ lora_A result (precomputed, shape M×R)
        # In practice, you'd compute this inline or pass it in
        lora_down_ptrs = x_ptr + offs_m[:, None] * R + offs_r[None, :]
        lora_down = tl.load(lora_down_ptrs, mask=(offs_m[:, None] < M) & (offs_r[None, :] < R))
        
        # Load lora_B tile (R×N)
        lora_B_ptrs = lora_B_ptr + offs_r[:, None] * N + offs_n[None, :]
        lora_B_tile = tl.load(lora_B_ptrs, mask=(offs_r[:, None] < R) & (offs_n[None, :] < N))
        
        # Accumulate partial matmul
        lora_acc += tl.dot(lora_down, lora_B_tile)
    
    # FUSED: scale + add in one step (no extra memory traffic!)
    output = base_out + lora_acc * scaling
    
    # Store
    out_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, output, mask=mask)


def fused_lora_forward(base_out, x_projected, lora_B, scaling):
    """Python wrapper for the Triton kernel."""
    M, N = base_out.shape
    _, R = x_projected.shape
    output = torch.empty_like(base_out)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    fused_lora_forward_kernel[grid](
        base_out, x_projected, None, lora_B, output,
        M, N, R, scaling,
    )
    return output
```

### Example: Fused RMSNorm + Residual

```python
@triton.jit
def fused_rmsnorm_residual_kernel(
    x_ptr,          # Input tensor
    residual_ptr,   # Residual connection
    weight_ptr,     # RMSNorm weight
    output_ptr,     # Output
    N: tl.constexpr,  # Hidden dimension (2048)
    eps: tl.constexpr = 1e-6,
    BLOCK_N: tl.constexpr = 2048,
):
    """
    Fused: output = RMSNorm(x + residual) × weight
    
    Without fusion (3 kernels):
      1. y = x + residual           (pointwise, read x + residual, write y)
      2. rms = sqrt(mean(y²))       (reduction, read y, write rms)
      3. output = y / rms * weight  (pointwise, read y + rms + weight, write output)
    
    With fusion (1 kernel):
      Read x, residual, weight ONCE → write output ONCE
      rms computed in registers (no HBM traffic!)
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    
    # Load input and residual (FUSED: add them immediately)
    x = tl.load(x_ptr + row_idx * N + col_offsets, mask=mask)
    res = tl.load(residual_ptr + row_idx * N + col_offsets, mask=mask)
    hidden = x + res  # Add in registers, no HBM write!
    
    # Compute RMS (FUSED: reduction in registers)
    sq = hidden * hidden
    mean_sq = tl.sum(sq, axis=0) / N
    rms = tl.sqrt(mean_sq + eps)  # Computed entirely in fast memory!
    
    # Normalize and scale (FUSED: no separate kernel)
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = (hidden / rms) * weight
    
    # Single write to HBM
    tl.store(output_ptr + row_idx * N + col_offsets, output, mask=mask)

# Memory traffic comparison:
#   Without fusion: 5 reads + 3 writes = 8 × N × sizeof(float16)
#   With fusion:    3 reads + 1 write  = 4 × N × sizeof(float16)
#   Speedup: ~2× from memory traffic reduction alone!
```

---

## CUDA Graphs: Replay-Based Optimization

### What CUDA Graphs Do

```
Normal execution:
  CPU ──→ launch kernel 1 ──→ launch kernel 2 ──→ launch kernel 3
  GPU ──→ wait ──→ run kernel 1 ──→ wait ──→ run kernel 2 ──→ ...
  
  Each kernel launch has ~5-10 μs overhead on CPU.
  For 100 kernels: 0.5-1 ms of JUST launch overhead!

CUDA Graph:
  RECORD PHASE (once):
    CPU: Record sequence: kernel 1, kernel 2, kernel 3
    GPU: Execute normally (recording what happens)
  
  REPLAY PHASE (every step):
    CPU: Launch ENTIRE graph in 1 call (~5 μs total!)
    GPU: Replay all kernels back-to-back without CPU involvement
  
  For 100 kernels: 5 μs instead of 500-1000 μs = 100-200× less overhead!
```

### When CUDA Graphs Help Most

```
CUDA graphs help when:
  ✅ Model has MANY small kernels (common in transformers)
  ✅ Kernels are fast (launch overhead > compute time)
  ✅ Same sequence of operations every step (static graph)

CUDA graphs DON'T help when:
  ❌ Few large kernels (matmul dominates, launch overhead negligible)
  ❌ Dynamic operations (different graph each step)
  ❌ Variable input shapes (graph must be re-recorded)

For torch.compile mode="reduce-overhead":
  Uses CUDA graphs automatically!
  Captures the compiled kernel sequence → replays every step
```

### CUDA Graphs with torch.compile

```python
# torch.compile with CUDA graphs:
model = torch.compile(model, mode="reduce-overhead")
# Internally:
#   1. First call: compile kernels → record CUDA graph
#   2. Subsequent calls: replay the CUDA graph

# Manual CUDA graphs (for inference):
with torch.cuda.graph(cuda_graph):
    output = model(sample_input)

# Replay:
for batch in dataloader:
    # Copy new input to the graph's input buffer
    sample_input.copy_(batch)
    cuda_graph.replay()  # ~5 μs!
    # output is updated in-place
```

---

## Operator-Level Kernel Design

### How Key Transformer Operations Map to Kernels

```
TRANSFORMER BLOCK KERNEL BREAKDOWN:

┌──────────────────────────────────────────────┐
│ Input Embedding Lookup                        │
│   Kernel: Simple index select                 │
│   Type: Memory-bound (just reading rows)      │
│   Optimization: None needed (trivial)         │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ RMSNorm                                       │
│   Kernels: 1 fused (reduction + normalize)    │
│   Type: Memory-bound                          │
│   Optimization: Fuse with residual add        │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Q/K/V Projections (with LoRA)                 │
│   Kernels:                                    │
│     1. cuBLAS GEMM (x @ W_base)  ← compute   │
│     2. Triton (x @ lora_A)        ← memory    │
│     3. Triton (down @ lora_B * s) ← memory    │
│     4. Triton (base + lora)       ← memory    │
│   Optimization: Fuse 3+4 into one kernel      │
│   torch.compile: Auto-fuses pointwise epilogue│
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Attention (Q @ Kᵀ / √d → softmax → @ V)     │
│   Kernel: FlashAttention (1 fused kernel!)    │
│   Type: Memory-bound → compute-bound (fused)  │
│   Optimization: FlashAttention does tiling    │
│   [See Attention guide for details]           │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Output Projection (with LoRA)                 │
│   Same structure as Q/K/V projections         │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Residual Add + RMSNorm                        │
│   Kernel: 1 fused (add + norm)                │
│   Type: Memory-bound                          │
│   Optimization: Already fused by Inductor     │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ MLP: GeGLU (gate_proj, up_proj, GELU, mul,   │
│      down_proj, optional LoRA on each)        │
│   Kernels:                                    │
│     1. cuBLAS GEMM (gate projection)          │
│     2. cuBLAS GEMM (up projection)            │
│     3. Triton fused (GELU + gate * up)        │
│     4. cuBLAS GEMM (down projection)          │
│   Optimization: Fuse GELU + elementwise mul   │
│   torch.compile: Handles fusion of step 3     │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Residual Add                                  │
│   Kernel: 1 pointwise (can fuse with next     │
│   RMSNorm in the next layer!)                 │
└──────────────────────────────────────────────┘

Total kernels per layer WITHOUT torch.compile: ~15-20
Total kernels per layer WITH torch.compile:    ~8-10 (fusion!)
For 18 layers: 144-180 kernels → CUDA graphs very beneficial
```

### Kernel Roofline Analysis

```
The ROOFLINE MODEL tells you whether a kernel is compute-bound
or memory-bound:

Arithmetic Intensity (AI) = FLOPs / Bytes transferred

  AI < crossover → memory-bound (optimize memory access)
  AI > crossover → compute-bound (optimize compute)

  Crossover = Peak FLOPS / Peak Bandwidth

RTX 4090 crossover:
  82.6 TFLOPS / 1 TB/s = 82.6 FLOP/byte

Kernel                │ AI (FLOP/byte) │ Bottleneck
──────────────────────┼────────────────┼───────────
ReLU/GELU (pointwise) │ 1              │ Memory (82× below)
Softmax               │ 5              │ Memory (16× below)
RMSNorm               │ 4              │ Memory (20× below)
Attention (unfused)   │ 10             │ Memory (8× below)
Attention (Flash)     │ 80+            │ BALANCED
Matrix multiply       │ 2048           │ Compute (25× above)
───────────────────────────────────────────────────

KEY INSIGHT: Most transformer ops are MEMORY-BOUND!
This is why kernel FUSION helps so much:
  Fewer memory reads/writes → closer to compute-bound → faster
```

---

## Profiling Compiled Kernels

### Using torch.profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.compile(model)

# Warmup (compilation)
for _ in range(3):
    output = model(**batch)

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("forward_pass"):
        output = model(**batch)
    with record_function("backward_pass"):
        output.loss.backward()

# Print results
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20))

# Expected output:
# Name                    CPU Time  CUDA Time  Calls  Shapes
# ─────────────────────────────────────────────────────────
# aten::mm                 2.1ms     1.8ms      36    [1,512,2048]×[2048,2048]
# triton_poi_fused_...     0.3ms     0.2ms      18    [1,512,2048]
# aten::_flash_attention    0.8ms     0.6ms      18    [1,8,512,256]
# triton_red_fused_...     0.2ms     0.15ms     18    [1,512,2048]
# ...
```

### Viewing Generated Triton Code

```python
# See what torch.compile actually generates:
import torch._inductor.config
torch._inductor.config.debug = True

# After compilation, kernels are saved to:
# /tmp/torchinductor_<user>/

# You can also use:
torch._dynamo.config.log_level = logging.DEBUG
# This prints the Triton kernel source code

# Or explicitly:
compiled_model = torch.compile(model)
output = compiled_model(sample_input)

# Check compilation cache:
import torch._inductor.codecache
print(f"Cache dir: {torch._inductor.codecache.cache_dir()}")
# Contains .py files with generated Triton kernels!
```

### NVIDIA Nsight Systems (Advanced)

```bash
# The gold standard for GPU profiling:
nsys profile -w true -t cuda,nvtx -o profile_output \
    python train.py --max_steps 10

# Opens in Nsight Systems GUI:
# - Shows kernel-level timeline
# - Identifies memory copies
# - Shows kernel overlap (or lack thereof)
# - Reveals CPU-GPU synchronization points

# Key things to look for:
# 1. Gaps between kernels → CPU overhead, CUDA graphs can help
# 2. Small kernels → fusion opportunity
# 3. Memory copies → unnecessary .cpu() calls
# 4. Low GPU utilization → batch size too small or CPU bottleneck
```

---

## The Full Picture: From Python to Silicon

```
YOUR CODE                        TIME SCALE
─────────                        ──────────
model = torch.compile(model)     ← ~60 seconds (one-time)
                                   
WHAT HAPPENS:                    
                                   
1. You call model(input)         ← microseconds
   │
2. TorchDynamo intercepts        ← milliseconds
   Python bytecodes               
   │
3. FX Graph created               ← milliseconds
   Pure computation IR             
   │
4. AOTAutograd traces             ← seconds
   forward + backward             
   │
5. Graph optimizations            ← seconds
   CSE, DCE, fusion planning      
   │
6. Inductor scheduling            ← seconds
   Groups ops into kernels         
   │
7. Triton code generation         ← seconds
   Writes kernel source code       
   │
8. Triton compilation             ← 10-60 seconds
   LLVM → PTX → CUBIN             
   │
9. CUBIN loaded onto GPU          ← milliseconds
   Cached for reuse               
   │
                                   
EVERY SUBSEQUENT CALL:           
                                   
1. Guard check (shapes match?)   ← microseconds
   │
2. Launch compiled kernels       ← microseconds per kernel
   │
3. GPU executes fused kernels    ← milliseconds
   In parallel across SMs          
   Using Tensor Cores for matmul   
   Optimized memory access         
   │
4. Results in GPU memory         ← microseconds

Total compiled step: 50-200ms (vs 100-400ms eager)
  Speedup: 1.3-2.0× per training step
  Over 1000 steps: saves 50-200 seconds of compute
  ROI: compilation cost recovered in ~100-300 steps
```
