# Day 60: GPU Optimization & CUDA Kernels
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Writing a Custom Kernel in Triton (Fused Softmax)

Writing raw CUDA is hard. Triton makes it accessible. Let's implement a fused Softmax kernel that is memory efficient.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # The rows of the softmax are independent, so we parallelize across rows
    row_idx = tl.program_id(0)
    
    # The pointer to the start of the row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # The block size is the next power of two greater than n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row into SRAM, masking out-of-bounds
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Compute max for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    
    # Compute numerator (exp)
    numerator = tl.exp(row_minus_max)
    
    # Compute denominator (sum)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Write back to HBM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    # Block size must be power of 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output buffer
    y = torch.empty_like(x)
    
    # Enqueue kernel
    # Grid size = number of rows
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

# Benchmarking
x = torch.randn(1823, 781, device='cuda')
y_triton = triton_softmax(x)
y_torch = torch.softmax(x, dim=1)
assert torch.allclose(y_triton, y_torch), "Triton implementation incorrect!"
```

### 2. FlashAttention Logic (Simplified)

The core idea of FlashAttention is tiling. We load blocks of Q, K, V into SRAM, compute attention, and accumulate results without writing the full $N \times N$ matrix to HBM.

**Algorithm:**
1. Divide Q, K, V into blocks $B_r$ (rows) and $B_c$ (cols).
2. Initialize output $O$ and normalization statistics $L, M$ in HBM.
3. Loop over blocks of Q (outer loop):
    a. Load block $Q_i$ into SRAM.
    b. Load block $O_i, L_i, M_i$ into SRAM.
    c. Loop over blocks of K, V (inner loop):
        i. Load $K_j, V_j$ into SRAM.
        ii. Compute $S_{ij} = Q_i K_j^T$.
        iii. Compute $P_{ij} = \text{exp}(S_{ij} - m_{ij})$.
        iv. Update running stats (online softmax).
        v. Update $O_i$ with $P_{ij} V_j$.
    d. Write $O_i$ back to HBM.

**Why it's faster:**
- **HBM Access:** $O(N^2)$ -> $O(N)$.
- **Compute:** Same FLOPs, but much higher utilization because we are not waiting for memory.

### 3. PyTorch Profiler & Nsight Systems

How to profile your model to find bottlenecks.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace.json")

model = MyModel().cuda()
inputs = torch.randn(128, 1024).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=trace_handler
) as p:
    for i in range(10):
        with record_function("model_inference"):
            model(inputs)
        p.step()
```

**Analysis:**
- Look for **"Gaps"** in the GPU timeline. Gaps mean the GPU is idle, waiting for CPU or memory.
- Look for **"Small Kernels"**. Many small kernels (e.g., 2us) have high launch overhead. Fuse them!
- Look for **"Low Occupancy"**. Kernels not using all SMs.

### 4. CUDA Graphs Implementation

Reduce CPU overhead by capturing the graph.

```python
# 1. Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(input)

# 2. Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)

# 3. Replay
g.replay()
```

**Constraints:**
- Input shapes must be static (or bucketed).
- Control flow (if/else) inside the graph must be static or handled within kernels.

### 5. Memory Coalescing Example

**Bad Access Pattern:**
```cpp
// Thread i accesses data[i * stride]
// If stride is large, each thread accesses a different cache line.
// 32 threads might require 32 memory transactions.
int idx = threadIdx.x * stride;
val = data[idx];
```

**Good Access Pattern (Coalesced):**
```cpp
// Thread i accesses data[i]
// 32 threads access contiguous memory.
// Can be served by 1 or 2 memory transactions.
int idx = threadIdx.x;
val = data[idx];
```

**Impact:** 10x difference in effective bandwidth.

### 6. Tensor Core Usage (PyTorch)

To ensure Tensor Cores are used:
1.  **Precision:** Use `torch.float16` or `torch.bfloat16`.
2.  **Dimensions:** Matrix dimensions (M, N, K) should be multiples of 8 (or 16 for best performance).
3.  **Math Mode:**
    ```python
    torch.set_float32_matmul_precision('high') # or 'medium'
    ```

### 7. Roofline Model

**Concept:** Visualizing performance limits.
- **X-axis:** Arithmetic Intensity (FLOPs / Byte).
- **Y-axis:** Performance (GFLOPS).

**Analysis:**
- **Memory Bound:** Slanted part of the roof. Performance increases with intensity.
- **Compute Bound:** Flat part of the roof. Performance limited by peak GFLOPS.
- **LLM Decoding:** Very low arithmetic intensity (1 token in, 1 token out, huge weights). Deeply memory bound.
- **LLM Prefill:** High arithmetic intensity (batch size * seq len). Can be compute bound.

### Summary of Optimization Strategy

1.  **Identify Bottleneck:** Is it Compute or Memory? (Use Profiler/Roofline).
2.  **Memory Bound?**
    - Fuse kernels (Triton).
    - Use FlashAttention.
    - Quantize (reduce bytes moved).
3.  **Compute Bound?**
    - Use Tensor Cores.
    - Check matrix dimensions (padding).
4.  **Overhead Bound?**
    - Use CUDA Graphs.
    - Increase batch size.
