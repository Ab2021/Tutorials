# Day 18: Model Parallelism
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Tensor Parallelism (Megatron-LM Style)

**The Concept:**
Parallelize matrix multiplications across GPUs without changing the mathematical result.
Focus on the **Feed-Forward Network (FFN)** and **Self-Attention**.

**Column Parallelism:**
Split the weight matrix $W$ by columns.
$Y = X W = X [W_1, W_2] = [X W_1, X W_2]$
- Each GPU gets a copy of input $X$.
- Each GPU computes a slice of output $Y$.
- **Output:** Partial results on each GPU.

**Row Parallelism:**
Split the weight matrix $W$ by rows.
$Y = X W = [X_1, X_2] \begin{pmatrix} W_1 \\ W_2 \end{pmatrix} = X_1 W_1 + X_2 W_2$
- Input $X$ is split across GPUs.
- Each GPU computes a partial sum.
- **Output:** Need to sum (All-Reduce) results to get $Y$.

**The Megatron-LM Trick:**
Combine Column and Row parallelism to minimize communication.
- **Layer 1 (MLP expansion):** Column Parallel. Output is split (no sync needed).
- **Layer 2 (MLP projection):** Row Parallel. Input is already split. Output needs All-Reduce.
- **Result:** Only ONE All-Reduce needed per MLP block (instead of 2).

### 2. Communication Primitives

**All-Reduce:**
- Every GPU starts with a tensor $T_i$.
- Ends with every GPU having $\sum T_i$.
- **Cost:** $2 \cdot (N-1) \cdot \text{Size} / \text{Bandwidth}$.

**All-Gather:**
- Every GPU starts with $T_i$.
- Ends with every GPU having $[T_1, T_2, \dots, T_N]$.

**Reduce-Scatter:**
- Every GPU starts with $[T_{i,1}, \dots, T_{i,N}]$.
- Ends with GPU $j$ having $\sum_i T_{i,j}$.

### 3. Pipeline Parallelism Schedules

**Naive Pipeline:**
- Batch 1: GPU1 -> GPU2 -> GPU3 -> GPU4.
- GPU4 finishes, sends gradients back.
- **Utilization:** Terrible. Most GPUs idle most of the time.

**GPipe (Micro-batching):**
- Split batch into $M$ micro-batches.
- Inject all $M$ into pipeline.
- **Flush:** Wait for all to finish before updating weights.
- **Bubble:** Still exists at start and end.

**1F1B (One Forward, One Backward):**
- Schedule: Forward 1, Forward 2, Backward 1, Forward 3, Backward 2...
- Interleaves forward and backward passes.
- **Memory Benefit:** Frees up activation memory (from Forward 1) as soon as Backward 1 is done.
- **Utilization:** Much higher than GPipe.

### 4. Sequence Parallelism (Ring Attention)

**Problem:**
Standard TP splits the *hidden dimension*.
What if sequence length is 1 Million? The activation memory ($B \times L \times H$) explodes on every GPU.

**Solution:**
Split the **Sequence Dimension** ($L$) across GPUs.
- GPU 1 holds tokens 1-1000.
- GPU 2 holds tokens 1001-2000.
- **Ring Attention:** Pass Key/Value blocks around in a ring.
- GPU 1 computes Attention(Q1, K1). Then receives K2 from neighbor. Computes Attention(Q1, K2).
- **Result:** Can train on infinite sequence lengths (limited only by number of GPUs).

### Code: Toy Tensor Parallel Linear Layer

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Assume world_size is set
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Split output dimension
        self.output_size_per_partition = output_size // world_size
        
        # Initialize weights for this partition
        self.weight = nn.Parameter(torch.randn(
            self.output_size_per_partition, input_size
        ))
        
    def forward(self, input):
        # Input is replicated on all GPUs
        # Output is split across GPUs
        return F.linear(input, self.weight)

class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        world_size = dist.get_world_size()
        
        # Split input dimension
        self.input_size_per_partition = input_size // world_size
        
        self.weight = nn.Parameter(torch.randn(
            output_size, self.input_size_per_partition
        ))
        
    def forward(self, input):
        # Input is split across GPUs
        partial_output = F.linear(input, self.weight)
        
        # All-Reduce to sum partial outputs
        dist.all_reduce(partial_output)
        
        return partial_output
```
