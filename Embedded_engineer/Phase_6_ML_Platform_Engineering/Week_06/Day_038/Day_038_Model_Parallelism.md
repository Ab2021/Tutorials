# Day 38: Model Parallelism - Tensor Parallel (TP) & Pipeline Parallel (PP)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 6: Distributed Training & Large Scale Systems

---

> **🎯 Focus Area:** Understand **Tensor Parallelism (TP)**, the technique used by Megatron-LM to train GPT-3. Instead of splitting data (DDP), we split the matrix multiplication itself across GPUs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Data Parallel (Split Batch) and Tensor Parallel (Split Weights).
2.  **Implement** a `ColumnParallelLinear` layer using `dist.all_gather`.
3.  **Implement** a `RowParallelLinear` layer using `dist.all_reduce`.
4.  **Analyze** the communication cost of TP (High bandwidth required, requires NVLink).
5.  **Explain** Pipeline Parallelism and the "Bubble" problem.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- At least 2 GPUs.
- **High-speed Interconnect (NVLink)** is highly recommended. TP is very chatty. On PCIe/Ethernet, TP is often slower than single GPU.

### Software Environment
- PyTorch Distributed.

---

## 📖 Theoretical Foundation

### 1. Matrix Multiplication splitting
Consider Linear Layer $Y = X A$.
*   $X$: Input vector.
*   $A$: Weight Matrix ($N \times M$).

**Column Parallelism:**
Split $A$ vertically into $[A_1, A_2]$.
*   GPU 1 computes $Y_1 = X A_1$.
*   GPU 2 computes $Y_2 = X A_2$.
*   Output $Y = [Y_1, Y_2]$ (Concatenation / AllGather).

**Row Parallelism:**
Split $A$ horizontally into $\begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$. Split $X$ into $[X_1, X_2]$.
*   GPU 1 computes $Y_1 = X_1 A_1$.
*   GPU 2 computes $Y_2 = X_2 A_2$.
*   Output $Y = Y_1 + Y_2$ (Sum / AllReduce).

### 2. The Megatron-LM Style (TP)
In a Transformer MLP: `Linear -> GeLU -> Linear`.
*   Layer 1: **Column Parallel**. Output is split.
*   Non-Linearity (GeLU): Applied on split output independently.
*   Layer 2: **Row Parallel**. Takes split input, produces partial sum.
*   **Result:** Only **ONE** synchronization (AllReduce) needed after Layer 2. This optimization is why Megatron is fast.

### 3. Pipeline Parallelism (PP)
Naive: GPU0 (L1-4) -> GPU1 (L5-8).
*   GPU 1 sits idle while GPU 0 works (Bubble).
*   **1F1B (One Forward One Backward):** Schedule micro-batches to fill bubbles.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Manual Tensor Parallel Layers

We will implement the building blocks of Megatron-LM.

#### 📁 `src/tensor_parallel.py`
```python
#!/usr/bin/env python3
"""
Day 38: Manual Tensor Parallelism
Phase 6: Distributed Training
usage: torchrun --nproc_per_node=2 src/tensor_parallel.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import os

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# Helper for Differentiable Communication
class CopyAndGather(torch.autograd.Function):
    """
    Forward: Pass input Identity.
    Backward: AllReduce gradients (because input was broadcasted to all GPUs, 
              so gradients from all GPUs must be summed).
    """
    @staticmethod
    def forward(ctx, x):
        return x
        
    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output

class GatherAndScatter(torch.autograd.Function):
    """
    Forward: AllGather (Concatenate).
    Backward: Split gradients (Scatter).
    """
    @staticmethod
    def forward(ctx, x):
        # x: [Batch, Hidden/World]
        world_size = dist.get_world_size()
        # Create list of tensors to gather into
        gather_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gather_list, x)
        # Concat along last dim
        return torch.cat(gather_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is full size. We only need our slice.
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # Split along last dim
        chunks = grad_output.chunk(world_size, dim=-1)
        return chunks[rank]

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        world_size = dist.get_world_size()
        # Split Output features
        assert out_features % world_size == 0
        self.out_features_per_partition = out_features // world_size
        
        # Define Weight Slice
        self.weight = nn.Parameter(torch.randn(self.out_features_per_partition, in_features))
        
    def forward(self, x):
        # x is duplicated on all GPUs (Identity)
        # weight is split
        # y_local = x @ weight_local.T
        
        # Important: In Column Parallel, inputs are usually replicated.
        # But if coming from RowParallel, we handle logic differently.
        # Here assuming replicated input 'x'.
        
        # x: [Batch, In]
        # w: [Out/P, In]
        # y: [Batch, Out/P]
        
        # We use CopyAndGather on Input to handle backward pass sync
        x_input = CopyAndGather.apply(x)
        
        y = nn.functional.linear(x_input, self.weight)
        return y

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        world_size = dist.get_world_size()
        # Split Input features
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        
        self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_partition))
        
    def forward(self, x):
        # x is split [Batch, In/P]
        # y local = x_split @ w_split.T -> [Batch, Out] (Partial Sum)
        
        y_partial = nn.functional.linear(x, self.weight)
        
        # Reduce to get full sum
        # In Megatron, they reduce at end.
        
        # Manual AllReduce
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
        return y_partial

def demo_megatron_block():
    rank = setup()
    
    # Define a simple MLP: Linear(10->20) -> RELU -> Linear(20->10)
    # Using Megatron Strategy:
    # L1: Column Parallel (Split 20 -> 10 per GPU)
    # L2: Row Parallel (Split 20 inputs -> 10 per GPU)
    
    # 2 GPUs
    l1 = ColumnParallelLinear(10, 20).cuda()
    l2 = RowParallelLinear(20, 10).cuda()
    
    # Input is replicated
    x = torch.randn(4, 10).cuda()
    
    # Forward
    # 1. Column Linear. 
    # x: [4, 10]. out: [4, 10] (Not 20! It's split)
    x_intermediate = l1(x)
    
    # 2. Nonlinearity
    # Applied on split parts individually! No comms needed!
    x_intermediate = torch.nn.functional.relu(x_intermediate)
    
    # 3. Row Linear
    # Takes [4, 10] slice. Computes partial. Sums.
    final_out = l2(x_intermediate)
    
    print(f"Rank {rank}: Output Shape {final_out.shape}") # Should be [4, 10]
    
    if rank == 0:
        print("Megatron Block Forward Pass Successful.")

if __name__ == "__main__":
    demo_megatron_block()
```

---

## 🔬 Lab Exercise: "DDP vs TP Latency"

### Task
1.  Run the TP script.
2.  Run a standard DDP script with the same model size.
3.  Profile which is faster for small models vs huge models.

### Insight
*   **Small Model:** TP is SLOWER. Communication overhead dominates matrix math.
*   **Huge Model (10B+):** TP is FASTER (or necessary). It fits in memory, and matrix math is so heavy that communication time is hidden.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Megatron-LM Trick:** Identify layers where you can split the computation and *avoid* communication until the very end. `ColLinear` -> `NonLinear` -> `RowLinear` requires only **one** AllReduce.
2.  **Bandwidth Hungry:** FSDP shards once per layer. TP shards inside the layer. TP requires extremely low latency (NVLink) or it stalls the GPU cores waiting for partial sums.
3.  **Use Cases:**
    *   **DDP/FSDP:** General purpose.
    *   **TP:** Only for creating massive models (GPT-4, Llama 3 70B) within a single node (8 GPUs with NVLink).
    *   **PP:** For scaling across multiple nodes when TP fails.

### API Summary
```python
dist.all_reduce(tensor) # Sums tensor across all GPUs
dist.all_gather(list, tensor) # Gathers tensors
```

---

**Day 38 Complete** ✅

*Next: Day 39 - RDMA & NCCL - The networking magic that makes distributed training possible.*
