# Day 37: Fully Sharded Data Parallel (FSDP)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 6: Distributed Training & Large Scale Systems

---

> **🎯 Focus Area:** Overcome GPU memory limits by using **FSDP (Fully Sharded Data Parallel)**. Learn how to train massive models (LLMs) by sharding weights, gradients, and optimizer states across the cluster.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** DDP (Replicated Weights) with FSDP (Sharded Weights).
2.  **Explain** the ZeRO (Zero Redundancy Optimizer) stages 1, 2, and 3.
3.  **Implement** FSDP wrapping strategies for Transformer models using `auto_wrap_policy`.
4.  **Manage** Mixed Precision (FP16/BF16) within FSDP.
5.  **Save/Load** huge checkpoints efficiently using Distributed State Dicts.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- at least 2 GPUs (FSDP needs multiple devices to shard across).
- *Scenario:* Using FSDP on a single GPU is meaningless (it just acts like standard training with overhead).

### Software Environment
- PyTorch > 1.12 (Native FSDP support).

### Prior Knowledge
- Day 36: DDP workflow.
- Transformer Architecture (Layers, Attention blocks).

---

## 📖 Theoretical Foundation

### 1. DDP vs FSDP Memory Usage

Assume a Model with parameters $\Psi$, Gradients $G$, Optimizer States $O$.
*   **DDP:** Memory per GPU = $\Psi + G + O$. (Redundant! Everyone stores the same $\Psi$).
*   **FSDP (ZeRO-3):** Memory per GPU = $\frac{\Psi + G + O}{N}$.
    *   If you have 8 GPUs, you use 1/8th the memory.
    *   *Trade-off:* Communication overhead. When Forward Pass reaches Layer 1, all GPUs must `AllGather` the full Layer 1 shards, compute, and then free them.

### 2. Wrapping & Auto-Wrap

You don't just wrap the whole model `FSDP(model)`. This would try to gather the *entire* model at once, defeating the purpose.
*   **Strategy:** Wrap *each transformer block* individually.
*   **Execution:**
    1.  Gather Block 1 Params.
    2.  Compute Block 1.
    3.  Discard Block 1 Params (Keep Shards only).
    4.  Gather Block 2 Params...

### 3. Mixed Precision
FSDP handles precision internally:
*   **Param Dtype:** Storage format (usually FP32 for stability).
*   **Reduce Dtype:** Gradient communication format (FP32 or FP16).
*   **Buffer Dtype:** Activation buffers.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: FSDP Transformer

We will define a simple Transformer and train it with FSDP.

#### 📁 `src/fsdp_training.py`
```python
#!/usr/bin/env python3
"""
Day 37: FSDP Training
Phase 6: Distributed Training
usage: torchrun --nproc_per_node=2 src/fsdp_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import torch.distributed as dist
import os
import functools

# 1. Define a Mini-Transformer Layer (The logic we want to wrap)
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [Seq, Batch, Dim]
        res, _ = self.attn(x, x, x)
        x = self.norm1(x + res)
        res = self.ffn(x)
        x = self.norm2(x + res)
        return x

class SimpleLLM(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        # Stack blocks
        self.blocks = nn.Sequential(*[TransformerBlock(128) for _ in range(layers)])
        self.head = nn.Linear(128, 1000)

    def forward(self, x):
        x = self.blocks(x)
        return self.head(x[0]) # Dummy head

# 2. Setup (Same as DDP)
def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    rank = setup()
    
    # 3. Define FSDP Config
    # Auto Wrap: Tell FSDP to treat 'TransformerBlock' as the unit of sharding
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    # Mixed Precision: Store in FP32, Compute in BF16 (requires Ampere) or FP16
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # 4. Initialize Model
    # Important: Move to GPU *before* wrapping if using CUDA init, 
    # OR use device_id argument in FSDP to init on device.
    model = SimpleLLM().cuda(rank)
    
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        # SHARD_GRAD_OP: ZeRO-2, FULL_SHARD: ZeRO-3
        sharding_strategy=ShardingStrategy.FULL_SHARD, 
        device_id=torch.cuda.current_device()
    )

    if rank == 0:
        print("Model Wrapped with FSDP.")
        # Print structure to see wrapping
        print(fsdp_model)

    optimizer = optim.AdamW(fsdp_model.parameters(), lr=1e-3)
    
    # 5. Training Loop
    # Dummy data
    data = torch.randn(32, 16, 128).cuda(rank) # Seq, Batch, Dim
    target = torch.randint(0, 1000, (16,)).cuda(rank)
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(10):
        optimizer.zero_grad()
        out = fsdp_model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        
        if rank == 0 and i % 2 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    # 6. Checkpointing (Advanced)
    # With FSDP, we can't just torch.save(model.state_dict()).
    # We must set state_dict_type.
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    
    # Save Full Model (Consolidated) on Rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = fsdp_model.state_dict()
        if rank == 0:
            torch.save(cpu_state, "fsdp_model.pth")
            print("Full state dict saved to disk.")

    cleanup()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Memory Profiling"

### Task
Measure the VRAM saved by FSDP compared to DDP.
1.  Increase `TransformerBlock` hidden dim to 4096 and layers to 12 (Make it heavy).
2.  Run with DDP (Day 36 script). Watch `nvidia-smi`. It will likely OOM (Out of Memory).
3.  Run with FSDP (Day 37 script).
4.  Observe max allocated memory using `torch.cuda.max_memory_allocated()`.

### Expectation
FSDP usage should be roughly `1/N` of DDP usage for parameters, allowing you to fit models N times larger.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Communication vs Computation:** FSDP trades network bandwidth for memory. If your network (Interconnect) is slow (e.g., 1Gbps Ethernet), FSDP will be unbearably slow. It requires **NVLink** or fast inter-node networking (100Gbps+).
2.  **Auto-Wrap:** Critical. If you wrap the whole model, FSDP gathers all params at start (OOMs). If you wrap layers, it gathers/frees layers one by one (Streaming).
3.  **ZeRO Stages:**
    *   Stage 1: Optimizer Sharding (4x memory savings).
    *   Stage 2: Gradient Sharding (2x savings + Stage 1).
    *   Stage 3: Parameter Sharding (Linear N savings).

### API Summary
```python
# Wrap
model = FSDP(model, auto_wrap_policy=policy)

# Save
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
    state = model.state_dict()
```

---

**Day 37 Complete** ✅

*Next: Day 38 - Model Parallelism (Tensor Parallel / Pipeline Parallel) - Splitting layers when FSDP isn't enough.*
