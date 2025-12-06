# Day 170: Splitting the Brain: Tensor & Pipeline Parallelism
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** A 175B parameter model requires 350GB of VRAM (FP16). The biggest GPU has 80GB. You cannot fit the brain in one skull. **Model Parallelism** is the art of slicing the Neural Network across multiple GPUs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline Parallelism (PP).
2.  **Implement** simple Tensor Parallelism using `torch.distributed`.
3.  **Explain** the communication overhead (All-Reduce vs All-Gather) involved in splitting Matrix Multiplications.
4.  **Architect** a 3D Parallelism strategy (DP + TP + PP) for massive model training.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Machine with at least 2 GPUs (Simulation possible on CPU).

### Software Environment
- `pip install torch deepspeed`.

---

## 📖 Theoretical Foundation

### 1. The Parallelism Hierarchy
*   **Data Parallel (DP):** Replicate full model on every GPU. Split Batch.
    *   *Constraint:* Model must fit on 1 GPU.
*   **Tensor Parallel (TP):** Split each Layer (Matrix) across GPUs.
    *   *Constraint:* Requires extremely fast interconnect (NVLink) because every layer triggers communication. (Intra-Node).
*   **Pipeline Parallel (PP):** Put layers 1-10 on GPU 0, 11-20 on GPU 1.
    *   *Constraint:* Bubble Inefficiency (GPU 1 waits for GPU 0). (Inter-Node).

### 2. Matrix Multiplication Splitting (TP)
If $Y = X \cdot A$, and we split $A$ into $[A_1, A_2]$ (Column Parallel):
*   GPU 1 computes $Y_1 = X \cdot A_1$.
*   GPU 2 computes $Y_2 = X \cdot A_2$.
*   No communication needed *during* compute.
*   **Result:** Outputs need to be Concatenated (All-Gather) for the next layer.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Toy Tensor Parallelism

Simulating splitting a Linear Layer.

#### 📁 `src/tp_demo.py`
```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_gpus, gpu_id):
        super().__init__()
        # Split output features across GPUs
        self.partition_size = out_features // num_gpus
        self.weight = nn.Parameter(torch.randn(in_features, self.partition_size))
        
    def forward(self, x):
        # x is replicated on all GPUs (Data Parallel input? No, Input assumed broadcasted)
        # Compute local part of output
        return torch.matmul(x, self.weight)

def run_tp(rank, world_size):
    # Simulate Distributed logic
    print(f"GPU {rank}: Initializing...")
    
    # Input (Batch=1, Dim=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]]) 
    
    # Layer: 4 -> 8. Split across 2 GPUs (each outputs 4 features)
    layer = ColumnParallelLinear(4, 8, num_gpus=world_size, gpu_id=rank)
    
    # Forward
    y_local = layer(x)
    print(f"GPU {rank}: Output Shape {y_local.shape}")
    
    # In real TP, we would All-Gather here to reconstruct full [1, 8] output
    # or pass Partial results to RowParallelLinear next.

if __name__ == "__main__":
    mp.spawn(run_tp, args=(2,), nprocs=2, join=True)
```

### 👨‍💻 Infrastructure: DeepSpeed Configuration

DeepSpeed abstracts this complexity.

#### 📁 `ds_config.json`
```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5
    }
  },
  "tensor_parallel": {
    "tp_size": 2   # Split layers across 2 GPUs
  },
  "pipeline_parallel": {
    "pp_size": 2   # Split depth across 2 Groups
  },
  "fp16": {
    "enabled": true
  }
}
```

### 👨‍💻 Infrastructure: Launching Distributed Training

```bash
deepspeed --num_gpus=4 train_llm.py --deepspeed_config ds_config.json
```
With 4 GPUs, `tp_size=2` and `pp_size=2`.
*   GPUs [0, 1] form Stage 1 of Pipeline. They split tensor computation.
*   GPUs [2, 3] form Stage 2 of Pipeline. They split tensor computation.

---

## 🔬 Lab Exercise: "The Bubble"

### Task
Visualize Pipeline Inefficiency.
1.  **Setup:** PP with 4 GPUs.
2.  **Forward Pass:**
    *   Time 0: GPU 0 processes Batch 1. GPU 1,2,3 Idle.
    *   Time 1: GPU 0 processes Batch 2. GPU 1 processes Batch 1. GPU 2,3 Idle.
3.  **Optimization:** **Micro-Batching**. Break "Batch 1" into "MicroBatch 1a, 1b, 1c".
4.  **Result:** Pipelining fills up faster. "Idle Bubble" reduces.
5.  **Tradeoff:** More Micro-Batches = More Memory overhead (Activation Stashing).

---

## 📖 Advanced Theory: Zero Redundancy Optimizer (ZeRO)
Data Parallelism replicates Weights, Gradients, and Optimizer States on ALL GPUs. (Redundant).
**ZeRO-3 (Partitioning):**
*   Shards the Model Weights across GPUs.
*   When GPU 0 needs weights for Layer 1, it fetches them from GPU 1, computes, then effectively discards them.
*   **Result:** Infinite model size support (linear with GPU count), but heavier communication.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Network is King:** Tensor Parallelism requires > 600 GB/s bandwidth (NVLink). Do not try TP over Ethernet (10 GB/s); it will be slower than a single GPU.
2.  **Granularity:**
    *   **TP:** Inside a single Node (8 GPUS).
    *   **PP:** Across Nodes (Scale Out).
    *   **DP:** Replicate entire setup for throughput.
3.  **Complexity:** Debugging distributed hangs is a nightmare. Use `NCCL_DEBUG=INFO` to trace communication blocks.

### API Summary
```python
dist.all_gather(tensor_list, tensor)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

---

**Day 170 Complete** ✅

*Next: Day 171 - Efficient Inference - vLLM & PagedAttention.*
