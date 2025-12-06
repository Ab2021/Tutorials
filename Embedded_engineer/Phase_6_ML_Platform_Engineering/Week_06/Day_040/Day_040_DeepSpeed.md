# Day 40: DeepSpeed & Megatron-LM
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 6: Distributed Training & Large Scale Systems

---

> **🎯 Focus Area:** Master **DeepSpeed**, the open-source industry standard for training massive models. Learn how to plug DeepSpeed into PyTorch to instantly gain access to ZeRO stages, CPU Offloading, and Fused Optimizers.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Convert** a standard PyTorch training script to use DeepSpeed.
2.  **Configure** ZeRO Stage 2 and Stage 3 (Offload) via `ds_config.json`.
3.  **Explain** CPU Offloading (NVMe Offloading) and when to use it.
4.  **Launch** distributed jobs using the `deepspeed` CLI.
5.  **Understand** the relationship between Hugging Face `Trainer` and DeepSpeed.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU.
- RAM (for CPU Offloading demos).

### Software Environment
```bash
pip install deepspeed mpi4py
# Ensure CUDA toolkit is visible for JIT compiling DS ops
```

### Prior Knowledge
- Day 37: ZeRO Concepts (Sharding).
- Day 36: DDP.

---

## 📖 Theoretical Foundation

### 1. DeepSpeed vs FSDP
*   **PyTorch FSDP:** Native, maintained by Meta. Good integration.
*   **Microsoft DeepSpeed:** Third-party library. Often "ahead" in features (ZeRO-Infinity, Sparse Attention, 1-bit Adam).
*   **Why use DeepSpeed?**
    *   **Ease:** Just write a JSON file.
    *   **Memory:** "ZeRO-Offload" allows training gradients/optimizer states in CPU RAM, allowing massive models on consumer GPUs (e.g., 10B parameters on an RTX 3090, albeit slow).

### 2. The `ds_config.json`
The brain of DeepSpeed.
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu" }
  },
  "fp16": { "enabled": true }
}
```

### 3. Megatron-LM
A robust, specialized repository by NVIDIA for training GPT/BERT using Tensor Parallelism. DeepSpeed is often combined with Megatron-LM (Megatron-DeepSpeed) to use **3D Parallelism** (Data + Tensor + Pipeline).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: DeepSpeed Integration

We will take a vanilla PyTorch model and wrap it with DeepSpeed.

#### 📁 `src/ds_config.json`
```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 10,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.001,
      "betas": [0.9, 0.999],
      "eps": 1e-8
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "offload_optimizer": {
        "device": "cpu"
    }
  }
}
```

#### 📁 `src/ds_train.py`
```python
#!/usr/bin/env python3
"""
Day 40: DeepSpeed Training
Phase 6: Distributed Training
usage: deepspeed src/ds_train.py --deepspeed_config src/ds_config.json
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import deepspeed
import argparse

# Dummy Components
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class RandomDataset(Dataset):
    def __init__(self, len=1000):
        self.data = torch.randn(len, 1024)
        self.target = torch.randn(len, 1024)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.target[i]

def main():
    parser = argparse.ArgumentParser()
    # DeepSpeed adds arguments automatically if you use deepspeed.add_config_arguments(parser)
    # But often we just pass --deepspeed_config manually
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    args = parser.parse_args()

    # 1. Initialize DeepSpeed
    # It initializes the distributed backend (nccl) automatically
    model = SimpleModel()
    dataset = RandomDataset()
    
    # We don't define Optimizer or Dataloader manually (usually)
    # DeepSpeed creates them based on Config
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config='src/ds_config.json'
    )
    
    # 2. Training Loop
    # Note: loop style changes slightly
    for step, batch in enumerate(trainloader):
        # Move data to GPU (model_engine.device)
        inputs = batch[0].to(model_engine.device)
        labels = batch[1].to(model_engine.device)
        
        # Forward
        outputs = model_engine(inputs)
        loss = nn.functional.mse_loss(outputs, labels)
        
        # Backward
        # Replaces loss.backward()
        model_engine.backward(loss)
        
        # Step
        # Replaces optimizer.step()
        model_engine.step()
        
        if step % 10 == 0 and args.local_rank == 0:
            print(f"Step {step}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
```

### 👨‍💻 Launching DeepSpeed

```bash
# Single Node, all GPUs
deepspeed src/ds_train.py

# With Explicit Config (if not handled in args)
deepspeed src/ds_train.py --deepspeed_config src/ds_config.json
```

---

## 🔬 Lab Exercise: "CPU Offload Stress Test"

### Task
1.  Set `zero_optimization.stage = 2`.
2.  Set `offload_optimizer.device = "cpu"`.
3.  Increase Model Size drastically (e.g., Linear(10000, 10000)).
4.  Run and observe `htop` (CPU Usage) and `nvidia-smi` (VRAM).
5.  **Observation:** CPU RAM fills up with Optimizer states ($M$ and $V$ in Adam). GPU VRAM stays relatively low (only Weights + Grads + Activations).
6.  *Bonus:* Change to `stage = 3` and `offload_param.device = "cpu"`. Now Weights live on CPU too. GPU VRAM is minimal. You can train massive models slowly.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Abstraction:** DeepSpeed hides the `InitProcessGroup`, `DistributedSampler`, and `FSDP` wrappers. You just call `deepspeed.initialize`.
2.  **Config Driven:** Changing from DDP (Stage 0) to FSDP (Stage 3) is a JSON edit, not a code rewrite. This is powerful for experimentation.
3.  **Hugging Face:** If you use HF Trainer, DeepSpeed is built-in. Just pass `--deepspeed ds_config.json`.
4.  **Optimizers:** DeepSpeed includes a custom `FusedAdam` kernel that is significantly faster than PyTorch's native implementation.

### API Summary
```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=params, config=config
)
model_engine.backward(loss)
model_engine.step()
```

---

**Day 40 Complete** ✅

*Next: Day 41 - Slurm & Cluster Management - How to run these scripts on a supercomputer/cluster.*
