# Day 36: Scaling to Multiple GPUs: Distributed Data Parallel (DDP)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 6: Distributed Training & Large Scale Systems

---

> **🎯 Focus Area:** Graduate from single-GPU training to Multi-GPU, Multi-Node training. Understand why **DistributedDataParallel (DDP)** is the industry standard over `DataParallel`, using the Ring-AllReduce algorithm.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the bottleneck of the Parameter Server model (DP) vs Ring-AllReduce (DDP).
2.  **Define** Distributed Terms: Rank, World Size, Local Rank, Master Addr/Port.
3.  **Refactor** a standard PyTorch training script to support DDP.
4.  **Use** `DistributedSampler` to shard datasets across GPUs.
5.  **Launch** a training job using `torchrun`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- At least 2 GPUs (To see DDP in action).
- *If only 1 GPU is available, you can simulate DDP, but speedup won't be visible.*

### Software Environment
```bash
# PyTorch comes with torch.distributed
pip install torch torchvision
```

### Prior Knowledge
- Standard PyTorch Training Loop.
- Linux Process Management (Signals, Environment Variables).

---

## 📖 Theoretical Foundation

### 1. The Problem with `DataParallel` (DP)
In early PyTorch, you used `model = nn.DataParallel(model)`.
*   **Architecture:** Single Process, Multi-Thread.
*   **Workflow:** GPU 0 (Server) splits batch, sends to GPU 1-3. GPU 1-3 compute grad, send back to GPU 0. GPU 0 updates weights, broadcasts new weights.
*   **Issues:**
    *   **GIL:** Python Global Interpreter Lock throttles multithreading.
    *   **Imbalance:** GPU 0 does 4x work (IO, Reduce, Broadcast).
    *   **Network:** Cannot scale across multiple nodes.

### 2. The Solution: `DistributedDataParallel` (DDP)
*   **Architecture:** Multi-Process (1 process per GPU).
*   **Workflow:** Every GPU has a copy of the model. They compute gradients independently.
*   **Synchronization:** They use **NCCL (NVIDIA Collective Communications Library)** to perform an `AllReduce` (Sum) of gradients.
*   **Efficiency:** No GIL issues. Scale to 1000s of GPUs.

### 3. Key Vocabulary
*   **World Size:** Total number of processes (GPUs) in the job (e.g., 2 nodes x 4 GPUs = 8).
*   **Rank:** Global ID (0 to 7). Rank 0 usually handles logging/checkpointing.
*   **Local Rank:** ID on the specific machine (0 to 3). Used to select device.
*   **Broadcast:** Sending data from one to all.
*   **AllReduce:** Summing data from all and sending result to all.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Converting to DDP

We will take a generic training script and inject DDP logic.

#### 📁 `src/ddp_training.py`
```python
#!/usr/bin/env python3
"""
Day 36: Distributed Data Parallel Training
Phase 6: Distributed Training
usage: torchrun --nproc_per_node=2 src/ddp_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import sys

# 1. Setup Function
def setup():
    # Torchrun sets these env vars automatically
    # RANK: Global rank
    # LOCAL_RANK: Rank on this specific node
    # WORLD_SIZE: Total GPUs
    
    if "RANK" not in os.environ:
        print("Not running in DDP mode. Use 'torchrun'.")
        sys.exit(1)
        
    dist.init_process_group("nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# Dummy Dataset
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# Simple Model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic_ddp():
    # A. Init
    local_rank = setup()
    global_rank = int(os.environ["RANK"])
    
    print(f"[Rank {global_rank}] Setup Complete. Running on GPU {local_rank}")
    
    # B. Model
    model = ToyModel().to(local_rank)
    # Wrap with DDP
    # device_ids tells DDP which GPU to use for this process
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # C. Data
    # Crucial: DDP requires DistributedSampler to ensure each GPU gets DIFFERENT data
    dataset = RandomDataset(10, 100)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=20, sampler=sampler)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # D. Training Loop
    for epoch in range(5):
        # Crucial: Set epoch for sampler (reshuffles data deterministically per epoch)
        sampler.set_epoch(epoch)
        
        for i, data in enumerate(dataloader):
            data = data.to(local_rank)
            
            optimizer.zero_grad()
            outputs = ddp_model(data)
            labels = torch.randn(20, 5).to(local_rank)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
        if global_rank == 0:
            print(f"Epoch {epoch} complete.")
            # Logging/Checkpointing only on Rank 0

    if global_rank == 0:
        # Saving model: Save the 'module' inside DDP
        torch.save(ddp_model.module.state_dict(), "model.pth")
        print("Model saved.")

    cleanup()

if __name__ == "__main__":
    demo_basic_ddp()
```

### 👨‍💻 Launching DDP

The correct way to run this script on a machine with 2 GPUs:

```bash
torchrun --nproc_per_node=2 src/ddp_training.py
```

*   `torchrun`: Handles expanding the script into 2 processes.
*   Assigns `RANK=0, LOCAL_RANK=0` to process 1.
*   Assigns `RANK=1, LOCAL_RANK=1` to process 2.
*   Handles Master IP/Port discovery automatically (localhost).

---

## 🔬 Lab Exercise: "Gradient Synchronization Check"

### Task
Verify that DDP actually synchronizes gradients.
1.  Insert a print statement in the training loop:
    ```python
    print(f"Rank {global_rank} Grad: {ddp_model.module.net1.weight.grad[0][0]}")
    ```
2.  Run with 2 GPUs.
3.  **Observation:** Even though each GPU sees *different data* (and computes different initial gradients), after `loss.backward()` (which triggers the AllReduce hook), the `.grad` values on both GPUs should be **IDENTICAL**.

If they are different, DDP is broken (or you forgot to zero_grad).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Shared-Nothing:** Unlike DataParallel (Threaded), DDP uses Processes. They share no memory. They communicate largely via NCCL over PCIe/NVLink/Infiniband.
2.  **Sampler Magic:** Accessing the same dataset file is fine, but `DistributedSampler` uses `rank` and `world_size` to partition indices so GPU 0 reads indices `[0, 2, 4]` and GPU 1 reads `[1, 3, 5]`.
3.  **Batch Size:** Effective Batch Size = `Batch_Per_GPU` * `World_Size`. If you use batch 20 on 4 GPUs, your actual mathematical batch size is 80. You may need to scale Learning Rate accordingly (Linear Scaling Rule).

### API Summary
```python
# Launch
torchrun --nproc_per_node=4 script.py

# Code
dist.init_process_group("nccl")
model = DDP(model.to(rank), device_ids=[rank])
sampler = DistributedSampler(dataset)
```

---

**Day 36 Complete** ✅

*Next: Day 37 - FSDP (Fully Sharded Data Parallel) - What to do when the model doesn't fit on one GPU.*
