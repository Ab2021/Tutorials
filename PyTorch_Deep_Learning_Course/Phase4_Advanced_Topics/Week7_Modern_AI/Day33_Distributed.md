# Day 33: Distributed Training - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: DDP, FSDP, and Multi-GPU

## 1. Theoretical Foundation: Data vs Model Parallelism

Single GPU memory (e.g., 24GB) is the bottleneck.
1.  **Data Parallelism (DDP)**: Replicate model on all GPUs. Split batch across GPUs. Sync gradients.
2.  **Model Parallelism (Pipeline/Tensor)**: Split model layers across GPUs.
3.  **Sharded Data Parallelism (FSDP/ZeRO)**: Shard model states across GPUs.

## 2. Distributed Data Parallel (DDP)

*   **Process**: Spawns 1 process per GPU.
*   **Forward**: Each GPU computes loss on its micro-batch.
*   **Backward**: Gradients computed locally.
*   **All-Reduce**: Gradients are averaged across all GPUs (Ring All-Reduce).
*   **Update**: Optimizer steps identically on all GPUs.

## 3. Fully Sharded Data Parallel (FSDP)

DDP replicates the model parameters. Redundant.
**FSDP (ZeRO-3)**:
*   Shard Parameters, Gradients, and Optimizer States.
*   Each GPU holds only $1/N$ of the model.
*   **All-Gather**: Before Forward/Backward, fetch necessary params from other GPUs.
*   Allows training 1T+ parameter models.

## 4. Implementation: DDP with PyTorch

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Move model to GPU 'rank'
    model = MyModel().to(rank)
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Dataloader with DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
    cleanup()

# Run with: torchrun --nproc_per_node=4 script.py
```

## 5. Implementation: FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyModel().to(rank)
model = FSDP(model) # Automatically shards layers
```
