# Lab 4: Distributed Training (DDP)

## Objective
Train on multiple GPUs (or simulated processes).
We will use `torch.distributed`.

## 1. The Script (`ddp.py`)

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Model
    model = torch.nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank]) if torch.cuda.is_available() else DDP(model)
    
    # Loss & Optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Step
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Rank {rank} finished step. Loss: {loss.item()}")
    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## 2. Running
Run `python ddp.py`. You should see output from Rank 0 and Rank 1.

## 3. Submission
Submit the console output showing both ranks.
