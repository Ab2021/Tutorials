# Day 100: The Migration: Porting PyTorch to Ray
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** You don't write new code for every project. Most often, you are porting an existing `train.py` from a research paper repo to your production Ray cluster.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** the 3 blocks of code that must be removed from legacy scripts (`device` selection, `sampler` setup, `process_group` init).
2.  **Refactor** a CLI-based script into a Function-based API.
3.  **Port** a PyTorch Lightning `pl.Trainer` to Ray's `LightningTrainer`.
4.  **Debug** common multi-node connectivity issues (Firewalls/SG).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU Machine (Recommended).

### Software Environment
- `pip install pytorch-lightning "ray[train]"`

---

## 📖 Theoretical Foundation

### 1. The "Device" Anti-Pattern
Legacy code often has:
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
In Ray Train, every worker thinks it is on `cuda:0` because Ray sets `CUDA_VISIBLE_DEVICES` to isolate a single GPU per worker (usually).
**Rule:** Always use `ray.train.torch.get_device()`.

### 2. The Sampler
Legacy:
```python
sampler = DistributedSampler(dataset, rank=rank)
loader = DataLoader(..., sampler=sampler)
```
Ray:
```python
loader = prepare_data_loader(loader)
```
Ray creates the `DistributedSampler` internally with correct seed and rank.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Porting Guide

#### ❌ Before (Legacy `main.py`)
```python
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(args.local_rank)
    
    model = MyModel().cuda()
    model = DDP(model, device_ids=[args.local_rank])
    # ... training loop ...
```

#### ✅ After (Ray `train_func`)
```python
import ray.train.torch

def train_func(config):
    # Args come from config dictionary
    epochs = config.get("epochs", 10)
    
    # Device management handled by Ray
    model = MyModel()
    model = ray.train.torch.prepare_model(model)
    
    # ... training loop ... (No dist.init, No DDP wrap)
```

### 👨‍💻 Core Implementation: PyTorch Lightning

Lightning is even easier because it abstracts the loop. Ray provides a specific integration.

```python
import pytorch_lightning as pl
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer

def train_func_lightning(config):
    model = MyLightningModule()
    
    # Ray sets up the PL Strategy plugins
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    
    # Validate the trainer setup
    trainer = prepare_trainer(trainer)
    
    trainer.fit(model, train_dataloaders=my_loader)

# Driver
scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
trainer = TorchTrainer(
    train_loop_per_worker=train_func_lightning,
    scaling_config=scaling_config
)
trainer.fit()
```

---

## 🔬 Lab Exercise: "The Bad Seed"

### Task
Reproducibility.
1.  Legacy DDP scripts often forget to set `shard_seed` or `seed`.
2.  In Ray, verify that `prepare_data_loader` shards data correctly.
3.  Run training with 2 workers.
4.  Print `batch[0]` in each worker.
5.  **Observation:** They must be different. If they are the same, you are training on duplicates (Double epoch, zero diversity).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Cleaner Code:** Ray Train forces you to decouple your *Model Logic* (`train_func`) from your *Infrastructure Logic* (Driver script). This is good software engineering.
2.  **Lightning:** If you use Lightning, the migration is literally 3 lines of code (Change Strategy to `RayDDPStrategy`).
3.  **Logs:** stdout from workers is streamed to the Driver. You don't need to SSH to check worker logs.

### API Summary
```python
prepare_data_loader(loader)
prepare_model(model)
```

---

**Day 100 Complete** ✅

*Next: Day 101 - Data Loading at Scale - Feeding the beast with Ray Data.*
