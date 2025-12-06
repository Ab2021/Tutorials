# Day 99: Beyond torch.distributed.launch: Ray Train
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** Managing `MASTER_ADDR` and `WORLD_SIZE` manually is fragile. **Ray Train** creates a fully managed Distributed Data Parallel (DDP) environment with a simple Python API.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** how Ray Train replaces the `torchrun` / `torch.distributed.launch` CLI.
2.  **Refactor** a standard PyTorch training loop into a `train_loop_per_worker`.
3.  **Configure** a `ScalingConfig` to request GPUs across the cluster.
4.  **Launch** a multi-worker training job from a Jupyter Notebook.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Machine with GPUs (or CPU simulation).

### Software Environment
- `pip install "ray[train]"`
- `pip install torch torchvision`

---

## 📖 Theoretical Foundation

### 1. The DDP Boilerplate Problem
To run standard DDP:
1.  Reserve 4 nodes (Slurm/K8s).
2.  SSH into each.
3.  Set `MASTER_ADDR=10.0.0.1`.
4.  Run `python train.py --rank=0`, `...rank=1`, etc.
5.  If one node fails, everything hangs.

### 2. The Ray Train Solution
*   **Trainer Actor:** The orchestrator. Resides on the Head/Driver.
*   **Worker Group:** Ray requests N actors with GPU resources.
*   **Coordination:** Ray automatically injects `MASTER_ADDR`, `WORLD_SIZE`, and `RANK` into the workers.
*   **Backend:** Sets up the NCCL ring.

### 3. API Structure
*   `train_loop_per_worker()`: Your training logic.
*   `ScalingConfig()`: Hardware request (`num_workers`, `use_gpu`).
*   `TorchTrainer()`: The wrapper class.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Training Loop

We start with a pure PyTorch function.

#### 📁 `src/01_ray_starter.py`
```python
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
import ray.train.torch

def train_loop_per_worker(config):
    # 1. Setup Distributed Environment
    # Ray automatically calls torch.distributed.init_process_group()
    # rank = ray.train.get_context().get_world_rank()
    
    # 2. Prepare Data (Mock)
    batch_size = config.get("batch_size", 32)
    dataset = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset, labels),
        batch_size=batch_size
    )
    # Wraps DistributedSampler automatically
    dataloader = prepare_data_loader(dataloader) 

    # 3. Prepare Model
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # Wraps DDP automatically
    model = prepare_model(model) 

    # 4. Loop
    for epoch in range(5):
        for X, y in dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()
        
        # 5. Report Metrics (only from Rank 0 usually, handled by Ray)
        ray.train.report({"loss": loss.item(), "epoch": epoch})

# Driver Code
if __name__ == "__main__":
    ray.init()
    
    # Define scaling
    # Request 2 workers. Each needs 0 GPUs (CPU mode for now)
    scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"batch_size": 16},
        scaling_config=scaling_config
    )
    
    # Run!
    result = trainer.fit()
    print(f"Final Loss: {result.metrics['loss']}")
```

### 👨‍💻 Key Differences from Pure PyTorch

1.  **Init:** No `init_process_group("nccl")`. Ray does it.
2.  **Wrapping:** `prepare_model(model)` detects available GPUs and wraps `DistributedDataParallel`.
3.  **Reporting:** `ray.train.report` instead of `print`. This aggregates metrics to the Ray Dashboard/MLflow.

---

## 🔬 Lab Exercise: "GPU Scaling"

### Task
Scale to GPUs.
1.  Change `use_gpu=True` in `ScalingConfig`.
2.  If you have no GPUs, Ray raises `RuntimeError: No GPUs found`.
3.  **Workaround:** `ray.init(num_gpus=2)` (Simulation).
4.  **Note:** Simulation allows the *Scheduler* to work, but `torch.cuda` will fail inside the worker. Only do this if you actually have CUDA devices or mock the training loop to skip cuda calls.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Config Injection:** Pass hyperparameters via `train_loop_config`. No need for argparse parsing inside the function.
2.  **Elasticity:** If you use the `RayCluster` from Week 14, changing `num_workers=10` in Python triggers the K8s Autoscaler to fetch 10 GPU nodes instantly.
3.  **Fault Tolerance:** If a worker dies, `TorchTrainer` can be configured to restart the whole group or restore from checkpoint (Day 102).

### API Summary
```python
result = trainer.fit()
ray.train.get_context()
```

---

**Day 99 Complete** ✅

*Next: Day 100 - Adapting Standard DDP - Moving existing repos to Ray Train.*
