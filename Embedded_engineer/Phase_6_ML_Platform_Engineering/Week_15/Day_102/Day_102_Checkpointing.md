# Day 102: Immortality: Checkpointing & Recovery
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** Losing 3 days of Llama-2 training because a Spot Instance terminated is unacceptable. Learn to use **Ray Checkpoints** to auto-resume training exactly where it left off.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** checkpoint saving logic inside the training loop.
2.  **Configure** `RunConfig` with `FailureConfig` to enable auto-retry.
3.  **Restore** a `TorchTrainer` from a previous run path.
4.  **Describe** how Ray manages distributed checkpoint storage (Cloud vs Shared File System).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Reporting API
In Ray Train, you don't save to disk directly. You:
1.  Save `state_dict` to a temp directory.
2.  Create `ray.train.Checkpoint` object.
3.  Call `ray.train.report(metrics, checkpoint=ckpt)`.
4.  Ray moves the checkpoint to persistent storage (S3/HDFS).

### 2. Failure Handling
*   **Worker Crash:** One node dies.
*   **Action:** Ray tears down all other workers.
*   **Restart:** Ray provisions a replacement node.
*   **Resume:** Ray starts workers. Passes the *latest checkpoint* into `train_func`.
*   **User Logic:** In `train_func`, check `get_checkpoint()`. If exists, `model.load_state_dict()`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Reliable Training Loop

#### 📁 `src/03_checkpointing.py`
```python
import ray
from ray.train import Checkpoint, report, get_checkpoint
import torch
import tempfile
import os

def train_func(config):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 1. Recovery Logic
    checkpoint = get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state = torch.load(os.path.join(checkpoint_dir, "model.pt"))
            model.load_state_dict(model_state)
            start_epoch = config.get("last_epoch", 0) + 1 # Use metadata if stored
            print(f"Resuming from epoch {start_epoch}")

    # 2. Loop
    for epoch in range(start_epoch, 10):
        # ... training step ...
        
        # 3. Save Logic
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(model.state_dict(), os.path.join(temp_dir, "model.pt"))
            
            # Create object
            checkpoint_obj = Checkpoint.from_directory(temp_dir)
            
            # Report to Ray
            report({"loss": 0.5, "last_epoch": epoch}, checkpoint=checkpoint_obj)
```

### 👨‍💻 Core Implementation: Configuration

```python
from ray.train import RunConfig, FailureConfig

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=2),
    run_config=RunConfig(
        name="my_resilient_job",
        storage_path="/mnt/cluster_storage", # Or "s3://my-bucket/ray_runs"
        failure_config=FailureConfig(max_failures=3) # Retry 3 times
    )
)
```

---

## 🔬 Lab Exercise: "The Plug Pull"

### Task
Simulate Crash.
1.  Add `if epoch == 5 and rank == 0: raise RuntimeError("Hardware Failure")` to loop.
2.  Run Trainer.
3.  **Observation:**
    *   Epoch 0-4 run.
    *   Crash happens.
    *   Ray Logs: "Worker crashed. Retrying 1/3".
    *   Workers restart.
    *   "Resuming from epoch 5".
    *   Training completes.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Storage Path:** For multi-node training, `storage_path` MUST be a shared location (S3, GCS, NFS). If you use local disk, other nodes can't download the checkpoint to recover.
2.  **Frequency:** Don't checkpoint every batch. It slows down training (upload bandwidth). Checkpoint every epoch or every N steps.
3.  **Keep Policy:** Configure `CheckpointConfig(num_to_keep=2)` to delete old checkpoints and save space.

### API Summary
```python
checkpoint.as_directory()
checkpoint.to_directory()
```

---

**Day 102 Complete** ✅

*Next: Day 103 - DeepSpeed Integration - Leveraging Ray for Massive Model Training.*
