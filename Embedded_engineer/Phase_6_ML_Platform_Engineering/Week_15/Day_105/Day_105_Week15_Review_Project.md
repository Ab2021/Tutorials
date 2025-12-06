# Day 105: Week 15 Review & Project - Zero-to-Hero Ray Training Pipeline
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** We have learned the components: Trainer, Scaler, Data, Checkpoints, Logger. Now, we assemble the **Production Training Pipeline**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Orchestrate** an end-to-end distributed training job (Data Ingest -> Training -> Checkpointing -> Logging).
2.  **Debug** common distributed training failures (OOM, Network Timeout).
3.  **Evaluate** the scaling efficiency (Time to Accuracy) of Ray Train.

---

## 📚 Week 15 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 99 | Ray Train Basics | "No more SSH loops. Just Python." |
| 100 | Porting DDP | "`prepare_model` does the DDP wrapping for me." |
| 101 | Ray Data | "I can stream terabytes of images without using RAM." |
| 102 | Checkpointing | "My Spot Instance died, but training resumed instantly." |
| 103 | DeepSpeed | "I can train models larger than GPU memory." |
| 104 | Logging | "One W&B project, 100 workers, clear charts." |

---

## 🏗️ Final Project: "CifarScale"

### Scenario
We need to train a ResNet-18 on CIFAR-10.
*   **Challenge 1:** Dataset is small, but treat it like it's 1TB (Use Streaming).
*   **Challenge 2:** Cluster is unstable (Simulate Preemption).
*   **Challenge 3:** We need accurate logs.

### Step 1: Data Pipeline (Streaming)

#### 📁 `project/data.py`
```python
import ray
import torchvision.transforms as transforms
import torch
import numpy as np

def get_datasets():
    # Load raw data from Hub/S3
    ds = ray.data.read_images("s3://anonymous@air-example-data/cifar-10/images")
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def preprocess(batch):
        # Ray processes numpy arrays or pandas DFs
        # Convert to torch for transform, then back?
        # Simpler: Just resize/normalize raw bytes/arrays
        batch["image"] = [transform(x) for x in batch["image"]]
        return batch

    ds = ds.map_batches(preprocess, batch_format="numpy")
    return ds
```

### Step 2: Training Logic (Resilient)

#### 📁 `project/train.py`
```python
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from ray.train import Checkpoint, report, get_checkpoint, get_context
from ray.train.torch import prepare_model, prepare_data_loader

def train_func(config):
    # 1. Setup
    model = resnet18(num_classes=10)
    model = prepare_model(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Dataset Shard
    # Use ray.train.get_dataset_shard("train")
    shard = ray.train.get_dataset_shard("train")
    iterator = shard.iter_torch_batches(batch_size=64, dtypes=torch.float32)

    # 3. Checkpoint Recovery
    start_step = 0
    ckpt = get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as cdir:
            model.load_state_dict(torch.load(f"{cdir}/model.pt"))
            start_step = config.get("step", 0)

    # 4. Loop
    step = start_step
    for batch in iterator:
        step += 1
        X, y = batch["image"], batch["label"] 
        # Note: image reading from S3 example above might not have labels 
        # For simplicity, assume labeled dataset
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # 5. Report & Save (Every 100 steps)
        if step % 100 == 0:
            with tempfile.TemporaryDirectory() as temp_dir:
                torch.save(model.state_dict(), f"{temp_dir}/model.pt")
                report(
                    {"loss": loss.item(), "step": step},
                    checkpoint=Checkpoint.from_directory(temp_dir)
                )
```

### Step 3: The Driver

#### 📁 `project/main.py`
```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, FailureConfig
from project.data import get_datasets

if __name__ == "__main__":
    ray.init()
    
    ds = get_datasets()
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        datasets={"train": ds},
        run_config=RunConfig(
            storage_path="/mnt/cluster_storage/cifar_runs",
            failure_config=FailureConfig(max_failures=3)
        )
    )
    
    result = trainer.fit()
    print(f"Best Checkpoint: {result.best_checkpoints[0]}")
```

---

## 🔬 Lab Exercise: "Chaos Monkey"

### Task
You are the Chaos Monkey.
1.  Start the training.
2.  Open Ray Dashboard. Find a Worker actor.
3.  Kill the process (PID) manually.
4.  **Observation:** Ray detects actor death. Re-schedules the actor. Driver waits. Training resumes.

---

## 📝 Success Criteria
1.  **Distributed:** `num_workers > 1` works and speedup is observed vs 1 worker (training time per batch decreases or throughput increases).
2.  **Streaming:** The memory usage remains flat even if the dataset is 100GB.
3.  **Resilience:** The job finishes successfully even if you kill a worker.

---

**Week 15 Complete** ✅

*Next Week: Week 16 - Ray Tune - The automated hunt for the best hyperparameters.*
