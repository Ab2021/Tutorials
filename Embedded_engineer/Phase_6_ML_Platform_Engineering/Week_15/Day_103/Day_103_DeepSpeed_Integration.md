# Day 103: Going Massive: Ray Train + DeepSpeed
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** DDP limits model size to single-GPU memory. **DeepSpeed ZeRO** shards the model across the entire cluster. Ray makes launching DeepSpeed jobs trivial.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the benefits of ZeRO-2 and ZeRO-3 over standard DDP.
2.  **Write** a DeepSpeed configuration JSON (Optimizer, Scheduler, FP16).
3.  **Deploy** a `TorchTrainer` configured for DeepSpeed.
4.  **Verify** memory savings using `nvidia-smi` inside the Ray Dashboard.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- At least 2 GPUs (simulated if needed).

### Software Environment
- `pip install deepspeed`.

---

## 📖 Theoretical Foundation

### 1. ZeRO Scaling (Recap)
*   **ZeRO-1:** Shard Optimize States. (4x Memory Reduc).
*   **ZeRO-2:** Shard Gradients. (8x Memory Reduc).
*   **ZeRO-3:** Shard Parameters. (Linear Memory Reduc with N GPUs). Allows training 175B param models.

### 2. Ray + DeepSpeed
Ray acts as the launcher.
Instead of `deepspeed --num_gpus=4 train.py`, we use `TorchTrainer`.
Ray handles the environment setup (`RANK`, `WORLD_SIZE`), so DeepSpeed initializes correctly inside the loop.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Training Loop

Note: `prepare_model` doesn't support DeepSpeed directly in the same way. We use `deepspeed.initialize`.

#### 📁 `src/04_deepspeed_ray.py`
```python
import ray
import deepspeed
import torch
import torch.nn as nn
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    # 1. Define Model
    model = nn.Linear(100, 10).cuda() # In real life, huge Transformer
    
    # 2. Init DeepSpeed
    # Ray automatically sets env vars for DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config["ds_config"]
    )
    
    # 3. Create Dummy Data
    loader = ... # (Use prepare_data_loader still!)
    
    # 4. Loop
    for epoch in range(5):
        for X, y in loader:
            X, y = X.cuda(), y.cuda() # Move to device
            
            # DeepSpeed handles forward/backward/step
            outputs = model_engine(X)
            loss = nn.MSELoss()(outputs, y)
            
            model_engine.backward(loss)
            model_engine.step()

# Configuration
ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2, # Shard Gradients & Optimizer
    }
}

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"ds_config": ds_config},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True)
)

trainer.fit()
```

### 👨‍💻 Alternate Way: PyTorch Lightning + DeepSpeed on Ray

If using PL, it is easier.
```python
from ray.train.lightning import RayDeepSpeedStrategy

strategy = RayDeepSpeedStrategy(stage=3)
trainer = pl.Trainer(strategy=strategy, ...)
```

---

## 🔬 Lab Exercise: "OOM Kill"

### Task
Demonstrate ZeRO.
1.  Define a `nn.Linear(40000, 40000)`. ~6GB weights.
2.  Run with DDP (ZeRO Stage 0) on a 4GB GPU (Simulation: reduce GPU limit).
3.  **Result:** OOM.
4.  Run with ZeRO Stage 3 (Sharding).
5.  **Result:** Fits. The model weights are split.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Orchestration:** Ray replaces the `deepspeed` launcher. You don't need `hostfiles`. Ray allocates nodes dynamically.
2.  **Config:** DeepSpeed config is passed via `train_loop_config` dictionary.
3.  **Integration:** Works seamlessly with Ray Data for loading.

### API Summary
```python
deepspeed.initialize()
```

---

**Day 103 Complete** ✅

*Next: Day 104 - Logging & Callbacks - W&B Integration.*
