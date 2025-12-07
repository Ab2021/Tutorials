# Days 99-105: Week 15 - Ray Train Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Quick Reference Labs

### Day 99: Ray Train Basics
```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

trainer = TorchTrainer(
    train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
result = trainer.fit()
```

### Day 100: PyTorch DDP
```python
def train_loop_per_worker():
    model = train.torch.prepare_model(model)
    dataloader = train.torch.prepare_data_loader(dataloader)
```

### Day 101: Ray Data Integration
```python
dataset = ray.data.read_parquet("s3://...")
trainer = TorchTrainer(
    train_loop,
    datasets={"train": dataset},
)
```

### Day 102: Checkpointing
```python
from ray.train import Checkpoint

checkpoint = Checkpoint.from_directory("/tmp/model")
train.report({"loss": loss}, checkpoint=checkpoint)
```

### Day 103: DeepSpeed
```python
from ray.train.torch import TorchTrainer
from ray.train import RunConfig

trainer = TorchTrainer(
    train_func,
    train_loop_config={"deepspeed": ds_config},
)
```

### Day 104: Logging
```python
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

run_config = RunConfig(
    callbacks=[WandbLoggerCallback(project="my-project")]
)
```

### Day 105: Week 15 Project
```python
# Distributed training pipeline
# 1. Ray Data for preprocessing
# 2. TorchTrainer for training
# 3. Checkpointing to S3
# 4. W&B logging
```

---

## 📝 Week 15 Summary
| Day | Topic |
|-----|-------|
| 99 | Train basics |
| 100 | DDP |
| 101 | Ray Data |
| 102 | Checkpoints |
| 103 | DeepSpeed |
| 104 | Logging |
| 105 | Project |
