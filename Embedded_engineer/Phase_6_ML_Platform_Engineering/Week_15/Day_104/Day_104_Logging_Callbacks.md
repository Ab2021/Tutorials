# Day 104: The Dashboard: Logging & Callbacks
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 15: Ray Train - Distributed Training

---

> **🎯 Focus Area:** A training run without logs didn't happen. Learn how Ray aggregates metrics from distributed workers and pipes them to **TensorBoard**, **MLflow**, and **Weights & Biases**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Worker Logs (stdout) and Driver Logs.
2.  **Configure** `RunConfig` to enable TensorBoard logging.
3.  **Integrate** `WandbLoggerCallback` to visualize loss curves.
4.  **Inject** API Keys into workers securely using `runtime_env`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Internet Connection (for W&B).

### Software Environment
- `pip install wandb mlflow`.
- W&B Account.

---

## 📖 Theoretical Foundation

### 1. The Metric Flow
1.  **Worker:** Calls `ray.train.report({"loss": 0.5})`.
2.  **Ray Backend:** Serializes metric. Sends to Driver Actor (Trainer).
3.  **Trainer:** Aggregates (e.g., if multiple workers report, it can average or just take Rank 0).
4.  **Callback:** `WandbLoggerCallback` running on the Driver pushes data to Cloud.

### 2. Logging Hierarchy
*   **System Logs:** `raylet.out`, `gcs_server.out`. (Infrastructure health).
*   **App Logs:** `print("hello")` from workers. Streamed to Driver stdout.
*   **Artifacts:** Checkpoints, Model weights.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: W&B Integration

#### 📁 `src/05_logging.py`
```python
import ray
import os
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train.wandb import WandbLoggerCallback

def train_func(config):
    # ... training loop ...
    for i in range(10):
        ray.train.report({"loss": 10-i, "accuracy": i*0.1})

# Driver
# We need to pass the API key to the Driver process (where Callback runs)
runtime_env = {
    "env_vars": {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}
}
ray.init(runtime_env=runtime_env)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=2),
    run_config=RunConfig(
        name="wandb_run_01",
        callbacks=[
            WandbLoggerCallback(
                project="ray_training_demo",
                api_key=os.environ["WANDB_API_KEY"]
            )
        ]
    )
)

trainer.fit()
```

### 👨‍💻 Core Implementation: MLflow

MLflow usually requires setting the Tracking URI.

```python
from ray.train.mlflow import MLflowLoggerCallback

callback = MLflowLoggerCallback(
    tracking_uri="http://tracking-server:5000",
    experiment_name="ray_exp_1",
    save_artifact=True
)
```

---

## 🔬 Lab Exercise: "Rank 0 filtering"

### Task
Metrics Noise.
1.  If all 100 workers report "loss", Ray aggregates them?
2.  Usually, we only want Rank 0 to report "Global Loss" if we sync manually.
3.  Or, we let Ray aggregate.
4.  **Experiment:** In loop, add `if ray.train.get_context().get_world_rank() == 0: ray.train.report(...)`.
5.  **Observation:** W&B charts become cleaner (1 point per step instead of 100). Note that `report` serves a dual purpose: checkpointing synchronization AND metric logging. If you skip reporting on other ranks, make sure you aren't blocking checkpointing logic. Ray Train handles synchronization automatically on `report`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Callbacks Run on Driver:** The W&B logic runs on the Head Node (Driver). The workers just send dictionaries. This is efficient.
2.  **API Keys:** Never hardcode keys. Use `runtime_env` or K8s Secrets mapped to Env Vars.
3.  **TensorBoard:** Enabled by default in Ray. Outputs to `~/ray_results`. View with `tensorboard --logdir ~/ray_results`.

### API Summary
```python
ray.train.report()
```

---

**Day 104 Complete** ✅

*Next: Day 105 - Week 15 Review & Project - Zero-to-Hero Ray Training Pipeline.*
