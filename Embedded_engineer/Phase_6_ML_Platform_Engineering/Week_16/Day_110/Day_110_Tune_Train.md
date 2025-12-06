# Day 110: Tuning the Cluster: Train + Tune
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** We have learned Distributed Training (Week 15) and HPO (Week 16). Now we combine them. Each "Trial" in our HPO search will be a full Distributed Data Parallel job over multiple GPUs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Pass** variable hyperparameters from `Tuner` into `TorchTrainer`.
2.  **Calculate** cluster capacity: (Num Nodes * GPUs per Node) / (GPUs per Trial) = Concurrency.
3.  **Configure** `RayTrainable` to handle lifecycle management correctly.
4.  **Execute** a multi-node hyperparameter sweep.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Cluster with multiple GPUs (e.g., 4 GPUs).

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Wrapper Pattern
Ray Tune wraps Ray Train.
*   **Tune:** "I want to try `lr=0.01`". Spawns a Trial Actor.
*   **Trial Actor:** Instantiates `TorchTrainer(scaling_config={workers: 2})`.
*   **Ray Train:** Requests 2 GPU actors from Ray Scheduler.
*   **Loop:** Runs. Reports metrics back to Trial Actor -> Tune.

### 2. Resource Math
*   Total Cluster: 8 GPUs.
*   Scaling Config: 2 Workers (2 GPUs per trial).
*   Tune can run: $8 / 2 = 4$ concurrent trials.
*   Optuna suggests 4 configs. 4 Distributed Jobs start.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Integration

#### 📁 `src/05_tune_train.py`
```python
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.tune import Tuner, TuneConfig

# 1. The Training Function (Accepts config)
def train_func(config):
    lr = config["lr"]
    batch_size = config["batch_size"]
    # ... Standard Ray Train logic (Day 99) ...
    # ray.train.report({"loss": loss})

# 2. Define Param Space
param_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64]),
    # We can also tune Training Resources!
    # "train_loop_config": ...
}

# 3. Setup Tuner with TorchTrainer
# Note: We pass the Class, not the instance, usually?
# Actually, nicely, Tuner accepts a Trainable creator.
# But for Ray Train, we pass the TorchTrainer directly.

tuner = Tuner(
    trainable=TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        run_config=RunConfig(stop={"training_iteration": 5})
    ),
    param_space={
        "train_loop_config": param_space # Inject space into config
    },
    tune_config=TuneConfig(
        metric="loss",
        mode="min",
        num_samples=10
    )
)

results = tuner.fit()
```

### 👨‍💻 Performance Tuning
If you have 4 GPUs and `num_workers=2`, you run 2 concurrent trials.
If you set `num_samples=100`, it will take 50 rounds.

---

## 🔬 Lab Exercise: "Pending Trials"

### Task
Oversubscribe.
1.  Cluster: 4 CPUs.
2.  Trainer: `num_workers=4` (Takes 4 CPUs).
3.  Tune: `num_samples=2`.
4.  **Observation:**
    *   Trial 1 starts. Consumes 4 CPUs. Status: RUNNING.
    *   Trial 2 starts. Request 4 CPUs. Available: 0. Status: PENDING.
    *   Trial 1 finishes. Resources released.
    *   Trial 2 starts.
5.  **Insight:** Ray manages the queue. You don't need to manually schedule batch jobs.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Complexity:** Debugging this stack is hard. Is it the Model? The Trainer? The Tuner? Test each layer independently first (Day 99 first, then Day 106).
2.  **Resources:** Be careful not to starve the Head Node. Leave CPUS for the Tuner orchestrator.
3.  **Output:** Tune results will contain the *best checkpoint* from the *best trial*.

### API Summary
```python
Tuner(trainable=TorchTrainer(...))
```

---

**Day 110 Complete** ✅

*Next: Day 111 - Analysis - Making sense of 1000 trials.*
