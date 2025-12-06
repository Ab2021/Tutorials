# Day 107: The Experiment Manager: Advanced Tune API
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** Tuning a large model takes days. You need a way to **Checkpoint** intermediate results so that if your cluster crashes, you don't lose the progress of 50 concurrent trials.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** the `tune.Trainable` class interface (`setup`, `step`, `save_checkpoint`).
2.  **Report** intermediate metrics to enable schedulers to see progress curves.
3.  **Resume** an interrupted tuning experiment using `Tuner.restore()`.
4.  **Configure** `RunConfig` for proper storage of trial artifacts.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Iterative Training Concept
HPO is not just "Input -> Output". It is "Input -> Epoch 1 -> Epoch 2 ...".
By reporting metrics at every epoch, we allow Ray Tune to:
*   **Visualize** learning curves in real-time.
*   **Kill** trials that are learning too slowly (ASHA - Day 109).

### 2. The Trainable Lifecycle
*   `setup(config)`: Init model/optimizer.
*   `step()`: Train 1 epoch. Return metrics dict.
*   `save_checkpoint(dir)`: Save state.
*   `load_checkpoint(dir)`: Restore state.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Class-Based Trainable

#### 📁 `src/02_class_api.py`
```python
import ray
from ray import tune
from ray.tune import Tuner
import os
import torch

class MyTrainable(tune.Trainable):
    def setup(self, config):
        # Called once at start
        self.lr = config["lr"]
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.epoch = 0

    def step(self):
        # Simulate training 1 epoch
        self.epoch += 1
        loss = (0.1 / self.lr) * (0.9 ** self.epoch) # Fake loss curve
        
        # Return metrics
        return {"loss": loss, "epoch": self.epoch}

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(self.model.state_dict(), path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        self.model.load_state_dict(torch.load(path))

# Run it
tuner = Tuner(
    MyTrainable,
    param_space={"lr": tune.uniform(0.001, 0.1)},
    run_config=ray.train.RunConfig(
        stop={"training_iteration": 10}, # Stop each trial after 10 steps
        name="my_experiment",
        storage_path="/tmp/ray_results"
    )
)

results = tuner.fit()
```

### 👨‍💻 Core Implementation: Resuming

If you interrupt the script (Ctrl+C), run this:

```python
if Tuner.can_restore("/tmp/ray_results/my_experiment"):
    tuner = Tuner.restore(
        "/tmp/ray_results/my_experiment",
        trainable=MyTrainable,
        col
        resume_unfinished=True,
        restart_errored=True
    )
    tuner.fit()
```

---

## 🔬 Lab Exercise: "The Interrupted Hunt"

### Task
Simulate Failure.
1.  Add `time.sleep(1)` in `step()`.
2.  Run experiment.
3.  Kill script halfway.
4.  Run Restore script.
5.  **Observation:** It skips completed trials. It loads the checkpoints of running trials and continues from `epoch=N`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Function vs Class:** Most users prefer the Function API (`session.report`) because it's less verbose. Ray Tune converts functions to Trainables internally. However, the Class API is explicit about lifecycle.
2.  **Storage:** Trials produce a lot of data. Ensure `storage_path` has space.
3.  **Output:** Check `~/ray_results/my_experiment/trial_XXXX/result.json` for the raw CSV/JSON logs of the metrics.

### API Summary
```python
tune.with_resources(trainable, {"cpu": 1, "gpu": 0.5})
```

---

**Day 107 Complete** ✅

*Next: Day 108 - Smart Search - BayesOpt and HyperOpt.*
