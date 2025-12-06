# Day 109: Fail Fast: ASHA and Schedulers
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** If a learning rate of 0.0001 produces 0% accuracy after 1 epoch, it won't magically work at epoch 50. **Schedulers** like ASHA allow you to aggressively terminate poor performers.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between a *Search Algorithm* (Optuna) and a *Trial Scheduler* (ASHA).
2.  **Configure** `ASHAScheduler` to prune trials based on intermediate metrics.
3.  **Explain** the "Rung" system in Successive Halving.
4.  **Implement** Population Based Training (PBT) to mutate hyperparameters dynamically.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. Successive Halving (The Hunger Games)
*   **Round 1:** Run 100 tributes (trials) for 1 epoch.
*   **Selection:** Kill the bottom 50%.
*   **Round 2:** Run survivors for 2 epochs.
*   **Result:** You spend compute resources proportional to the quality of the trial.
*   **Async (ASHA):** Does not wait for all 100 to finish. As soon as a trial finishes epoch 1, it checks if it's in the top percentiles of *finished* trials. If yes, continue.

### 2. PBT (Evolutionary)
Instead of just killing, PBT *exploits*.
*   If Trial A is losing and Trial B is winning...
*   Copy Trial B's weights to Trial A.
*   Mutate Trial A's Hyperparameters (e.g., perturb LR by 1.2x).
*   Resume Trial A.
*   **Result:** A schedule of hyperparameters (Static -> Dynamic).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: ASHA

#### 📁 `src/04_asha.py`
```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner, TuneConfig

def train_func(config):
    # Simulate training 100 steps
    for i in range(100):
        # Fake loss
        loss = (config["x"] - 0.5)**2 + (0.9 ** i)
        
        # Report metric AND step!
        tune.report({"loss": loss, "training_iteration": i})

# Setup Scheduler
scheduler = ASHAScheduler(
    time_attr="training_iteration",
    max_t=100, # Max epochs
    grace_period=5, # Run at least 5 epochs before killing
    reduction_factor=2 # Halve the population each rung
)

tuner = Tuner(
    train_func,
    param_space={"x": tune.uniform(0, 1)},
    tune_config=TuneConfig(
        metric="loss",
        mode="min",
        num_samples=50,
        scheduler=scheduler
    )
)

results = tuner.fit()
```

### 👨‍💻 Core Implementation: PBT

PBT requires Checkpointing to work (to copy weights).

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-4, 1e-1),
    }
)
```

---

## 🔬 Lab Exercise: "Resource Savings"

### Task
Measure efficiency.
1.  Run 50 trials without Scheduler (Full 100 epochs each). Total Epochs = 5000.
2.  Run 50 trials with ASHA.
    *   25 run 5 epochs.
    *   12 run 10 epochs.
    *   ...
    *   1 runs 100 epochs.
    *   Total Epochs = ~500.
3.  **Observation:** ASHA is 10x cheaper.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Reporting Frequency:** For ASHA to work, you must call `tune.report` frequently (e.g., every epoch).
2.  **Budget:** If using ASHA, increase `num_samples`. Since trials are cheap, you can afford to start 1000 of them.
3.  **Compatibility:** Can you combine Optuna + ASHA? Yes. Optuna suggests configurations, ASHA kills them if they perform poorly.

### API Summary
```python
ASHAScheduler(grace_period=1)
```

---

**Day 109 Complete** ✅

*Next: Day 110 - Tuning Distributed Trainers - Tune + Ray Train.*
