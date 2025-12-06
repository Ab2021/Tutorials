# Day 106: The Black Box: Introduction to Hyperparameter Optimization
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** Picking the Learning Rate by hand is "Graduate Student Descent". It is not scalable. **Ray Tune** automates the search for the optimal configuration of your model.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Model Parameters (Weights) and Hyperparameters (Config).
2.  **Explain** why Grid Search suffers from the "Curse of Dimensionality".
3.  **Use** `ray.tune` to define a Search Space (`loguniform`, `choice`).
4.  **Execute** a basic Random Search optimization on a synthetic function.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install "ray[tune]"`

---

## 📖 Theoretical Foundation

### 1. The Optimization Problem
We want to find $x$ that minimizes $f(x)$.
*   $f(x)$ is the Validation Loss of a neural network trained with config $x$.
*   Calculating $f(x)$ is expensive (takes hours to train).
*   We cannot calculate gradients of $f$ with respect to $x$ (it's a black box).

### 2. Search Strategies
*   **Grid Search:** `lr in [0.1, 0.01]`, `batch in [32, 64]`. Checks $2 \times 2 = 4$ points. Scales exponentially ($N^D$).
*   **Random Search:** Sample `lr` from distribution. Superior because not all hyperparameters are equally important.
*   **Bayesian:** Use past results to predict where to look next (Day 108).

### 3. Ray Tune Architecture
*   **Tuner:** The orchestrator.
*   **Trial:** One execution of $f(x)$ with a specific config.
*   **Search Algo:** Suggests new configs.
*   **Scheduler:** Stops bad trials early (Early Stopping).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Objective Function

We define a fast function to optimize (Training Simulation).

#### 📁 `src/01_basics.py`
```python
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig

# 1. Define Objective
# Config is a dictionary sampled from the search space
def objective(config):
    x = config["x"]
    y = config["y"]
    
    # We want to minimize score
    # Let's say optimal is at x=0.5, y=0.5
    score = (x - 0.5)**2 + (y - 0.5)**2
    
    # Report metric to Tune
    return {"score": score}

# 2. Define Search Space
search_space = {
    "x": tune.uniform(0, 1), # Float between 0 and 1
    "y": tune.uniform(0, 1),
    "method": tune.choice(["fast", "slow"]) # Categorical
}

# 3. Setup Tuner
tuner = Tuner(
    trainable=objective,
    param_space=search_space,
    tune_config=TuneConfig(
        metric="score",
        mode="min",
        num_samples=20, # Try 20 random samples
    )
)

# 4. Run
results = tuner.fit()

# 5. Analyze
print("Best Config:", results.get_best_result().config)
print("Best Score:", results.get_best_result().metrics["score"])
```

### 👨‍💻 Lab: Parallel Execution

Ray Tune automatically parallelizes trials.
1.  Add `import time; time.sleep(1)` to `objective`.
2.  If you have 8 CPUs and `num_samples=20`:
3.  Ray will launch 8 trials in parallel.
4.  Total time: $20 / 8 \approx 3$ seconds (instead of 20s).

---

## 🔬 Lab Exercise: "Grid Search Fail"

### Task
Compare Grid vs Random.
1.  Grid: `x` in `[0.1, 0.2, ... 1.0]` (10 points). `y` in `[0.1...1.0]` (10 points). Total 100 trials.
2.  Random: 20 trials.
3.  **Observation:** Often, random search finds a better result (e.g., `x=0.48, y=0.52`) with 5x fewer trials because it explores the continuous space rather than fixed grid points.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Search Space:** Defining a good search space is an art. Use `loguniform` for Learning Rate (sample $10^{-4}$ as often as $10^{-2}$).
2.  **Resource Allocation:** By default, a trial uses 1 CPU. You can specify `resources={"cpu": 2, "gpu": 1}` to tune GPU models.
3.  **Reproducibility:** Set `tune.run(..., set_seed=42)`? No, usually Tuner doesn't have a global seed easily. You handle seeding inside the objective if needed, but randomness is part of HPO.

### API Summary
```python
tune.uniform(min, max)
tune.loguniform(min, max)
tune.choice(list)
tuner.fit()
```

---

**Day 106 Complete** ✅

*Next: Day 107 - The Ray Tune API - Config, Checkpoints, and Resumption.*
