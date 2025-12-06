# Day 108: The Smart Search: Bayesian Optimization
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** Random Search shoots in the dark. **Bayesian Optimization** builds a map of the terrain, guessing where the "Global Minimum" is likely to be.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Explore vs Exploit trade-off in HPO.
2.  **Integrate** `OptunaSearch` into Ray Tune.
3.  **Compare** the convergence speed of Random Search vs BayesOpt.
4.  **Handle** Conditional Hyperparameters (e.g., if `optimizer=adam` tune `beta1`, if `sgd` tune `momentum`).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install optuna`.

---

## 📖 Theoretical Foundation

### 1. The Surrogate Model
BayesOpt maintains a "Surrogate Model" (e.g., Gaussian Process or Tree Parzen Estimator - TPE).
*   **Input:** History of trials `[(config1, loss1), (config2, loss2)]`.
*   **Output:** Predictive distribution of Loss for untried configs.
*   **Benefit:** Sample promising areas more densely.

### 2. Integration
Ray Tune doesn't implement these algorithms. It provides a wrapper `search_alg` that talks to Optuna/HyperOpt. Ray manages the *Distributed Execution* (Parallelism), Optuna manages the *math*.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Optuna Integration

#### 📁 `src/03_bayesopt.py`
```python
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Tuner, TuneConfig

def objective(config):
    # Min at x=0, y=0
    score = config["x"]**2 + config["y"]**2
    return {"score": score}

# 1. Provide Initial Points (Help the algo start)
# Optuna will start with these before doing its magic
points_to_evaluate = [
    {"x": 10, "y": 10}, 
    {"x": -10, "y": -10}
]

# 2. Setup Algo
optuna_search = OptunaSearch(
    metric="score",
    mode="min",
    points_to_evaluate=points_to_evaluate
)

# 3. Param Space
# Note: For Search Algos, we often define space inside the algo or use standard tune.*
# Optuna supports Define-by-Run, but Tune creates the bridge.
space = {
    "x": tune.uniform(-100, 100),
    "y": tune.uniform(-100, 100)
}

tuner = Tuner(
    objective,
    tune_config=TuneConfig(
        search_alg=optuna_search,
        num_samples=50 # Number of iterations
    ),
    param_space=space
)

results = tuner.fit()
print("Best config:", results.get_best_result().config)
```

### 👨‍💻 Core Implementation: Conditional Parameters

Conditional spaces are tricky. "momentum" only matters if "optimizer" is SGD.

```python
# Tune's native 'choice' handles this, but Optuna handles it better via Define-by-Run.
# In Ray Tune, simply ignoring the parameter in 'objective' handles it, 
# although the search space might look slightly inefficient.
```

---

## 🔬 Lab Exercise: "Convergence Race"

### Task
Benchmarking.
1.  Run Random Search (50 samples). Record Best Score.
2.  Run Optuna Search (50 samples). Record Best Score.
3.  **Observation:** On convex problems (like $x^2 + y^2$), Optuna converges to $0.0001$ much faster than Random Search ($0.5$).
4.  **Note:** BayesOpt adds overhead (calculating next point takes time). For very fast functions (<1s), Random might be faster overall. For DL Training (Hours), BayesOpt is essential.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Concurrency:** Optuna is naturally sequential. Limitation: If you run 50 trials in *parallel*, Optuna effectively becomes Random Search for the first batch because it has no history to update the model. It works best with `max_concurrent_trials` set to a small number relative to `num_samples`.
2.  **Define-by-Run:** Optuna's strength.
3.  **Warm Start:** Always provide `points_to_evaluate` with known good configs (defaults).

### API Summary
```python
OptunaSearch(metric="loss", mode="min")
```

---

**Day 108 Complete** ✅

*Next: Day 109 - Optimal Resource Allocation - ASHA and Schedulers.*
