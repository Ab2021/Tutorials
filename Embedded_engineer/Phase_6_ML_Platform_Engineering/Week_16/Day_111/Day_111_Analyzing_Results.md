# Day 111: The Post-Mortem: Analyzing Tune Results
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 16: Ray Tune - Hyperparameter Optimization

---

> **🎯 Focus Area:** Burning GPU hours is useless if you don't learn from the results. Learn how to use the **ResultGrid** and Pandas to correlate "High Learning Rate" with "Exploding Gradients".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Extract** a Pandas DataFrame from the `ResultGrid`.
2.  **Filter** failed trials (`error != NaN`).
3.  **Plot** Parallel Coordinate charts to visualize parameter sensitivity.
4.  **Retrieve** the best checkpoint from persistent storage.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install pandas matplotlib seaborn`.

---

## 📖 Theoretical Foundation

### 1. The Result Grid
When `Tuner.fit()` finishes, it returns a `ResultGrid`.
*   It handles the mess of distributed logs.
*   It aggregates `metrics.json` from 100 subfolders.

### 2. Sensitivity Analysis
*   **Correlation:** If `correlation(lr, loss)` is high, LR is important.
*   **Parallel Coordinates:** A line chart where each vertical axis is a parameter. Helps indentify "Sweet Spots" (e.g., "Good performance only happens when Batch < 32 AND LR > 0.01").

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Parsing Results

#### 📁 `src/06_analysis.py`
```python
import ray
from ray.tune import ResultGrid

# 1. Restore the grid from path (if script finished previously)
experiment_path = "/tmp/ray_results/my_experiment"
# Or just use the object returned by fit()
# results = tuner.fit()

try:
    results = ResultGrid(tuple(ray.tune.ExperimentAnalysis(experiment_path).trials)) 
    # Note: Easier API is usually via Tuner.restore().get_results()
except:
    print("Run Day 107 first to generate data.")
    exit()

# 2. Get DataFrame
df = results.get_dataframe()

# 3. Clean and Sort
# Columns are like 'config/lr', 'custom_metrics/loss'
target_metric = "loss"
if target_metric not in df.columns:
    print("Available columns:", df.columns)
else:
    best_df = df.sort_values(target_metric, ascending=True)
    print("Top 5 Configs:")
    print(best_df[["config/lr", target_metric]].head(5))

# 4. Filter Errors
failed = df[df["error"].notna()]
print(f"Number of failed trials: {len(failed)}")
```

### 👨‍💻 Core Implementation: Parallel Coordinates

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_parcoords(df, metrics, params):
    # Filter only relevant columns
    subset = df[params + metrics].dropna()
    
    # Normalize for plotting
    for col in subset.columns:
        subset[col] = (subset[col] - subset[col].min()) / (subset[col].max() - subset[col].min())
    
    plt.figure(figsize=(10, 6))
    pd.plotting.parallel_coordinates(subset, "loss", colormap='viridis') # 'loss' is class column? 
    # Standard parallel_coordinates expects a class column for color.
    # Better to use plotly for real interactivity.
```

---

## 🔬 Lab Exercise: "The Cliff"

### Task
Visualize Non-Convexity.
1.  Run a sweep of LR from $10^{-6}$ to $10$.
2.  Plot Log(LR) vs Loss.
3.  **Observation:** You will see a U-shape (or a Cliff).
    *   Too small: Loss decreases too slow (High final loss).
    *   Optimal: Minimum loss.
    *   Too large: Divergence (Massive loss or NaN).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Artifacts:** The DataFrame gives you the numbers, but `result.checkpoint` gives you the *model*. Always ensure you can load the best model (Day 113).
2.  **Reproducibility:** The `ResultGrid` stores the `config`. You can re-run the exact same trial.
3.  **TensorBoard:** Tune automatically provides HPARAMS tab support in TensorBoard. `tensorboard --logdir ...`.

### API Summary
```python
results.get_best_result(metric="loss", mode="min")
```

---

**Day 111 Complete** ✅

*Next: Day 112 - Week 16 Review & Project - Auto-Tuning a Transformer.*
