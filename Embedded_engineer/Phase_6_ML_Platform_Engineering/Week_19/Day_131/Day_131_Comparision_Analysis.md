# Day 131: The Analyst's Workbench: Experiment Comparison
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** You ran 50 experiments. Run 12 has the lowest loss, but Run 34 has the best accuracy on the sub-group "Females over 50". Use **Pandas** and **Statistical Tests** to prove your model is actually better, not just lucky.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Search** runs programmatically using the MLflow Search Syntax (SQL-like).
2.  **Export** run data to a Pandas DataFrame for custom analysis.
3.  **Perform** significance testing (T-Test) between two model versions.
4.  **Visualize** Pareto Frontiers (Accuracy vs Latency) using Matplotlib.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install mlflow pandas scipy matplotlib seaborn`.

---

## 📖 Theoretical Foundation

### 1. The Search API
MLflow stores runs in a database.
*   **Filter String:** `params.lr < 0.01 and metrics.accuracy > 0.9`.
*   **Ordering:** `order_by=["metrics.accuracy DESC"]`.

### 2. Statistical Significance
*   **Null Hypothesis ($H_0$):** Model A and Model B have the same performance.
*   **P-Value:** If $< 0.05$, we reject $H_0$. Model B is significantly different.
*   **Why?** A generic "AVG Accuracy" metric hides variance. Cross-Validation scores give us a distribution.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Search and Export

#### 📁 `src/06_analysis.py`
```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Phase6_Week19_Demo"
exp = mlflow.get_experiment_by_name(experiment_name)

# 1. Search Runs
# Syntax: "attribute.name = 'value' AND metrics.name > 0"
df = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.loss < 0.5 AND params.learning_rate > 0.001",
    order_by=["metrics.loss ASC"]
)

# 2. Inspect DataFrame
print("Columns:", df.columns)
# 'run_id', 'status', 'params.learning_rate', 'metrics.loss', ...

# 3. Clean Data
# Convert prefix columns to normal names
df["lr"] = df["params.learning_rate"].astype(float)
df["loss"] = df["metrics.loss"]
df["accuracy"] = df["metrics.accuracy"]

print(df[["run_id", "lr", "loss", "accuracy"]].head())

# 4. Visualization: Boxplot of Loss vs LR
plt.figure(figsize=(10, 6))
sns.boxplot(x="lr", y="loss", data=df)
plt.title("Loss Distribution by Learning Rate")
plt.savefig("analysis_boxplot.png")
```

### 👨‍💻 Core Implementation: Significance Testing

Did the new architecture actually improve results?

```python
from scipy import stats

# Assume we have Cross-Validation results for 2 models
# (Logged as separate metrics or retrieved from multiple runs)

# Metric: Accuracy across 10 folds
model_a_scores = [0.88, 0.89, 0.87, 0.88, 0.90, 0.88, 0.89, 0.88, 0.87, 0.89]
model_b_scores = [0.90, 0.91, 0.89, 0.90, 0.92, 0.90, 0.91, 0.90, 0.89, 0.91]

t_stat, p_val = stats.ttest_rel(model_a_scores, model_b_scores)

print(f"P-Value: {p_val}")
if p_val < 0.05:
    print("Model B is statistically significantly better.")
else:
    print("Difference might be due to random noise.")
```

---

## 🔬 Lab Exercise: "Pareto Frontier"

### Task
Trade-offs.
1.  Assume runs logged `metrics.accuracy` and `metrics.inference_time_ms`.
2.  Plot Scatter: X=Time, Y=Accuracy.
3.  **Goal:** Find models in the top-left corner (High Acc, Low Time).
4.  **Action:** Filter `df` to find points where no other point has (Better Acc AND Lower Time). These are the Pareto Optimal models.

```python
def is_pareto_efficient(costs):
    # costs is a (n_points, 2) array. 
    # We want to minimize costs (so invert accuracy to -acc)
    pass # Implementation of generic culgo
```

In Pandas, manual Inspection is usually enough:
```python
efficient = df[(df["accuracy"] > 0.9) & (df["inference_time"] < 20)]
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Data Freedom:** The dashboard is great for quick checks, but Pandas is the ultimate tool for specific questions ("Show me the average loss of runs created on Tuesday").
2.  **SQL:** The underlying DB is Postgres. You *can* query it directly with SQL if the API is too slow for 1M+ runs.
3.  **Automation:** You can write a script that runs every night, finds the best run, and generates a PDF report for the team.

### API Summary
```python
mlflow.search_runs(filter_string="...")
```

---

**Day 131 Complete** ✅

*Next: Day 132 - Lineage & Reproducibility - CSI: ML Edition.*
