# Day 132: CSI - Model: Lineage & Reproducibility
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** If a regulator asks, "Why did this model deny the loan?", you need to produce the exact training data, code version, and environment that created it. **Lineage** links the Artifact to the Source.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Capture** Git Commit Hashes automatically within runs.
2.  **Log** Dataset Versions (Delta versions or CSV hashes) as input params.
3.  **Validate** reproducibility by retraining a model from an old Run ID.
4.  **Visualize** the upstream/downstream lineage graph (Data -> Run -> Model -> Serving).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install mlflow gitpython`.
- Git repository initialized.

---

## 📖 Theoretical Foundation

### 1. The Trinity of Lineage
To completely define a model:
$$ Model = Code(Data, Config) |_{Environment} $$
1.  **Code:** Git Hash (`HEAD`).
2.  **Data:** S3 URI + Version (or Hash).
3.  **Config:** Hyperparameters.
4.  **Environment:** Docker Image SHA or Conda yaml.

### 2. Delta Tables (Data Versioning)
We cannot log 1TB of data to MLflow. We log:
`dataset_source: s3://bucket/table`
`dataset_version: 5`
This requires using a versioned datastore (Delta Lake / DVC / LakeFS).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Automated Git capture

MLflow automatically captures git info if run from a Project. But locally, we can force it.

#### 📁 `src/07_lineage.py`
```python
import mlflow
import git
import hashlib
import pandas as pd

# 1. Get Git Hash
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
print(f"Git Commit: {sha}")

# 2. Get Data Hash
def get_data_hash(df):
    # For small data, hash the content
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# Simulate loading data
df = pd.DataFrame({"feature": range(100), "target": range(100)})
data_version = get_data_hash(df)
data_source = "s3://my-bucket/training_data.csv"

# 3. Training Run
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Lineage_Demo")

with mlflow.start_run():
    # A. Log Code Version
    mlflow.set_tag("mlflow.source.git.commit", sha)
    mlflow.set_tag("mlflow.source.name", "src/07_lineage.py")
    
    # B. Log Data Lineage
    mlflow.log_param("dataset_source", data_source)
    mlflow.log_param("dataset_version", data_version)
    
    # C. Train
    mlflow.log_metric("accuracy", 0.95)
    
    print(f"Run {mlflow.active_run().info.run_id} linked to Commit {sha}")
```

### 👨‍💻 Core Implementation: Reproducibility Check

Script to verify if we can rebuild the model.

```python
def verify_run(run_id):
    run = mlflow.get_run(run_id)
    
    # 1. Check Git State
    logged_sha = run.data.tags.get("mlflow.source.git.commit")
    current_sha = git.Repo(".").head.object.hexsha
    
    if logged_sha != current_sha:
        print(f"WARNING: Current code ({current_sha}) != Logged code ({logged_sha})")
        print("Please 'git checkout' the logged commit before retraining.")
        return False
        
    # 2. Check Data
    logged_data_ver = run.data.params.get("dataset_version")
    # Verify current data matches...
    
    print("Environment matches. Ready to retrain.")
    return True
```

---

## 🔬 Lab Exercise: "Dirty Repo"

### Task
Simulate "Uncommitted Changes".
1.  Run the script. It logs the HEAD SHA.
2.  Modify the script (e.g., change LR). **Do not commit.**
3.  Run the script. It logs the SAME HEAD SHA.
4.  **Problem:** The logged SHA implies code that *doesn't match* what ran (because uncommitted changes existed).
5.  **Solution:** Add a check at the start:
    ```python
    if repo.is_dirty():
        raise RuntimeError("Cannot allow run with uncommitted changes! Commit or stash.")
    ```

---

## 📝 Daily Summary

### Key Takeaways
1.  **DVC Integration:** Use DVC files (`.dvc`) as artifacts. Log the content of the `.dvc` file to MLflow to link the data version.
2.  **MLflow Projects:** `.mlflow-project` files define the entry point and environment capability. Running via `mlflow run .` enforces Conda environment creation, ensuring environment reproducibility.
3.  **Traceability:** If a Model fails in Production, tracing back to the SQL query that generated the training data is the only way to debug "Data Drift" issues.

### API Summary
```python
mlflow.set_tag("mlflow.source.git.commit", sha)
git.Repo().head.object.hexsha
```

---

**Day 132 Complete** ✅

*Next: Day 133 - Week 19 Review & Project - Building a Platform.*
