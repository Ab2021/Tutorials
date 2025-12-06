# Day 133: Week 19 Review & Project - The MLOps Platform Facade
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** We have learned the primitives (Tracking, Registry, Lineage). Now, we build a **Unified SDK** that abstracts these details away from the Data Scientist, enforcing best practices automatically.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a Python wrapper (`MLPlatform`) that standardizes logging.
2.  **Enforce** mandatory tags (Owner, Jira ID) before allowing a run to start.
3.  **Implement** a "One-Click Deploy" method that handles the Registry transition.
4.  **Verify** the platform with an end-to-end integration test.

---

## 📚 Week 19 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 127 | Tracking | "I can see the loss curve of a model I trained 3 months ago." |
| 128 | Registry | "I just load 'Production' version. I don't care about filenames." |
| 129 | W&B | "The 3D point cloud visualization is amazing for debugging." |
| 130 | Artifacts | "ONNX runs without PyTorch installed." |
| 131 | Analysis | "Run 42 is statistically significantly better than Run 12." |
| 132 | Lineage | "The model failed because the dataset hash changed." |

---

## 🏗️ Final Project: "EzML"

### Scenario
You are the Platform Engineer. The Data Scientists keep forgetting to log params and commit code.
You build `EzML`, a library that:
1.  Fails if Git is dirty.
2.  Auto-logs params.
3.  Auto-registers successful models.

### Step 1: The Facade Class

#### 📁 `project/ezml.py`
```python
import mlflow
import git
import os
from mlflow.tracking import MlflowClient

class EzML:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.repo = git.Repo(search_parent_directories=True)

    def start_training(self, developer_name, jira_ticket):
        # 1. Enforce Clean Git
        if self.repo.is_dirty():
            raise RuntimeError("Git is dirty! Commit changes or stash them.")

        # 2. Start Run with Mandatory Tags
        run = mlflow.start_run()
        mlflow.set_tag("user", developer_name)
        mlflow.set_tag("jira", jira_ticket)
        mlflow.set_tag("git_commit", self.repo.head.object.hexsha)
        
        print(f"Started Run: {run.info.run_id}")
        return run

    def log_model(self, model, artifact_path, input_example=None):
        # 3. Log Model
        mlflow.sklearn.log_model(model, artifact_path, input_example=input_example)
        self.latest_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    def register_and_promote(self, model_name, metric_name, threshold):
        # 4. Conditional Registration
        run = mlflow.active_run()
        metric_val = run.data.metrics.get(metric_name, 0)
        
        if metric_val > threshold:
            print(f"Metric {metric_val} > {threshold}. Promoting to Staging.")
            
            # Register
            result = mlflow.register_model(self.latest_model_uri, model_name)
            time.sleep(1) # Wait
            
            # Transition
            self.client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage="Staging"
            )
        else:
            print(f"Metric {metric_val} too low. Not registering.")

    def end(self):
        mlflow.end_run()
```

### Step 2: The Data Scientist's Usage

#### 📁 `project/train.py`
```python
from ezml import EzML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Init Platform
platform = EzML("Iris_Project")

# 2. Start Training (Enforces Logic)
try:
    platform.start_training(developer_name="Abhi", jira_ticket="ML-123")
    
    # 3. Standard ML Code
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # 4. Log via Platform (or direct mlflow calls, platform still captures context)
    mlflow.log_metric("accuracy", acc)
    platform.log_model(model, "rf_model")
    
    # 5. Auto-Promote Logic
    platform.register_and_promote("Iris_Production", "accuracy", 0.90)

finally:
    platform.end()
```

### Step 3: Verification

1.  **Test 1 (Dirty Git):** Modify file, run script. Expect Error.
2.  **Test 2 (Low Acc):** Accuracy < 0.90. Expect "Not registering".
3.  **Test 3 (High Acc):** Accuracy > 0.90.
    *   Check MLflow UI.
    *   See Tag `jira=ML-123`.
    *   See Model `Iris_Production` Version N in `Staging`.

---

## 🔬 Lab Exercise: "Dependency Injection"

### Task
Make EzML agnostic.
1.  Currently `log_model` assumes `sklearn`.
2.  Refactor `log_model` to accept a `flavor` argument (`'sklearn'`, `'pytorch'`, `'onnx'`).
3.  Use `getattr(mlflow, flavor).log_model(...)` dynamic dispatch.

---

## 📝 Success Criteria
1.  **Guardrails:** Impossible to maximize metric on uncommitted code.
2.  **Metadata:** Every run has an Owner and Ticket.
3.  **Automation:** Good models appear in Registry automatically. Bad models stay as Artifacts.

---

**Week 19 Complete** ✅

*Next Phase: CI/CD for ML - Automating the Loop.*
