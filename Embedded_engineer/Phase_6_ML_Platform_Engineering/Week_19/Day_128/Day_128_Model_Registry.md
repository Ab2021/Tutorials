# Day 128: The Gatekeeper: MLflow Model Registry
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** Thousands of experiment runs create chaos. Which model is currently in Production? Which one is being tested in Staging? The **Model Registry** is the single source of truth for deployment.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Register** a model artifact from an experiment run into the Central Registry.
2.  **Transition** model versions between stages (`None`, `Staging`, `Production`, `Archived`).
3.  **Validate** model stages programmatically to ensure only approved models are served.
4.  **Implement** a basic approval workflow using MLflow Client API.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (Connected to Day 127 Server).

### Software Environment
- `pip install mlflow`.

---

## 📖 Theoretical Foundation

### 1. Artifact vs Registered Model
*   **Artifact:** A file (`model.pkl`) living in S3 inside `runs/<run_id>/artifacts`. It is immutable. It represents "What happened".
*   **Registered Model:** A pointer to an artifact. It has a Name (e.g., `FraudDetector`), Versions (v1, v2), and Stages. It represents "What we intend to use".

### 2. The Lifecycle Stages
*   **None:** Just registered. Developer is testing it.
*   **Staging:** Candidate for release. Undergoing integration tests / QA.
*   **Production:** The live model serving traffic.
*   **Archived:** Deprecated models.

### 3. The Deployment Contract
In MLOps, deployment scripts NEVER hardcode paths.
*   *Bad:* `load_model("s3://bucket/runs/e45a.../model.pkl")`
*   *Good:* `load_model("models:/FraudDetector/Production")`

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Registration Workflow

#### 📁 `src/02_registry_workflow.py`
```python
import mlflow
from mlflow.tracking import MlflowClient
import time

# Setup
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
model_name = "Phase6_Regressor"

# 1. Create a dummy run and log a model
print("Training Model...")
with mlflow.start_run() as run:
    mlflow.log_param("type", "prototype")
    # Log a fake model (using sklearn for simplicity)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit([[1]], [[1]])
    
    # Log it. This puts it in Artifact Store.
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

print(f"Run ID: {run_id}")

# 2. Register the Model
# This creates a new Version (e.g., v1)
print(f"Registering {model_name}...")
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)
time.sleep(1) # Wait for backend to register

version = result.version
print(f"Registered Version: {version}")

# 3. Add Description
client.update_model_version(
    name=model_name,
    version=version,
    description="Initial prototype trained on dummy data."
)
```

### 👨‍💻 Core Implementation: Stage Transitions

#### 📁 `src/03_transition.py`
```python
import mlflow
client = mlflow.MlflowClient()
model_name = "Phase6_Regressor"
version = 1  # Assume v1 exists

# 1. Promote to Staging
print("Promoting to Staging...")
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

# 2. Deployment Code (Simulation)
def load_for_testing():
    # Load whatever is in Staging
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")
    print("Loaded Staging Model for QA.")
    return model

qa_model = load_for_testing()
# Run tests...
tests_passed = True

# 3. Promote to Production
if tests_passed:
    print("Promoting to Production...")
    # Archive existing Prod version automatically?
    # MLflow allows multiple versions in Prod unless archive_existing_versions=True
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

# 4. Production Serving Code
def load_prod():
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    print("Serving traffic with Production Model.")
    return model

prod_model = load_prod()
```

---

## 🔬 Lab Exercise: "The Rollback"

### Task
Simulate a failure.
1.  Train v2 (Bad Model). Register it. Promote to Production.
2.  `load_prod()` loads v2.
3.  Users complain ("Accuracy dropped!").
4.  **Emergency:** Rollback.
    *   Find v1.
    *   `client.transition_model_version_stage(name, version=1, stage="Production")`.
5.  `load_prod()` now loads v1 immediately.
6.  **Constraint:** This requires the serving service to reload the model. Ray Serve / Seldon usually poll the Registry for changes.

---

## 📖 Advanced Theory: Webhooks & Automation

How do we trigger the "Tests" automatically?
MLflow (managed) supports Webhooks.
*   **Event:** `MODEL_VERSION_TRANSITIONED_TO_STAGING`
*   **Action:** POST to Jenkins/GitHub Actions.
*   **Jenkins:** Runs `pytest`.
*   **Post-Action:** If Pass -> API call to transition to Production. If Fail -> Reject.

For Open Source MLflow, we usually poll the registry or use a wrapper service.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decoupling:** The "Training Engineer" produces artifacts. The "Release Engineer" manages the Registry. The "Serving Service" only reads `models:/MyModel/Production`.
2.  **Context:** Description and Tags in the registry are vital. Put links to the Jira Ticket or PR that authorized the model.
3.  **Governance:** You can put Access Control (RBAC) on the Registry to prevent Junior Engineers from overwriting Production.

### API Summary
```python
mlflow.register_model(uri, name)
client.transition_model_version_stage()
mlflow.pyfunc.load_model(model_uri)
```

---

**Day 128 Complete** ✅

*Next: Day 129 - Weights & Biases - The Social Network for Models.*
