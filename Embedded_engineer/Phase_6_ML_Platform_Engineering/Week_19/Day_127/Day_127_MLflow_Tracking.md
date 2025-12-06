# Day 127: The Diary of a Data Scientist: MLflow Tracking
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** "I trained a model 2 weeks ago that had 98% accuracy. I don't remember the learning rate, the dataset version, or where I saved the `.pth` file." **MLflow Tracking** is the solution to reproducibility amnesia.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** a production-grade MLflow Tracking Server with SQL backend and S3 artifact store.
2.  **Instrument** training code to log Parameters, Metrics, and Artifacts.
3.  **Utilize** `mlflow.autolog()` to capture framework-specific details automatically.
4.  **Query** the MLflow UI to compare runs and visualize learning curves.
5.  **Differentiate** between an "Experiment" and a "Run".

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine or Cloud VM.
- Access to an S3 Bucket (or MinIO) and a PostgreSQL DB (or SQLite for dev).

### Software Environment
- `pip install mlflow psycopg2-binary boto3 sklearn torch matplotlib`.
- Docker (for hosting the server).

---

## 📖 Theoretical Foundation

### 1. The Reproducibility Crisis
In software engineering, Git versions the code.
In Machine Learning, 3 components change:
1.  **Code:** (Git)
2.  **Data:** (DVC/Delta Lake)
3.  **Config/Hyperparameters:** (MLflow)

If you miss one, you cannot reproduce the model.

### 2. MLflow Architecture
*   **Tracking Server:** A Flask app. Accepts REST calls.
*   **Backend Store:** Database (Postgres/MySQL). Stores metadata (Run ID, Start Time, Params, Metrics).
*   **Artifact Store:** Blob Storage (S3/GCS/Azure Blob). Stores heavy files (Models, Plots, Parquet files).
*   **Client:** `mlflow` Python package.

### 3. Key Concepts
*   **Experiment:** A logical group of runs (e.g., "ResNet50-Cifar10").
*   **Run:** A single execution of the code.
*   **Parameters:** Inputs (Batch Size, LR). Constant for the run.
*   **Metrics:** Outputs (Loss, Accuracy). Changing over time (TimeSeries).
*   **Tags:** Metadata (User, Git Hash, Description).
*   **Artifacts:** Output files.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Setting up the Server

We don't use `mlflow ui` (Local) for teams. We use a centralized server.

#### 📁 `infra/docker-compose.yml`
```yaml
version: '3.7'
services:
  db:
    image: postgres:13
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: miniopassword
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.7.1
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:password@db:5432/mlflow
      --default-artifact-root s3://mlflow-bucket/
      --host 0.0.0.0
    ports:
      - "5000:5000"
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: miniopassword
    depends_on:
      - db
      - minio

volumes:
  db_data:
  minio_data:
```

### 👨‍💻 Core Implementation: The Data Scientist's Workflow

#### 📁 `src/01_tracking_demo.py`
```python
import mlflow
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuration - Point to the Server
# If running outside docker network, use localhost
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniopassword"

mlflow.set_tracking_uri("http://localhost:5000")

# 2. Define Experiment
experiment_name = "Phase6_Week19_Demo"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass # Exists
mlflow.set_experiment(experiment_name)

# 3. Training Function
def train_model(lr, epochs):
    # START RUN
    with mlflow.start_run(run_name=f"LR_{lr}"):
        # A. Log Parameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "SGD")
        
        # B. Log Tags (Context)
        mlflow.set_tag("developer", "Abhishek")
        mlflow.set_tag("phase", "prototype")

        # Simulation of Training
        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_history = []
        
        print(f"Training with LR={lr}...")
        for epoch in range(epochs):
            # Fake loss curve
            loss = (0.5 - lr * epoch * 0.1)**2 + np.random.normal(0, 0.01)
            loss_history.append(loss)
            
            # C. Log Metrics (Step-wise)
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", 0.1 * epoch, step=epoch)
        
        # D. Log Artifacts (Plots)
        plt.figure()
        plt.plot(loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("loss_curve.png")
        
        mlflow.log_artifact("loss_curve.png")
        
        # E. Log Model (The Artifact)
        # MLflow has native support for PyTorch serialization
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")

# 4. Execute Runs
train_model(0.01, 10)
train_model(0.05, 10) # Aggressive LR
```

### 👨‍💻 Core Implementation: Autologging

For standard libraries (Sklearn, Lightning, Keras), MLflow can capture everything automatically.

```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()

with mlflow.start_run():
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(X_train, y_train)
    # Automatically logs:
    # - n_estimators, max_depth, ...
    # - feature_importance plot
    # - confusion matrix (if classifier)
    # - model.pkl
```

---

## 🔬 Lab Exercise: "The Comparison"

### Task
Use the UI.
1.  Navigate to `http://localhost:5000`.
2.  Select Experiment `Phase6_Week19_Demo`.
3.  Select the two runs (checkboxes).
4.  Click **Compare**.
5.  **Scatter Plot:** X-axis: `learning_rate`, Y-axis: `loss` (min).
6.  **Contour Plot:** Visualize the parameter space.
7.  **Parallel Coordinates:** See how params flow into metrics.
8.  **Insight:** Run 2 (LR=0.05) converged faster but had higher variance.

---

## 🐞 Debugging Guide

### Common Issues

#### 1. "Connection Refused"
*   **Symptom:** Client cannot talk to server.
*   **Fix:** Ensure `MLFLOW_TRACKING_URI` is correct. If running in Docker, `localhost` refers to the container, not the host. Use `host.docker.internal` or Docker networking aliases.

#### 2. "S3 Upload Failed"
*   **Symptom:** `boto3` cannot connect to MinIO.
*   **Fix:**
    *   Ensure `AWS_ACCESS_KEY_ID` are set in the *Client* environment.
    *   Ensure `MLFLOW_S3_ENDPOINT_URL` is reachable from the client.
    *   Create the bucket `mlflow-bucket` in MinIO console (`localhost:9001`) manually before running code.

#### 3. Database Migration
*   **Symptom:** Server crash on startup.
*   **Fix:** MLflow server usually auto-migrates. Check Postgres logs.

---

## 🔒 Security & Best Practices

1.  **Never log PII:** Do not log usernames or emails as Params/Tags. Use Hashed IDs.
2.  **Dataset Versioning:**
    *   Instead of logging the *whole dataset* as an artifact (Slow), log the **Commit Hash** of the data creation script or the **S3 Path + Version ID**.
3.  **Clean up:**
    *   Use `mlflow gc` to manage deleted runs.
    *   Don't keep 1TB of failed run checkpoints.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Centralization:** A shared tracking server allows the whole team to see "What works". It democratizes knowledge.
2.  **Artifacts:** The ability to download the EXACT model binary that produced a specific result 6 months ago is the definition of reproducibility.
3.  **Autologging:** Use it for baselines, but prefer manual logging for custom architectures where you want specific business metrics.

### API Summary
```python
mlflow.set_tracking_uri(uri)
mlflow.create_experiment(name)
with mlflow.start_run():
    mlflow.log_param(key, val)
    mlflow.log_metric(key, val)
    mlflow.log_artifact(local_path)
```

---

**Day 127 Complete** ✅

*Next: Day 128 - The Registry - From Experiment to Production.*
