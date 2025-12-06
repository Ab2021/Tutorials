# Day 134: The Factory Floor: Designing ML Pipelines
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** A Jupyter Notebook is a laboratory. An ML Pipeline is a factory. We must transition from "cells executed in random order" to a Directed Acyclic Graph (DAG) of immutable steps.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Decompose** a monolithic training script into discrete pipeline steps (Ingest, Train, Eval).
2.  **Contrast** Orchestrators: generic (Airflow) vs ML-native (Kubeflow, TFX, ZenML).
3.  **Implement** a stateless component interface that accepts S3 URIs and outputs S3 URIs.
4.  **Design** a Caching strategy to avoid retraining on identical data.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install fire pandas scikit-learn`.

---

## 📖 Theoretical Foundation

### 1. The Monolith Problem
*   **Monolith:** `train.py` does `pd.read_csv`, `clean`, `train`, `evaluate`.
*   **Issue:** If `evaluate` fails, you must re-run `train` (expensive). If data changes, you don't know if `train` needs re-running.

### 2. The TFX Standard Components
Google TFX defined the standard ML DAG:
1.  **ExampleGen:** Ingest data.
2.  **StatisticsGen:** Calculate distributions.
3.  **SchemaGen:** Infer types.
4.  **ExampleValidator:** Detect anomalies.
5.  **Transform:** Feature engineering.
6.  **Trainer:** Model training.
7.  **Evaluator:** Deep analysis.
8.  **Pusher:** Deploy to registry.

### 3. Caching
If Step A's inputs (Data Hash + Code Hash + Config) haven't changed, Step A's output is retrieved from cache. This speeds up iteration loops by 100x.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Functional Components

We design a simple Python-based pipeline interface.

#### 📁 `src/pipeline_components.py`
```python
import pandas as pd
import pickle
import os
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Helper for artifact passing
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 1. Ingest
def ingest(source_url: str, output_path: str):
    print(f"Ingesting from {source_url}...")
    # Simulate fetch
    df = pd.DataFrame({"x": range(100), "y": [i*2 for i in range(100)]})
    ensure_dir(output_path)
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

# 2. Split
def split(input_path: str, train_path: str, test_path: str, ratio: float):
    print("Splitting data...")
    df = pd.read_parquet(input_path)
    train, test = train_test_split(df, train_size=ratio, random_state=42)
    ensure_dir(train_path)
    ensure_dir(test_path)
    train.to_parquet(train_path)
    test.to_parquet(test_path)

# 3. Train
def train(train_path: str, model_path: str, n_estimators: int):
    print(f"Training with n_estimators={n_estimators}...")
    df = pd.read_parquet(train_path)
    X = df[["x"]]
    y = df["y"]
    
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X, y)
    
    ensure_dir(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

# 4. Evaluate
def evaluate(test_path: str, model_path: str, metrics_path: str):
    print("Evaluating...")
    df = pd.read_parquet(test_path)
    X = df[["x"]]
    y = df["y"]
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    
    metrics = {"mse": mse, "passed": mse < 0.1}
    ensure_dir(metrics_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    
    if not metrics["passed"]:
        raise ValueError(f"Model failed criteria! MSE: {mse}")
```

### 👨‍💻 Core Implementation: The Orchestrator Script

A simple script that wires them together (poor man's Airflow).

#### 📁 `src/run_pipeline.py`
```python
import fire
from pipeline_components import ingest, split, train, evaluate

class Pipeline:
    def run(self, build_id="1"):
        base_dir = f"artifacts/{build_id}"
        
        # Define Paths
        raw_data = f"{base_dir}/data.parquet"
        train_data = f"{base_dir}/train.parquet"
        test_data = f"{base_dir}/test.parquet"
        model_file = f"{base_dir}/model.pkl"
        metrics_file = f"{base_dir}/metrics.json"
        
        # Execute DAG
        ingest("s3://bucket/raw.csv", raw_data)
        split(raw_data, train_data, test_data, 0.8)
        train(train_data, model_file, n_estimators=100)
        evaluate(test_data, model_file, metrics_file)
        
        print("Pipeline Success!")

if __name__ == "__main__":
    fire.Fire(Pipeline)
```

---

## 🔬 Lab Exercise: "Containerization"

### Task
Make it portable.
1.  Create a `Dockerfile`:
    ```dockerfile
    FROM python:3.9
    COPY src /app
    RUN pip install pandas scikit-learn fire pyarrow
    WORKDIR /app
    ENTRYPOINT ["python", "run_pipeline.py"]
    ```
2.  Build: `docker build -t ml-pipeline:v1 .`.
3.  Run: `docker run -v $(pwd)/artifacts:/app/artifacts ml-pipeline:v1 run --build_id=2`.
4.  **Insight:** Each step could be its own container image if dependencies differ (e.g., Training needs CUDA, Ingest only needs Pandas).

---

## 📖 Advanced Theory: Pipeline DSLs

In production, we use DSLs to define this graph.
*   **Kubeflow Pipelines (KFP):**
    ```python
    @dsl.pipeline(name='MyPipeline')
    def my_pipeline():
        ingest_task = ingest_op()
        train_task = train_op(ingest_task.output)
    ```
*   **ZenML:** Wrapper that allows swapping backends (running the same code on Local, Airflow, or Kubeflow).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Artifact Passing:** Steps communicate via *Files* (Parquet/Pickle), not RAM. This allows steps to run on different machines.
2.  **Idempotency:** A step should always produce the same output for same input. Avoid `random.random()` without seeding.
3.  **Validation:** The `evaluate` step acts as a Quality Gate. If it fails, the pipeline breaks, preventing bad model deployment.

### API Summary
```python
# Design Principle
def step(input_artifact, config) -> output_artifact
```

---

**Day 134 Complete** ✅

*Next: Day 135 - GitHub Actions - The CI Runner.*
