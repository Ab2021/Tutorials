# Day 28.1: MLOps Fundamentals - A Practical Guide

## Introduction: Beyond the Notebook - Productionizing Machine Learning

So far, our focus has been on the research and development phase of machine learning: exploring data, building models, and training them in an interactive environment like a Jupyter notebook. However, to deliver real business value, a model must be deployed into a production environment where it can serve predictions to users or other systems. This is where **Machine Learning Operations (MLOps)** comes in.

MLOps is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. It adapts the principles of **DevOps** (collaboration, automation, CI/CD) to the unique challenges of the machine learning lifecycle. These challenges include managing large datasets, tracking experiments, versioning models, monitoring performance, and ensuring reproducibility.

This guide will provide a practical introduction to the fundamental concepts of MLOps, moving beyond theory to illustrate the core components of a modern MLOps workflow.

**Today's Learning Objectives:**

1.  **Understand the ML Lifecycle:** Grasp the end-to-end stages of a machine learning project, from data ingestion to model monitoring.
2.  **Learn about Experiment Tracking:** Understand why tracking experiments is crucial and use a tool like MLflow to log parameters, metrics, and models.
3.  **Master Model Versioning and the Model Registry:** Learn how to version and manage trained models for reproducibility and safe deployment using a model registry.
4.  **Explore Model Deployment Strategies:** Get an overview of different ways to serve a model, from a simple batch script to a real-time REST API.
5.  **Appreciate the Importance of Monitoring:** Understand why monitoring a model's performance in production is critical for detecting drift and maintaining value.

---

## Part 1: The End-to-End Machine Learning Lifecycle

An MLOps view of a project extends far beyond `model.fit()`.

1.  **Data Ingestion & Versioning:** Sourcing, cleaning, and transforming data. It's crucial to version your datasets (e.g., using DVC) so you can always trace a model back to the exact data it was trained on.
2.  **Model Training & Experiment Tracking:** This is the research phase. We need to systematically track every experiment: the code version, data version, hyperparameters, and resulting metrics.
3.  **Model Packaging & Validation:** Before deployment, the model must be packaged with all its dependencies and validated on a hold-out test set to ensure it meets business requirements.
4.  **Model Deployment & Serving:** The packaged model is deployed to a production environment. This could be as a REST API for real-time predictions, a batch process, or embedded in an edge device.
5.  **Model Monitoring & Retraining:** Once deployed, the model's performance must be continuously monitored for concept drift (when the statistical properties of the target variable change) or data drift (when the properties of the input data change). When performance degrades, a trigger is fired to retrain the model on new data.

![MLOps Lifecycle](https://i.imgur.com/9I8A9Z2.png)

---

## Part 2: Experiment Tracking with MLflow

When you're trying dozens of different model architectures, hyperparameters, and feature engineering techniques, it's easy to lose track of what worked. Experiment tracking tools solve this by providing a central, organized repository for your results.

**MLflow** is a popular open-source platform for managing the ML lifecycle. We'll focus on its **Tracking** component.

### 2.1. Code: Logging a Training Run with MLflow

First, install MLflow: `pip install mlflow scikit-learn`

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

print("---" + "-" * 25 + " Part 2: Experiment Tracking with MLflow " + "-" * 25 + "---")

# --- 1. Prepare Data ---
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Start an MLflow Run ---
# MLflow runs can be organized into "experiments"
mlflow.set_experiment("Iris Classification Fun")

with mlflow.start_run(run_name="RandomForest_Run_1") as run:
    # --- 3. Log Parameters ---
    # These are the inputs to your model
    n_estimators = 150
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    print(f"Starting run: {run.info.run_name}")
    print(f"  Logging parameters: n_estimators={n_estimators}, max_depth={max_depth}")

    # --- 4. Train the Model ---
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # --- 5. Log Metrics ---
    # These are the outputs/results of your model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    print(f"  Logging metric: accuracy={acc:.4f}")

    # --- 6. Log the Model ---
    # This saves the trained model itself
    mlflow.sklearn.log_model(model, "random_forest_model")
    print("  Model logged.")

print("\nRun complete!")
print("To see your results, run 'mlflow ui' in your terminal and open http://localhost:5000")
```

After running this script, you can launch the MLflow UI. You will see your experiment, the run, and all the parameters, metrics, and artifacts (the model) you logged.

---

## Part 3: The Model Registry

Once you have a model that performs well, you need a safe way to manage its lifecycle from development to production. This is the job of a **Model Registry**.

The MLflow Model Registry provides:
*   **Versioning:** Every model you register gets a version number (e.g., `iris-classifier v1`, `v2`).
*   **Staging:** Models can be moved through predefined stages, typically: `Staging`, `Production`, and `Archived`.
*   **Annotations and Descriptions:** You can add metadata to your models, describing what they do, what data they were trained on, etc.

### 3.1. Code: Registering a Model with MLflow

We can extend the previous script to register the model we logged.

```python
print("\n--- Part 3: Registering a Model ---")

# This code would typically run after you've reviewed the run in the UI
# and decided it's a good candidate for production.

# We need the run_id from the previous run
run_id = run.info.run_id
model_name = "IrisClassifierRF"

# The model URI follows the format: "runs:/<run_id>/<path_to_model>"
model_uri = f"runs:/{run_id}/random_forest_model"

# Register the model
# This will create version 1 of the model
registered_model = mlflow.register_model(model_uri, model_name)

print(f"Model '{model_name}' registered with version {registered_model.version}")

# You can now go to the "Models" tab in the MLflow UI to see it.
# From there, you can manually transition its stage to "Production".
```

---

## Part 4: Model Deployment and Serving

Deployment is the process of making your model available to users. MLflow provides built-in tools to serve models from the registry as a REST API.

### 4.1. Serving a Model with the MLflow CLI

This is the simplest way to deploy a model for real-time inference. Once a model is in the registry, you can serve it with a single command in your terminal.

**Command:**
```bash
# Serve version 1 of our registered model
# The model URI follows the format: "models:/<model_name>/<version_or_stage>"
mlflow models serve -m "models:/IrisClassifierRF/1" -p 1234
```

This command starts a local web server on port 1234. You can now send it data and get predictions back.

### 4.2. Code: Querying the Deployed Model

We can use Python's `requests` library to send a POST request to the running model server.

```python
import requests
import json

print("\n--- Part 4: Querying the Deployed Model ---")

# The server expects data in a specific JSON format
# We'll use the first test sample
data = X_test[0].tolist()

request_body = {
    "dataframe_records": [data]
}

# Send the request
url = "http://localhost:1234/invocations"
headers = {"Content-Type": "application/json"}

print("Sending request to the model server...")
# In a separate terminal, you must have the `mlflow models serve` command running

# try:
#     response = requests.post(url, data=json.dumps(request_body), headers=headers)
#     response.raise_for_status() # Raise an exception for bad status codes
#     prediction = response.json()['predictions'][0]
#     print(f"\nReceived prediction: {prediction}")
#     print(f"True label: {y_test[0]}")
# except requests.exceptions.ConnectionError as e:
#     print("\nCould not connect to the model server.")
#     print("Please run the following command in a separate terminal:")
#     print('mlflow models serve -m "models:/IrisClassifierRF/1" -p 1234')

print("\n(Code to query the server is commented out to prevent errors during automated runs)")
print("To test this, uncomment the code and run the `mlflow models serve` command.")
```

## Conclusion

MLOps is a critical discipline for any organization that wants to derive real, sustained value from machine learning. It transforms ML from a research activity into a reliable and automated engineering process.

**Key Takeaways:**

1.  **The Lifecycle is a Loop, Not a Line:** MLOps is a continuous cycle of training, deploying, monitoring, and retraining.
2.  **Track Everything:** Use tools like MLflow to log your experiments. If you can't reproduce a result, it might as well have not happened.
3.  **Version Your Assets:** Your code, data, and models should all be versioned to ensure reproducibility and enable safe rollbacks.
4.  **A Model Registry is Your System of Record:** It provides a central place to manage and govern the lifecycle of your production models.
5.  **Deployment is Not the End:** The work isn't over when the model is deployed. Monitoring for performance degradation is essential for long-term success.

By adopting these fundamental MLOps practices, you can bridge the gap between your notebook and a production system, ensuring your models are robust, reliable, and valuable.

## Self-Assessment Questions

1.  **MLOps vs. DevOps:** What is one key challenge in MLOps that is not typically present in traditional software DevOps?
2.  **Experiment Tracking:** What are the three main things you should log during an MLflow run?
3.  **Model Registry:** What are the three common stages for a model in a registry, and what does each one signify?
4.  **Model URI:** What is the difference between a model URI starting with `runs:/` and one starting with `models:/` in MLflow?
5.  **Model Drift:** What is the difference between "concept drift" and "data drift"?
