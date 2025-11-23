# Lab 1: MLflow Experiment Tracking

## Objective
Stop guessing which hyperparameters worked. Track everything.
We will log params, metrics, and the model itself.

## 1. Setup

```bash
poetry add mlflow
```

## 2. The Experiment (`track.py`)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("LLM_Finetuning_Proxy")

with mlflow.start_run():
    # 1. Log Params
    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)
    
    # 2. Train (Mocking LLM training with RF for speed)
    model = RandomForestRegressor(**params)
    model.fit([[1]], [1]) # Dummy data
    
    # 3. Log Metrics
    loss = 0.1
    mlflow.log_metric("eval_loss", loss)
    
    # 4. Log Model
    mlflow.sklearn.log_model(model, "model")
    
    print("Run Complete. Check MLflow UI.")
```

## 3. Running the Lab
1.  Run `python track.py`.
2.  Run `mlflow ui`.
3.  Open `localhost:5000`.

## 4. Submission
Submit a screenshot of the MLflow dashboard showing your run.
