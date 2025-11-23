# Day 65: Experiment Tracking & Versioning
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. MLflow Tracking Pipeline Implementation

A complete training loop integrated with MLflow.

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Define Model
class SimpleNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. Training Function
def train_model(params):
    # Start MLflow Run
    with mlflow.start_run(run_name="experiment_v1"):
        # Log Parameters
        mlflow.log_params(params)
        
        # Setup
        model = SimpleNet(hidden_dim=params['hidden_dim'])
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()
        
        # Dummy Data
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)
        
        # Training Loop
        for epoch in range(5):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            # Log Metrics
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
            
        # Log Model
        # This saves the model + conda env + requirements
        mlflow.pytorch.log_model(model, "model")
        
        return avg_loss

# 3. Run Experiment
params = {
    "lr": 0.01,
    "hidden_dim": 128,
    "batch_size": 32
}
train_model(params)
```

### 2. Hyperparameter Sweep with Optuna

Automating the search for best parameters.

```python
import optuna

def objective(trial):
    # Define search space
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    
    params = {
        "lr": lr,
        "hidden_dim": hidden_dim
    }
    
    # Nested run for tracking
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        final_loss = train_model_mock(params) # Assume this returns validation loss
        mlflow.log_metric("val_loss", final_loss)
        
    return final_loss

def train_model_mock(params):
    # Mock training function
    return (params['lr'] - 0.01)**2 + (params['hidden_dim'] - 64)**2

# Run Optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best params:", study.best_params)
```

### 3. Model Registry Promotion Workflow

Script to promote a model from Staging to Production.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_model(model_name, version, new_stage):
    print(f"Promoting {model_name} v{version} to {new_stage}...")
    
    # 1. Transition Stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=new_stage,
        archive_existing_versions=True # Move old Production to Archived
    )
    
    # 2. Update Description
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Promoted to {new_stage} on {datetime.now()}"
    )

# Usage
# promote_model("MyLlamaModel", "5", "Production")
```

### 4. Weights & Biases (W&B) Integration

Comparing W&B syntax with MLflow.

```python
import wandb

# 1. Initialize
wandb.init(project="llm-finetuning", config={
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32
})

# 2. Access Config
config = wandb.config

# 3. Log Training
for epoch in range(config.epochs):
    loss = 0.5 * (10 - epoch) # Dummy loss
    
    # Log metrics
    wandb.log({"loss": loss, "epoch": epoch})
    
    # Log Artifacts (Images/Text)
    if epoch % 5 == 0:
        table = wandb.Table(columns=["prompt", "generation"])
        table.add_data("Hello", "World")
        wandb.log({"examples": table})

# 4. Finish
wandb.finish()
```

### 5. Reproducibility Wrapper

A decorator to ensure reproducibility.

```python
import random
import numpy as np
import torch
import os

def make_reproducible(seed=42):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["PYTHONHASHSEED"] = str(seed)
            
            print(f"Random seeds set to {seed}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@make_reproducible(seed=123)
def run_experiment():
    print(torch.randn(1))

run_experiment()
```

### 6. Logging LLM Prompts (MLflow)

```python
# MLflow 2.3+ supports LLM tracking
import mlflow.llm

with mlflow.start_run():
    # Log the prompt template
    mlflow.log_param("prompt_template", "Summarize this: {text}")
    
    # Log inputs and outputs
    data = [
        {"text": "Long article...", "summary": "Short summary"},
        {"text": "Another article...", "summary": "Another summary"}
    ]
    
    mlflow.llm.log_predictions(data, artifact_path="predictions")
```
