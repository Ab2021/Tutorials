# Lab 3: Model Registry

## Objective
Don't lose your models.
Build a simple **Model Registry** API.

## 1. The API (`registry.py`)

```python
models = {} # {name: [v1, v2]}

def register(name, path, metrics):
    if name not in models:
        models[name] = []
    
    version = len(models[name]) + 1
    entry = {
        "version": version,
        "path": path,
        "metrics": metrics,
        "stage": "staging"
    }
    models[name].append(entry)
    print(f"Registered {name} v{version}")

def promote(name, version, stage):
    models[name][version-1]['stage'] = stage
    print(f"Promoted {name} v{version} to {stage}")

# Usage
register("llama-finetune", "s3://bucket/v1", {"acc": 0.8})
register("llama-finetune", "s3://bucket/v2", {"acc": 0.85})

promote("llama-finetune", 2, "production")
```

## 2. Analysis
Tools like MLflow or W&B do this automatically.
Key concept: **Immutability** of artifacts.

## 3. Submission
Submit the JSON dump of the registry.
