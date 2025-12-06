# Day 130: The Vault: Model Artifacts & Storage
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 19: Experiment Tracking & Model Registry

---

> **🎯 Focus Area:** A "Model" is not just weights. It's the Schema, the Dependencies (`requirements.txt`), the Code, and the Config. Storing just `model.pt` leads to "It works on my machine" hell.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** serialization formats: Pickle (Dangerous) vs ONNX (Portable) vs SafeTensors (Fast/Safe).
2.  **Log** custom artifacts (JSON config, confusion matrix PNG) alongside model weights.
3.  **Use** MLflow's `log_model` specific flavors (`python_function` wrapper).
4.  **Configure** S3/MinIO for efficient multipart uploads of 10GB+ checkpoints.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install mlflow onnx safetensors boto3`.

---

## 📖 Theoretical Foundation

### 1. The Serialization War
*   **Pickle (`torch.save`):** Python-specific. Executes arbitrary code on load (Security Risk). Brittle (requires exact same class definition).
*   **ONNX (Open Neural Network Exchange):** Graph format. Language agnostic (Run in C++, JS). Optimized inference.
*   **SafeTensors:** HuggingFace format. Zero-copy memory mapping. No code execution. Safe.

### 2. The MLflow Model Format
When you log a model in MLflow, it creates a folder:
```
my_model/
├── MLmodel            # YAML metadata (flavors, run_id)
├── conda.yaml         # Environment definition
├── model.pkl          # Physical binary
└── python_env.yaml    # Python version
```
This allows `mlflow models serve` to rebuild the environment automatically.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Logging Custom Flavors

#### 📁 `src/05_custom_artifacts.py`
```python
import mlflow
import json
import torch
from safetensors.torch import save_file
import os

# Create dummy model
model = torch.nn.Linear(10, 2)
weights = model.state_dict()

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Artifacts_Demo"
mlflow.set_experiment(experiment_name)

class CustomWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Called when model is loaded via pyfunc
        self.config = json.load(open(context.artifacts["config"]))
    
    def predict(self, context, model_input):
        return f"Predicted with thresh {self.config['thresh']}"

with mlflow.start_run():
    # 1. Save locally first
    os.makedirs("output", exist_ok=True)
    
    # A. JSON Config
    config = {"thresh": 0.5, "architecture": "Linear"}
    with open("output/config.json", "w") as f:
        json.dump(config, f)
        
    # B. SafeTensors (Weights)
    save_file(weights, "output/model.safetensors")
    
    # 2. Log General Artifacts (Just files)
    mlflow.log_artifact("output/config.json", artifact_path="metadata")
    mlflow.log_artifact("output/model.safetensors", artifact_path="weights")
    
    # 3. Log as PyFunc Model (Executable)
    # This packs the logic + artifacts together
    mlflow.pyfunc.log_model(
        artifact_path="model_package",
        python_model=CustomWrapper(),
        artifacts={"config": "output/config.json"}
    )
    
    print("Run Complete.")
```

### 👨‍💻 Core Implementation: ONNX Conversion

```python
import torch.onnx

# Convert
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "output/model.onnx")

# Log ONNX
import mlflow.onnx
with mlflow.start_run():
    mlflow.onnx.log_model(
        onnx_model=onnx.load("output/model.onnx"),
        artifact_path="onnx_model"
    )
```

---

## 🔬 Lab Exercise: "Dependency Hell"

### Task
Reproduce the environment.
1.  Navigate to MLflow UI artifacts.
2.  Download `conda.yaml`.
3.  **Observation:** It captures `torch==2.0.1`, `numpy==1.24.3`.
4.  Try to load the pickle model in a fresh environment *without* torch. It fails ("No module named 'torch'").
5.  Try to serve the ONNX model in an environment with only `onnxruntime` (No torch).
6.  **Success:** ONNX decouples the model from the training framework.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Immutability:** Once an artifact is uploaded to the Run, it should never be overwritten. If you retrain, create a NEW Run.
2.  **Storage Costs:** Deep Learning models are huge (GBs). Configure Lifecycle Rules on S3 (e.g., Delete artifacts older than 30 days unless marked `Production`).
3.  **Security:** Prefer ONNX/SafeTensors for Production. Use Pickle only for internal research reproducibility.

### API Summary
```python
mlflow.log_artifact(local_path, remote_path)
mlflow.log_artifacts(local_dir)
```

---

**Day 130 Complete** ✅

*Next: Day 131 - Experiment Comparison & Analysis.*
