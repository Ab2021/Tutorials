# Day 66: Model Registry & Artifact Management
## Core Concepts & Theory

### The Governance Layer

**Problem:** You have 1000 trained models. Which one is currently in production? Which one was deployed last month? Which one is safe to delete?
- **Experiment Tracking:** Logs *everything* (Research phase).
- **Model Registry:** Manages *candidates* (Engineering phase).

**Role of Registry:**
- **Central Source of Truth:** The definitive record of available models.
- **Lifecycle Management:** State machine for models (Dev -> Staging -> Prod -> Archived).
- **Governance:** Who approved this model? Did it pass security checks?

### 1. Artifact Management

**Artifacts:** Binary files produced by training.
- Model Weights (`pytorch_model.bin`, `model.safetensors`).
- Tokenizer Files (`tokenizer.json`, `vocab.txt`).
- Config Files (`config.json`).
- Evaluation Reports (`eval_results.json`).

**Storage Strategy:**
- **Object Storage:** S3, GCS, Azure Blob (Standard for scalability).
- **Directory Structure:** `models/{model_name}/{version}/{artifact_file}`.
- **Immutability:** Once a version is published, artifacts should never change.

### 2. Model Packaging

**Containerization:**
- **Model + Code + Env:** Package the model weights with the inference code (`predict.py`) and dependencies (`Dockerfile`).
- **Formats:**
  - **MLflow Models:** Standard flavor.
  - **TorchServe (.mar):** Archive format.
  - **ONNX:** Interoperable format.

### 3. Metadata & Lineage

**What to store in Registry:**
- **Hyperparameters:** Link to the experiment run.
- **Dataset Hash:** Link to the training data.
- **Metrics:** Validation accuracy, latency.
- **Schema:** Input/Output signature (e.g., `{"text": string} -> {"label": string}`).

### 4. Promotion Gates (CI/CD for Models)

**Automated Gates:**
1.  **Format Check:** Is the file valid?
2.  **Smoke Test:** Does it run without crashing?
3.  **Performance Test:** Is Accuracy > Threshold?
4.  **Latency Test:** Is P99 Latency < Threshold?
5.  **Security Scan:** Are there malicious pickles?

**Manual Gates:**
- "Approved by Lead Data Scientist".

### 5. Model Serving Integration

**Pull vs Push:**
- **Pull:** Serving service polls Registry for "Production" tag. Updates automatically.
- **Push:** CI/CD pipeline triggers deployment when tag changes.

### 6. Archival & Cleanup

**Cost Management:**
- LLM weights are huge (100GB+).
- **Policy:** Keep "Production" and last 3 versions. Delete "Staging" older than 1 week. Archive "Dev" immediately.

### 7. Security (Model Signing)

**Supply Chain Security:**
- **Signing:** Sign model artifacts with a private key.
- **Verification:** Inference server verifies signature before loading.
- **Prevents:** Man-in-the-middle attacks, malicious model injection.

### 8. Safetensors

**Why Pickle is Bad:**
- Python `pickle` allows arbitrary code execution.
- **Safetensors:** A safe, fast, zero-copy format for storing tensors.
- **Standard:** HuggingFace default.

### 9. Tools

- **MLflow Registry:** Integrated with tracking.
- **AWS SageMaker Registry:** Cloud native.
- **HuggingFace Hub:** Public/Private registry with versioning.

### Summary

**Registry Strategy:**
1.  **Centralize:** All models go to **Model Registry**.
2.  **Storage:** Use **S3** for artifacts.
3.  **Format:** Use **Safetensors** for security.
4.  **Lifecycle:** Implement **Staging -> Production** gates.
5.  **Automation:** Use **CI/CD** to run tests before promotion.

### Next Steps
In the Deep Dive, we will implement a Model Registry interaction script, a CI/CD gate for model promotion, and a Safetensors conversion script.
