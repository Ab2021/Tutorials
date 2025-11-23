# Day 65: Experiment Tracking & Versioning
## Core Concepts & Theory

### The Experimentation Crisis

**Problem:** "Which hyperparameters worked best last week?" "Where is the model checkpoint that generated this output?"
- **Ad-hoc:** Spreadsheets, random filenames (`model_final_v2_real.pt`).
- **Reproducibility:** Impossible to recreate results.
- **Collaboration:** Team members overwrite each other's work.

**Solution:** Structured Experiment Tracking and Model Versioning.

### 1. Experiment Tracking

**What to Track:**
- **Config:** Hyperparameters (LR, Batch Size, LoRA rank).
- **Code:** Git commit hash.
- **Data:** Dataset version hash (DVC).
- **Metrics:** Loss, Accuracy, Perplexity, BLEU, ROUGE.
- **Artifacts:** Checkpoints, Logs, Sample Generations.
- **Environment:** Docker image, Python requirements.

**Tools:**
- **MLflow:** Open source, industry standard.
- **Weights & Biases (W&B):** Popular for visualization and collaboration.
- **Comet ML / Neptune:** Alternatives.

### 2. Model Registry

**Concept:** Central repository for managing model lifecycle.
- **Stages:** None -> Staging -> Production -> Archived.
- **Versioning:** `v1`, `v2`, `v1.1`.
- **Metadata:** Who trained it? On what data? Accuracy metrics?
- **Approval:** Manual or automated gate before promotion to Production.

### 3. The MLflow Ecosystem

**Components:**
1.  **Tracking:** Logging parameters and metrics.
2.  **Projects:** Packaging code for reproducibility.
3.  **Models:** Standard format for packaging models.
4.  **Registry:** Central model store.

### 4. Weights & Biases (W&B) Features

- **Live Dashboards:** Watch loss curves in real-time.
- **Comparison:** Overlay runs to see impact of hyperparameters.
- **System Metrics:** GPU usage, Memory, Temperature.
- **Reports:** Markdown reports with embedded charts for stakeholders.

### 5. Reproducibility Checklist

To ensure a run is reproducible:
1.  **Seed:** Set random seeds (Python, Numpy, Torch).
2.  **Code:** Commit uncommitted changes or log the diff.
3.  **Data:** Version the data (DVC).
4.  **Env:** Capture `requirements.txt` or Docker container.
5.  **Hardware:** Note GPU type (determinism varies across architectures).

### 6. Hyperparameter Sweeps

**Concept:** Automated search for best parameters.
- **Grid Search:** Exhaustive.
- **Random Search:** Efficient.
- **Bayesian Optimization:** Smart search (W&B Sweeps, Optuna).
- **Tracking:** Each sweep run is an experiment. Visualization helps identify "good regions".

### 7. Prompt Versioning (LLM Specific)

**Prompts are Hyperparameters:**
- Changing the system prompt is like changing the model weights.
- **Track:** Prompt Template, Temperature, Top-P.
- **Tools:** W&B Prompts, MLflow LLM Tracking.

### 8. Distributed Training Tracking

**Challenge:** Multiple GPUs logging to the same server.
- **Strategy:** Only Rank 0 logs metrics. All ranks log system metrics (to detect load imbalance).

### 9. A/B Testing Integration

**Flow:**
1.  Train `Model A` and `Model B`.
2.  Register both in Model Registry.
3.  Deploy both to endpoint.
4.  Log **Inference Traces** linking back to Model Version.
5.  Analyze which version performs better in production.

### 10. Best Practices

- **Log Everything:** Storage is cheap. Retraining is expensive.
- **Tagging:** Tag runs with `experiment_id`, `user`, `purpose`.
- **Clean Up:** Delete failed/garbage runs periodically.
- **One Config Source:** Use `config.yaml` or `Hydra`, don't hardcode params.

### Summary

**Tracking Strategy:**
1.  **Tool:** Use **MLflow** or **W&B**.
2.  **Scope:** Track **Code, Data, Config, Metrics, Artifacts**.
3.  **Registry:** Use **Model Registry** for lifecycle management.
4.  **Reproducibility:** Fix **Seeds** and **Environment**.
5.  **LLMs:** Track **Prompts** and **Generation Samples**.

### Next Steps
In the Deep Dive, we will implement a full MLflow tracking pipeline, including a hyperparameter sweep and model registry promotion workflow.
