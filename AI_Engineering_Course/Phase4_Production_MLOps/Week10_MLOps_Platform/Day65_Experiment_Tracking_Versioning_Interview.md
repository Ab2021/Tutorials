# Day 65: Experiment Tracking & Versioning
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why isn't Git enough for versioning ML models?

**Answer:**
- **Size:** Git is designed for text/code. It chokes on large binary files (100GB model weights). Git LFS helps, but isn't optimized for ML workflows.
- **Metadata:** Git tracks *changes*, but not *metrics* (Accuracy, Loss). You can't query Git to "find the commit that gave 95% accuracy".
- **Artifacts:** ML experiments produce plots, logs, and checkpoints that need to be associated with a specific run, not just the code.

#### Q2: What is the difference between Model Versioning and Data Versioning?

**Answer:**
- **Model Versioning:** Tracking the *output* of the training process (weights, architecture config). Managed by Model Registry (MLflow).
- **Data Versioning:** Tracking the *input* to the training process (datasets, cleaning scripts). Managed by Data Version Control (DVC).
- **Link:** A reproducible run requires linking `Model v5` to `Data v3`.

#### Q3: How do you handle non-deterministic GPU operations in reproducibility?

**Answer:**
- **Seeds:** Set seeds for Python, Numpy, Torch.
- **CuDNN:** Set `torch.backends.cudnn.deterministic = True`. This disables some fast but non-deterministic kernels.
- **Hardware:** Even with settings, floating point accumulation order can vary across different GPU architectures (A100 vs V100).
- **Acceptance:** In production, we often accept "statistical reproducibility" (within variance) rather than bit-exact reproducibility.

#### Q4: Explain the "Staging" vs "Production" model stages.

**Answer:**
- **Staging:** A candidate model that has passed offline evaluation (test set). It is deployed to a shadow/canary environment for integration testing.
- **Production:** The live model serving real user traffic.
- **Promotion:** Moving a model from Staging to Production involves manual approval or automated gates (e.g., "Latency < 50ms" AND "Accuracy > Baseline").

#### Q5: What is a Hyperparameter Sweep?

**Answer:**
- Automated process of searching for the best combination of hyperparameters (LR, Batch Size, Dropout).
- **Methods:** Grid Search (slow), Random Search (better), Bayesian Optimization (smart).
- **Tooling:** W&B Sweeps, Ray Tune, Optuna.

---

### Production Challenges

#### Challenge 1: "It works on my machine" (Environment Drift)

**Scenario:** Model trained on Dev laptop fails on Prod server.
**Root Cause:** Different library versions (PyTorch 1.13 vs 2.0), CUDA drivers, or OS.
**Solution:**
- **Docker:** Train in the exact same Docker container that will be used for inference.
- **MLflow Projects:** Captures `conda.yaml` or `requirements.txt` automatically.
- **Lock Files:** Use `poetry.lock` or `pip freeze` to pin exact versions.

#### Challenge 2: Lost Weights

**Scenario:** Researcher trains a great model, closes the notebook, and loses the weights.
**Root Cause:** No auto-logging.
**Solution:**
- **Auto-save:** Configure training loop to save checkpoints to S3/MLflow every epoch.
- **Artifact Store:** Never store weights on local disk only.

#### Challenge 3: Metric Divergence

**Scenario:** Training Loss goes down, but Production Metric (User Satisfaction) goes down.
**Root Cause:** Optimization metric (Cross Entropy) is not aligned with Business metric.
**Solution:**
- **Proxy Metrics:** Track multiple metrics during training (Perplexity, BLEU, Toxicity).
- **Online Eval:** Correlate offline metrics with online A/B test results to find a better proxy.

#### Challenge 4: Storage Cost Explosion

**Scenario:** MLflow server disk is full.
**Root Cause:** Saving a 10GB checkpoint every 100 steps for 100 experiments.
**Solution:**
- **Retention Policy:** Delete artifacts older than 30 days unless tagged "Production".
- **Best-Only:** Only save the "Best" checkpoint (lowest val loss) and "Latest" checkpoint.
- **Remote Storage:** Use S3/GCS for artifacts, not local disk.

#### Challenge 5: Concurrent Experiment Conflicts

**Scenario:** Two engineers tune the same model and overwrite each other's tags.
**Root Cause:** Shared experiment ID.
**Solution:**
- **Unique Runs:** Each run gets a UUID.
- **User Tags:** Automatically tag runs with `user_id`.
- **Branching:** Use Git branches for experimental code.

### System Design Scenario: MLOps Platform for a Startup

**Requirement:** Build a platform for 5 Data Scientists to train and deploy LLMs.
**Design:**
1.  **Tracking:** Hosted MLflow or W&B.
2.  **Storage:** S3 bucket for artifacts.
3.  **Compute:** Kubernetes cluster (or Slurm) for training jobs.
4.  **Orchestration:** Airflow/Prefect to trigger training.
5.  **Registry:** MLflow Model Registry.
6.  **CI/CD:** GitHub Actions triggers training on merge, promotes to Staging on success.

### Summary Checklist for Production
- [ ] **Tracking:** Log **all** params and metrics.
- [ ] **Artifacts:** Save checkpoints to **S3/Remote**.
- [ ] **Environment:** Use **Docker** for reproducibility.
- [ ] **Registry:** Use **Model Stages** (Staging/Prod).
- [ ] **Cleanup:** Implement **Retention Policies** for old runs.
