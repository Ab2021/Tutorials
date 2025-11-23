# Day 67: Continuous Training (CT) Pipelines
## Core Concepts & Theory

### The Need for CT

**Problem:** Models decay.
- **Data Drift:** User behavior changes (e.g., new slang, new products).
- **Concept Drift:** The relationship between input and output changes.
- **Static Models:** A model trained in 2023 doesn't know about 2024 events.

**Solution:** Continuous Training (CT).
- Automating the Retraining -> Evaluation -> Deployment loop.

### 1. CT Pipeline Architecture

**Stages:**
1.  **Data Ingestion:** Fetch new data (Feedback loop, new logs).
2.  **Data Validation:** Check for drift/quality.
3.  **Preprocessing:** Clean and format.
4.  **Training (Fine-tuning):** Run LoRA/Full fine-tuning.
5.  **Evaluation:** Compare against "Champion" model.
6.  **Registration:** Register "Challenger" model.
7.  **Deployment:** Promote if Challenger > Champion.

### 2. Triggers for Retraining

**Schedule-Based:**
- Retrain every Sunday night.
- **Pros:** Predictable.
- **Cons:** Might retrain when unnecessary or too late.

**Metric-Based:**
- Retrain when `Accuracy < Threshold` or `Drift > Threshold`.
- **Pros:** Responsive.
- **Cons:** Hard to define thresholds.

**Data-Volume Based:**
- Retrain when `New Samples > 10,000`.

### 3. Orchestration Tools

**Airflow:**
- DAG-based workflow. Industry standard.
- **Pros:** Mature, huge ecosystem.
- **Cons:** Heavyweight.

**Kubeflow Pipelines (KFP):**
- Kubernetes-native.
- **Pros:** Great for containerized ML steps.
- **Cons:** Complex setup.

**Prefect / Dagster:**
- Modern, pythonic alternatives.

### 4. Parameter-Efficient CT (LoRA)

**Why LoRA for CT?**
- **Speed:** Fine-tuning takes minutes instead of days.
- **Cost:** Cheap compute.
- **Storage:** Save only adapter weights (100MB vs 100GB).
- **Strategy:** Keep base model frozen. Periodically train a new LoRA adapter on recent data.

### 5. Evaluation in CT

**The Gatekeeper:**
- You must *never* deploy a retrained model without rigorous evaluation.
- **Offline Eval:** Test set metrics.
- **Shadow Deployment:** Run in parallel with Prod, compare outputs.
- **Canary:** Deploy to 1% users.

### 6. Data Freshness vs Stability

**Trade-off:**
- **Freshness:** Train on last 24h data. (Model knows latest news).
- **Stability:** Train on last 1 year data. (Model is robust, doesn't overfit to yesterday's noise).
- **Solution:** Sliding window (e.g., train on last 30 days).

### 7. Catastrophic Forgetting

**Risk:**
- New model learns new data but forgets old knowledge.
- **Mitigation:**
  - **Replay Buffer:** Mix 10% old data with new data.
  - **EWC (Elastic Weight Consolidation):** Regularization term.

### 8. Infrastructure

**Training Store:**
- Feature Store serves point-in-time correct data for training.

**Compute:**
- Spot instances for training jobs (cost saving).

### 9. Human-in-the-Loop (HITL)

**Safety Valve:**
- Even with CT, have a human review the evaluation report before the final "Promote to Prod" button press.

### 10. Summary

**CT Strategy:**
1.  **Trigger:** Schedule or Metric driven.
2.  **Method:** **LoRA** for efficiency.
3.  **Orchestrator:** **Airflow** or **Kubeflow**.
4.  **Eval:** Automated **Golden Set** evaluation.
5.  **Safety:** **Shadow Mode** before full rollout.

### Next Steps
In the Deep Dive, we will implement a simple CT pipeline using Python/Airflow concepts and a Drift Detection trigger.
