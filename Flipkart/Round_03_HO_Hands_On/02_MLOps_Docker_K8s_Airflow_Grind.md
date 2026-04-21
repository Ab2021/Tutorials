# 🔧 HO GRIND: MLOps, Docker, Kubernetes & Airflow Operations
### What they will ask about your deployment stack

> Your resume lists: MLflow, Kubeflow, Docker, Kubernetes, CI/CD (GitHub Actions), Airflow, Model Monitoring, A/B Testing. Expect operational deep-dives.

---

## ═══════════════════════════════════════
## SECTION 1: DOCKER — Interview Questions & Answers
## ═══════════════════════════════════════

### Q1: "Walk me through a production Dockerfile for serving an ML model."
```dockerfile
# Multi-stage build to keep image small
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY models/ ./models/

# Non-root user for security
RUN useradd -m appuser
USER appuser

# Health check
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["gunicorn", "src.app:app", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "120"]
```

**Key talking points:**
- **Multi-stage build:** Builder stage installs deps. Runtime stage copies only the installed packages. Image goes from ~2GB to ~400MB.
- **Non-root user:** Security best practice. Prevents container escape attacks.
- **Health check:** K8s liveness probe hits `/health`. If model loading fails, container restarts.
- **Gunicorn workers:** 4 workers handle concurrent requests. For GPU models, use 1 worker + async (uvicorn).

### Q2: "What's the difference between COPY and ADD? Between CMD and ENTRYPOINT?"
- **COPY vs ADD:** COPY is simple file copy. ADD can auto-extract tarballs and fetch URLs — avoid it (unpredictable side effects, always use COPY).
- **CMD vs ENTRYPOINT:** ENTRYPOINT = the binary that always runs. CMD = default arguments. If user passes args at `docker run`, CMD is overridden but ENTRYPOINT is not. Best practice: `ENTRYPOINT ["python"]` + `CMD ["serve.py"]` — user can override to `CMD ["train.py"]` without changing the image.

### Q3: "Your Docker image with PyTorch + CUDA is 8GB. How do you reduce it?"
- Use NVIDIA's official base images (`nvcr.io/nvidia/pytorch:latest`) which are optimized.
- Multi-stage build: install deps in builder, copy only needed artifacts.
- `.dockerignore`: exclude `.git/`, `data/`, `notebooks/`, `__pycache__/`.
- Pin exact versions (avoid pulling unnecessary transitive deps).
- If serving only (no training), install `torch` CPU-only version (500MB vs 2GB).

---

## ═══════════════════════════════════════
## SECTION 2: KUBERNETES — Interview Questions & Answers
## ═══════════════════════════════════════

### Q4: "Explain how you'd deploy an ML model on K8s with autoscaling."
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-model
  template:
    metadata:
      labels:
        app: fraud-model
    spec:
      containers:
      - name: model-server
        image: gcr.io/my-project/fraud-model:v2.3
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
---
# HPA - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Key talking points:**
- **Liveness vs Readiness Probe:** Liveness = "is the container still alive?" (restart if dead). Readiness = "is it ready to serve traffic?" (don't send requests until the model is loaded into memory). ML models can take 30s+ to load.
- **Resource requests vs limits:** Requests = guaranteed minimum (scheduler uses for placement). Limits = hard ceiling (OOMKilled if exceeded). For ML: set memory limit generously—models can have unpredictable memory spikes.
- **Rolling update:** K8s default strategy. Spin up new pods with v2.4, health-check them, then drain old v2.3 pods. Zero downtime.

### Q5: "How do you do Canary Deployments of a new model version?"
Use Istio/nginx-ingress traffic splitting:
- Route 95% traffic to `fraud-model:v2.3` (champion)
- Route 5% traffic to `fraud-model:v2.4` (canary)
- Monitor Precision@K, P99 latency, error rate on the canary dashboard
- If canary metrics are equal or better after 24 hours, gradually increase to 25% → 50% → 100%
- If degradation: instant rollback by shifting traffic split back to 100% v2.3

---

## ═══════════════════════════════════════
## SECTION 3: AIRFLOW — Interview Questions & Answers
## ═══════════════════════════════════════

### Q6: "Design an Airflow DAG for a daily fraud model retraining pipeline."
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitJobOperator,
    DataprocDeleteClusterOperator,
)
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'email_on_failure': True,
    'email': ['ds-alerts@company.com'],
}

with DAG(
    'fraud_model_retrain_daily',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,  # Don't backfill on DAG creation
    max_active_runs=1,
    tags=['ml', 'fraud', 'production'],
) as dag:
    
    create_cluster = DataprocCreateClusterOperator(
        task_id='create_ephemeral_cluster',
        cluster_name='fraud-retrain-{{ ds_nodash }}',
        num_workers=4,
        worker_machine_type='n1-highmem-8',
        # AUTO-DELETE after 3 hours (safety net)
        idle_delete_ttl=10800,
    )
    
    run_feature_eng = DataprocSubmitJobOperator(
        task_id='feature_engineering',
        job={'pyspark_job': {
            'main_python_file_uri': 'gs://ml-pipelines/fraud/feature_eng.py',
            'args': ['--date', '{{ ds }}'],
        }},
    )
    
    run_training = DataprocSubmitJobOperator(
        task_id='model_training',
        job={'pyspark_job': {
            'main_python_file_uri': 'gs://ml-pipelines/fraud/train.py',
            'args': ['--date', '{{ ds }}', '--output', 'gs://models/fraud/{{ ds }}'],
        }},
    )
    
    validate_model = PythonOperator(
        task_id='validate_model_quality',
        python_callable=validate_champion_challenger,
        op_kwargs={'model_path': 'gs://models/fraud/{{ ds }}'},
    )
    
    delete_cluster = DataprocDeleteClusterOperator(
        task_id='delete_cluster',
        cluster_name='fraud-retrain-{{ ds_nodash }}',
        trigger_rule='all_done',  # Delete even if upstream fails!
    )
    
    create_cluster >> run_feature_eng >> run_training >> validate_model >> delete_cluster
```

**Key talking points:**
- **`catchup=False`:** Prevents Airflow from running all historical dates on DAG creation. Critical for ML pipelines.
- **`trigger_rule='all_done'`:** The cluster deletion task runs even if training fails. This prevents orphaned clusters burning cloud spend.
- **Idempotent tasks:** Feature engineering writes to `gs://features/{{ ds }}/` — rerunning the same date overwrites (idempotent). No duplicates.
- **`max_active_runs=1`:** Prevents parallel runs from competing for resources.

### Q7: "How do you handle Airflow backfills for ML pipelines?"
If the pipeline was down for 3 days and you need to catch up:
```bash
airflow dags backfill fraud_model_retrain_daily \
  --start-date 2025-04-18 \
  --end-date 2025-04-20
```
Because tasks use `{{ ds }}` (execution date) for data partitioning, each backfill run processes only its date partition. No data overlap or duplication.

---

## ═══════════════════════════════════════
## SECTION 4: MLflow & EXPERIMENT TRACKING
## ═══════════════════════════════════════

### Q8: "What is the MLflow Model Registry and how does Champion-Challenger work?"
```python
import mlflow

# Log experiment run
with mlflow.start_run(run_name="fraud_xgb_v2.4"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({"pr_auc": 0.72, "ks_stat": 0.48, "ece": 0.03})
    mlflow.xgboost.log_model(model, "model")
    
    # Register to Model Registry
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "fraud_detector"
    )

# Transition stages
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud_detector",
    version=4,
    stage="Staging"  # Options: None, Staging, Production, Archived
)

# Champion-Challenger:
# 1. New model registers as "Staging"
# 2. Shadow scoring: run Staging model on live data alongside Production model
# 3. Compare metrics over 1 week
# 4. If Staging PR-AUC >= Production PR-AUC - 0.01: Promote to Production
# 5. Archive old Production version
```

### Q9: "How do you detect model drift in production?"
Three types of drift:
1. **Data Drift (Covariate Shift):** Feature distributions change. Detect with KS test or PSI (Population Stability Index) on each feature weekly. Alert if PSI > 0.25.
2. **Concept Drift:** The relationship between features and target changes. Detect by monitoring production model metrics (PR-AUC, Precision@K) on delayed labels. Alert if metric drops >5%.
3. **Label Drift:** Target distribution changes (fraud rate increases). Detect by monitoring the actual positive rate vs. model predicted positive rate.

```python
def compute_psi(expected, actual, buckets=10):
    """Population Stability Index for drift detection."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pcts = np.histogram(expected, np.percentile(expected, breakpoints))[0] / len(expected)
    actual_pcts = np.histogram(actual, np.percentile(expected, breakpoints))[0] / len(actual)
    
    # Avoid log(0)
    expected_pcts = np.clip(expected_pcts, 0.001, None)
    actual_pcts = np.clip(actual_pcts, 0.001, None)
    
    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return psi
    # PSI < 0.1: No significant shift
    # 0.1 < PSI < 0.25: Moderate shift, investigate
    # PSI > 0.25: Significant shift, retrain!
```

---

*This file covers the operational MLOps stack that interviewers will drill into when they see Docker, Kubernetes, Airflow, and MLflow on your resume.*
