# MLOPS PIPELINE DEEP DIVE — Interview Mastery
> Critical gap: You say "we use MLflow / Kubernetes" but can't explain what happens inside each step. This file fixes that.

---

## SECTION 1: THE END-TO-END MLOPS PIPELINE

### Complete Pipeline (left to right)
```
Code Commit
    ↓
CI Pipeline (unit tests, integration tests, code quality)
    ↓
Artifact Build (Docker image, .whl package)
    ↓
Model Registry (versioning, staging)
    ↓
CD Pipeline (deploy to staging)
    ↓
Shadow/A-B Testing
    ↓
Production Deploy (Kubernetes, HPA)
    ↓
Monitoring (PSI, performance metrics, latency)
    ↓
Drift Alert
    ↓
Retrain Pipeline (automated or on-demand)
    ↓ (loops back to Code Commit)
```

### Responsibilities — Who Does What

| Stage | Data Scientist | MLOps / Platform Engineer |
|-------|---------------|---------------------------|
| Code Commit | Write model training code, feature pipeline, FastAPI endpoint | Set up Git hooks, branch protection |
| CI Pipeline | Write unit tests, integration tests | Set up CI runner (GitHub Actions, Jenkins) |
| Artifact Build | Write Dockerfile, requirements.txt | Manage Docker registry, build optimization |
| Model Registry | Register model with MLflow, set metrics | Manage MLflow server, storage |
| CD Pipeline | Define validation criteria | Automate deployment to K8s |
| Monitoring | Define KPIs, set alert thresholds | Build dashboards (Grafana), integrate Prometheus |
| Retrain | Retrain pipeline code | Schedule retrain (Airflow), resource management |

---

## SECTION 2: CI/CD FOR ML — DEEP DIVE

### CI — What Data Scientists Are Responsible For

**Unit Testing:**
- Test each function in isolation with expected inputs and outputs
- Example: `test_psi_calculation()` — given known distributions, verify PSI formula output
- Example: `test_woe_encoding()` — given training data, verify WoE values for known category
- Tools: `pytest`, `unittest`

```python
# Example unit test for PSI
def test_psi_stable_distributions():
    expected = [0.1, 0.2, 0.3, 0.4]
    actual = [0.1, 0.2, 0.3, 0.4]  # Same distribution
    psi = calculate_psi(expected, actual)
    assert psi < 0.1, f"PSI should be ~0 for identical distributions, got {psi}"

def test_psi_drifted_distributions():
    expected = [0.5, 0.3, 0.1, 0.1]
    actual = [0.1, 0.1, 0.3, 0.5]   # Very different
    psi = calculate_psi(expected, actual)
    assert psi > 0.2, f"PSI should signal significant drift, got {psi}"
```

**Integration Testing:**
- Test full pipeline: data input → feature engineering → model prediction → output validation
- Example: given sample claim data, full scoring pipeline returns valid fraud_probability in [0,1]
- Check: no null outputs, correct schema, reasonable value ranges

**Code Quality (SonarQube / flake8):**
- **SOLID Principles in ML code:**
  - **S**ingle Responsibility: FeatureEngineer class only does features, not model training
  - **O**pen/Closed: add new features by subclassing, not modifying existing code
  - **L**iskov Substitution: XGBoostModel and LightGBMModel both inherit BaseModel, interchangeable
  - **I**nterface Segregation: don't force classes to implement methods they don't need
  - **D**ependency Inversion: depend on abstractions (BaseModel), not concrete implementations

- **No nested functions:** Extract complex logic to named methods (makes testing possible)
- **Docstrings and type hints:** Every public function must have these
- **Test coverage > 80%:** Lines of code touched by tests / total lines

**What Abhishek packages:**
1. **Python wheel (.whl):** `pip install model-package-1.0.0-py3-none-any.whl`
   - Contains: feature engineering code, model loading, inference function
   - MLOps team imports this into their serving container
2. **FastAPI app in Docker:** Complete REST API in a container
   - `POST /score` endpoint that loads model from MLflow registry and returns predictions

**Interview answer for CI responsibilities:**
> "My responsibility as data scientist in CI: I write unit tests (pytest) for each feature transformation function and integration tests for the full scoring pipeline. I follow SOLID principles — each class has one job, BaseModel interface lets MLOps swap XGBoost for LightGBM without changing serving code. I run SonarQube locally to ensure >80% test coverage before push. I also validate data schemas in tests so input drift is caught before it reaches production."

---

### CD — What MLOps Does (You Should Know This to Talk Intelligently About It)

**Docker Build and Push:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model_api/ .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Deployment Strategies:**

**Blue-Green Deployment:**
- Run old model (blue) and new model (green) simultaneously
- Route 100% traffic to blue, test green on shadow traffic
- If green passes: switch load balancer to route 100% to green
- If green fails: no downtime (blue still serves), just don't switch
- Used when: zero-downtime deployment required

**Canary Deployment:**
- Route 5% of traffic to new model, 95% to old model
- Monitor metrics for 24-48 hours
- If metrics hold: increase to 20% → 50% → 100%
- If metrics degrade: immediately revert to 100% old model
- Used when: want real traffic validation before full rollout

**Shadow Deployment:**
- New model gets all requests, but its predictions are NOT used for decisions
- Compare new vs old model predictions offline
- Zero risk: user never sees new model's output
- Used when: validating new model on production traffic pattern before any exposure

**Rollback:**
- MLflow Model Registry maintains all model versions
- Rollback = change registry pointer from "Production" to previous version
- Kubernetes deployment automatically restarts pods with previous Docker image
- Time to rollback: < 5 minutes with automated pipeline

---

## SECTION 3: MLFLOW — WHAT IT ACTUALLY DOES (STEP BY STEP)

### The 4 MLflow Components

**1. MLflow Tracking — Experiment Logging**

Every training run is logged:
```python
import mlflow

with mlflow.start_run(run_name="xgboost_v2_tuned"):
    # Log hyperparameters
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("scale_pos_weight", 49)
    
    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("pr_auc_val", pr_auc)
    mlflow.log_metric("precision_70recall", precision_at_70)
    mlflow.log_metric("capture_at_25", capture_25)
    
    # Log model artifact
    mlflow.xgboost.log_model(model, "model")
    
    # Log plots as artifacts
    mlflow.log_artifact("pr_curve.png")
    mlflow.log_artifact("feature_importance.html")
```

What gets stored: run metadata, parameters, metrics, model file, plots, custom files.
What you can do: compare runs in UI, filter by metric, reproduce any experiment by run_id.

**2. MLflow Projects — Reproducible Code**
```yaml
# MLproject file
name: fraud-model-training
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      max_depth: {type: int, default: 6}
      learning_rate: {type: float, default: 0.1}
    command: "python train.py --max-depth {max_depth} --learning-rate {learning_rate}"
```
Anyone can reproduce your run: `mlflow run . -P max_depth=8`

**3. MLflow Models — Standard Serialization**
```python
# Save with signature (input/output schema)
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, model.predict_proba(X_train))
mlflow.xgboost.log_model(model, "model", signature=signature)
```
Loaded as different "flavors":
- `mlflow.xgboost.load_model(uri)` → XGBoost native object
- `mlflow.pyfunc.load_model(uri)` → generic Python function (for serving)

**4. MLflow Model Registry — Lifecycle Management**

Stages flow:
```
None → Staging → Production → Archived
```

```python
# Programmatic promotion
client = mlflow.MlflowClient()

# Register model from a training run
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "fraud-model-xgboost"
)

# Transition to Staging after validation
client.transition_model_version_stage(
    name="fraud-model-xgboost",
    version=result.version,
    stage="Staging"
)

# After A/B test passes, promote to Production
client.transition_model_version_stage(
    name="fraud-model-xgboost",
    version=result.version,
    stage="Production"
)
```

**Interview answer for MLflow:**
> "MLflow has 4 components: tracking (logs every run's params, metrics, artifacts), projects (reproducible environment), models (standard serialization with input/output schema), and the registry (versioning and lifecycle). In CI/CD, the pipeline automatically registers the model, moves it to Staging, runs integration tests, and promotes to Production if metrics exceed the champion. Rollback = transition previous version back to Production. Databricks MLflow integrates with Unity Catalog for full data-to-model lineage."

---

## SECTION 4: KUBERNETES FOR ML — WHAT YOU NEED TO KNOW

### Core Concepts

**Pod:** Smallest unit. Contains your FastAPI container + optional sidecar containers.
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: fraud-scorer
    image: myregistry/fraud-api:v2.1
    ports:
    - containerPort: 8080
    resources:
      requests:
        memory: "512Mi"
        cpu: "500m"   # 0.5 CPU cores
      limits:
        memory: "2Gi"
        cpu: "2000m"  # 2 CPU cores
```

**Deployment:** Manages N replicas of a pod, handles rolling updates.
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3   # Always keep 3 pods running
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
```

**HPA (Horizontal Pod Autoscaler) — CRITICAL:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    kind: Deployment
    name: fraud-scorer
  minReplicas: 2      # Never below 2 (prevents cold start)
  maxReplicas: 20     # Never above 20 (cost control)
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale up when >70% CPU
```

**How HPA Works — Step by Step:**
1. **Metrics Server** polls each pod every 15 seconds for CPU/memory
2. **HPA Controller** computes: `desired_replicas = ceil(current * current_metric / target_metric)`
3. Example: 3 pods, 90% CPU, target 70% → desired = ceil(3 * 90/70) = ceil(3.86) = 4 pods
4. **Kubernetes Scheduler** finds node with available capacity, provisions pod
5. **New pod:** pulls Docker image (~30-60 sec), loads model from MLflow (~5-10 sec), passes readiness probe
6. **Load balancer** starts routing traffic to new pod

**Readiness Probe:** K8s only sends traffic to pods that pass this check.
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30   # Wait 30s after start (model load time)
  periodSeconds: 5          # Check every 5 seconds
```

**Liveness Probe:** K8s restarts pods that become unresponsive.
```yaml
livenessProbe:
  httpGet:
    path: /alive
    port: 8080
  periodSeconds: 10
  failureThreshold: 3   # Restart after 3 consecutive failures
```

**Batch Models — CronJob:**
```yaml
apiVersion: batch/v1
kind: CronJob
spec:
  schedule: "0 2 * * *"   # 2am daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: batch-fraud-scorer
            image: myregistry/fraud-batch:v2.1
            command: ["python", "score_batch.py"]
          restartPolicy: OnFailure
```

**Interview answer for Kubernetes:**
> "Real-time fraud models run as FastAPI + Docker in a Kubernetes Deployment. HPA monitors CPU utilization, scaling pods between min=2 (always running for low latency) and max=20 (handles peak claim volume). HPA works by: metrics server checks CPU every 15s, HPA computes desired replicas, scheduler provisions pods, readiness probe ensures pods only receive traffic after model loads. Batch models run as CronJobs at 2am, terminate when done, no persistent resources."

---

## SECTION 5: MONITORING IN PRODUCTION — COMPLETE FRAMEWORK

### Three Layers of Monitoring

**Layer 1 — Data/Input Monitoring**

PSI (Population Stability Index):
```python
def calculate_psi(expected, actual, buckets=10):
    """
    expected: training distribution
    actual: production distribution
    """
    # Create bins from expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets+1))
    
    # Compute bin proportions
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 1e-10, None)
    actual_pct = np.clip(actual_pct, 1e-10, None)
    
    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

Thresholds:
- PSI < 0.10: Stable (no action)
- PSI 0.10-0.20: Warning (investigate)
- PSI > 0.20: Significant shift (investigate and potentially retrain)

KS Test (Kolmogorov-Smirnov):
```python
from scipy.stats import ks_2samp
stat, pvalue = ks_2samp(training_distribution, production_distribution)
# p-value < 0.05 = statistically significant drift
```

Run daily for top 20 features by importance. Alert on > 0.20 PSI.

**Layer 2 — Model/Output Monitoring**

Prediction distribution drift:
```python
# Compare distribution of fraud_probability scores
psi_score = calculate_psi(training_scores, production_scores_last_7_days)
```

Performance metrics (when ground truth is available):
- For fraud: SIU team provides feedback (confirmed fraud/not fraud) with 2-4 week lag
- Compare: predicted fraud flags vs confirmed fraud cases
- Track: Precision, Recall, PR-AUC on confirmed cases

Champ-Challenger testing:
- Always route 10% of traffic to "challenger" model
- Compare performance weekly
- Promote challenger if it outperforms champion for 2 consecutive weeks

**Layer 3 — Infrastructure Monitoring**

Key metrics to track (Prometheus + Grafana):
- **Latency:** p50, p95, p99 response time (NOT just mean — p99 shows tail behavior)
- **Throughput:** requests per second
- **Error rate:** 4xx (bad requests), 5xx (server errors)
- **Pod health:** CPU%, Memory%, number of ready pods
- **Queue depth:** if async, how many requests waiting

Latency SLAs:
- Real-time fraud: p99 < 200ms
- Batch scoring: complete 1M claims before 6am

**Interview answer for monitoring:**
> "We run three monitoring layers: input data drift (PSI daily on top 20 features, alert if > 0.2), output distribution drift (KS test on prediction distribution weekly), and business performance (Precision/Recall on confirmed fraud cases from SIU feedback). Infrastructure metrics via Prometheus/Grafana: p99 latency must stay under 200ms, error rate under 0.1%. When drift is detected, I first investigate root cause — is it a data pipeline issue, a genuine fraud pattern shift, or a feature distribution change? Then decide: retrain, roll back, or update feature engineering."

---

## SECTION 6: BATCH vs REAL-TIME ARCHITECTURE

### Batch Processing Pipeline (Overnight Fraud Scoring)

```
Airflow CronJob (2am)
    ↓
Read from Snowflake: SELECT * FROM claims WHERE created_date = today
    ↓
Load into Databricks Spark DataFrame (partitioned by claim_type)
    ↓
Feature Engineering (PySpark transformations)
    │  → Join with feature store Delta tables
    │  → Compute behavioral aggregates
    │  → WoE encode categorical features
    ↓
Distributed Inference (pandas_udf with XGBoost model)
    │  → Each partition scores independently
    │  → model loaded per worker from MLflow
    ↓
Post-processing (threshold application, segment assignment)
    ↓
Write to Snowflake: fraud_scores table
    ↓
Trigger downstream: notify SIU dashboard, send alerts
```

**PySpark Inference with pandas_udf:**
```python
import mlflow
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

# Load model once per executor
model_uri = "models:/fraud-model-xgboost/Production"
model = mlflow.xgboost.load_model(model_uri)

@pandas_udf(FloatType())
def score_claims(claim_features: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(claim_features)[:, 1])

# Apply distributed scoring
scored_df = claims_df.withColumn("fraud_probability", score_claims(*feature_cols))
```

**Interview answer for batch:**
> "Batch scoring uses Databricks Spark with pandas_udf for distributed inference. The Airflow CronJob triggers at 2am, reads from Snowflake, runs PySpark feature engineering, loads XGBoost model from MLflow registry once per executor, scores all claims in parallel across the cluster, and writes results back to Snowflake by 5am. Partitioning by claim_type avoids data skew. The job takes ~45 minutes for 1M claims."

### Real-Time Pipeline (Claim Arrives Live)

```
Claim Created Event
    ↓
Kafka Topic: 'claim-created' (event streaming)
    ↓
Kafka Consumer (FastAPI background worker)
    ↓
Feature Extraction:
    │  → Static features from JSON payload (claim amount, type)
    │  → Real-time lookups from Redis (claimant history, risk scores)
    │  → Feature validation (null checks, range checks)
    ↓
LightGBM Model Inference (<15ms)
    ↓
Post-processing (threshold: >0.8 → SIU flag, 0.5-0.8 → enhanced review)
    ↓
Response: {fraud_probability: 0.91, fraud_flag: true, top_factors: [...]}
    ↓
Downstream: SIU alert, claims management system update
```

**FastAPI Implementation:**
```python
from fastapi import FastAPI
import mlflow
import json

app = FastAPI()

# Load model at startup (once per pod)
model = mlflow.lightgbm.load_model("models:/fraud-model-lgbm/Production")

@app.post("/score")
async def score_claim(claim: ClaimRequest):
    # Extract and validate features
    features = extract_features(claim)
    
    # Score (< 15ms)
    fraud_prob = model.predict_proba([features])[0][1]
    
    # Apply thresholds
    fraud_flag = fraud_prob > 0.8
    review_flag = 0.5 < fraud_prob <= 0.8
    
    # SHAP explanation for top factors
    top_factors = get_shap_factors(model, features, top_k=3)
    
    return {
        "fraud_probability": float(fraud_prob),
        "fraud_flag": fraud_flag,
        "review_flag": review_flag,
        "top_factors": top_factors
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}
```

**Interview answer for real-time:**
> "Real-time scoring uses FastAPI deployed in Kubernetes. The model is loaded into memory at pod startup from MLflow registry (30-second initialization, handled by readiness probe). Each request: extract features from payload + Redis lookups → LightGBM inference → threshold application → return fraud_probability + top SHAP factors. p99 latency is under 80ms. HPA scales pods based on request rate (target: 60% CPU), min=3 pods to prevent cold starts."

---

## SECTION 7: DATABRICKS + DELTA LAKE

### Why Databricks for ML

**Delta Lake on top of Parquet:**
- ACID transactions: no corrupt reads during parallel writes
- Time travel: `SELECT * FROM delta.claims VERSION AS OF 5` — access any historical version
- Schema enforcement: rejects data that doesn't match expected schema
- Merge (upsert) support: update/insert in one operation (critical for feature stores)

```python
# Write Delta table with versioning
claims_df.write.format("delta").mode("overwrite").save("/mnt/delta/claims")

# Time travel — reproduce training data from any date
training_data = spark.read.format("delta") \
    .option("timestampAsOf", "2025-01-01") \
    .load("/mnt/delta/claims")

# This guarantees reproducibility: same model, same data, same result
```

**MLflow on Databricks:**
- Fully integrated: `mlflow.autolog()` automatically tracks sklearn/XGBoost/LightGBM experiments
- Model registry stored in Databricks-managed storage
- Unity Catalog integration: full lineage from raw data → Delta table → MLflow experiment → model version

**PySpark for Feature Engineering:**
```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Window function for behavioral features
window_spec = Window.partitionBy("claimant_id").orderBy("claim_date") \
    .rowsBetween(-90, 0)  # 90-day rolling window

claims_with_features = claims_df \
    .withColumn("claims_last_90d", F.count("claim_id").over(window_spec)) \
    .withColumn("avg_amount_last_90d", F.mean("claim_amount").over(window_spec)) \
    .withColumn("days_since_last_claim", 
                F.datediff(F.col("claim_date"), 
                           F.lag("claim_date", 1).over(window_spec)))
```

---

## SECTION 8: RETRAINING PIPELINE

### When to Retrain

| Trigger | Description | Action |
|---------|-------------|--------|
| PSI > 0.20 on key features | Input distribution shifted | Investigate → retrain if not data error |
| PR-AUC drops > 3% | Model performance degraded | Immediate retrain + investigate |
| SIU feedback shows false pattern | Fraud pattern evolved | Retrain + review feature set |
| Scheduled cadence | Preventive | Retrain every 4-8 weeks for fraud |
| New fraud pattern identified | Business-driven | Ad-hoc retrain + new features |

### Retraining Pipeline Steps

```
1. Data Validation
   - Check for new fraud patterns in recent claims
   - Validate data completeness (no unexpected nulls)
   - Run PSI comparison: new training data vs original training data

2. Feature Engineering
   - Same pipeline as original training (reproducibility)
   - Check for new features to add, stale features to drop
   - Validate WoE/IV values still meaningful

3. Model Training
   - Train new model on expanded dataset (original + recent)
   - Same hyperparameter search (Optuna)
   - Early stopping on validation PR-AUC

4. Validation (Challenger vs Champion)
   - Compare challenger vs champion on held-out test set
   - Compare on multiple evaluation windows (last 1 month, last 3 months)
   - Check PR-AUC, Precision@Recall, Capture@25, Brier score
   - Challenger must beat champion by > 1% PR-AUC (significance threshold)

5. Register Challenger in MLflow
   - Stage: "Staging"
   - Run integration tests

6. A/B Test in Production
   - Route 10% traffic to challenger
   - Monitor for 2 weeks
   - If metrics hold: promote to Production
   - Archive previous Production version

7. Documentation
   - Update model card (performance metrics, training data window, known limitations)
   - Alert stakeholders of model update
```

**Interview answer:**
> "Retraining is triggered by three things: PSI > 0.2 on key features, PR-AUC degradation > 3% on recent data, or a new fraud scheme identified by SIU. The pipeline: validate new data, retrain challenger model, compare against champion on held-out set, A/B test 10% traffic for 2 weeks, promote if statistically better. We don't retrain more frequently than necessary — for our fraud model, every 4-6 weeks is sufficient."

---

## SECTION 9: AIRFLOW — ORCHESTRATION

### Airflow DAG for Batch Fraud Scoring
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
from datetime import datetime, timedelta

with DAG(
    'fraud_batch_scoring',
    schedule_interval='0 2 * * *',  # 2am daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={'retries': 2, 'retry_delay': timedelta(minutes=5)}
) as dag:
    
    validate_data = PythonOperator(
        task_id='validate_input_data',
        python_callable=validate_daily_claims
    )
    
    score_claims = DatabricksSubmitRunOperator(
        task_id='score_claims_spark',
        databricks_conn_id='databricks_default',
        json={
            'new_cluster': {'spark_version': '13.3.x-scala2.12', 'node_type_id': 'm5.xlarge', 'num_workers': 10},
            'spark_python_task': {'python_file': '/Repos/fraud/batch_score.py'}
        }
    )
    
    check_psi = PythonOperator(
        task_id='check_psi_drift',
        python_callable=run_psi_checks
    )
    
    write_results = PythonOperator(
        task_id='write_to_snowflake',
        python_callable=write_scores_to_snowflake
    )
    
    validate_data >> score_claims >> check_psi >> write_results
```

---

## SECTION 10: QUICK REFERENCE CARD

| Topic | Key Points |
|-------|-----------|
| MLflow stages | None → Staging → Production → Archived |
| MLflow components | Tracking, Projects, Models, Registry |
| PSI thresholds | <0.1 stable, 0.1-0.2 warn, >0.2 retrain |
| HPA mechanism | Metrics Server → HPA Controller → Scheduler → New Pod → Readiness Probe |
| HPA min replicas | Min=2 prevents cold start, Min=3 for critical services |
| CI responsibility | Unit tests, integration tests, SOLID code, SonarQube, packaging |
| CD responsibility | Docker build, K8s deploy, blue-green, canary, rollback |
| Batch tool stack | Airflow + Databricks + Spark + MLflow + Snowflake |
| Real-time stack | Kafka + FastAPI + Kubernetes + Redis + LightGBM |
| Retrain triggers | PSI drift, performance degradation, business feedback, scheduled cadence |
| Delta time travel | SELECT * FROM delta.table VERSION AS OF N |
| Blue-green | Zero downtime: old model + new model in parallel, switch traffic atomically |
| Canary | Gradual rollout: 5% → 20% → 50% → 100% with monitoring at each step |
| Shadow deployment | New model processes all traffic but decisions not used (zero risk validation) |

---

## SECTION 11: A/B TESTING AND EXPERIMENTATION FOR ML

### Why A/B Testing Is Different for ML

Unlike software A/B tests that measure click-through rates immediately, ML A/B tests must wait for ground truth labels. In fraud detection, a claim flagged today may not be confirmed as fraud for 4 to 6 weeks.

### Design of an ML A/B Test

**Randomization unit:**
- Use the entity that receives the treatment. For fraud, randomize by claim or claimant, not by request.
- Avoid randomizing by time or geography because fraud patterns may vary across those dimensions.

**Sample size calculation:**
- Base the calculation on the primary metric and the minimum detectable effect.
- For fraud: if baseline precision is 60% and you want to detect a 5 percentage point improvement, you need enough confirmed cases to reach statistical power.
- Rule of thumb: plan for 2 to 4 weeks of traffic for fraud models to accumulate enough labels.

**Primary and guardrail metrics:**
- Primary: PR-AUC, precision, recall, or Capture@25 on confirmed cases
- Guardrail: false positive rate, customer complaints, investigator workload, latency, error rate

**Stopping rules:**
- Stop early if guardrail metrics degrade significantly
- Do not stop early just because the primary metric looks promising unless you have a pre-specified stopping rule
- Run for the planned duration to avoid peeking bias

### Champion-Challenger in Production

A champion-challenger framework keeps a challenger model running on a fraction of traffic indefinitely.

**Benefits:**
- Detects model degradation before full rollout
- Provides a safe environment to test new features
- Reduces risk of deploying an untested model

**Typical split:**
- Champion: 90% of traffic
- Challenger 1: 5% of traffic
- Challenger 2: 5% of traffic

**Promotion rule:**
- Promote challenger if it outperforms champion by a pre-defined margin for 2 to 4 consecutive weeks
- The margin must exceed normal metric volatility

**Interview one-liner:**
> "For ML A/B tests, randomization is by claim, not by request, because fraud patterns cluster. We wait 4 to 6 weeks for SIU confirmation before computing precision and recall. A challenger model must beat the champion by more than 1 percentage point on PR-AUC for at least two consecutive weeks before promotion."

---

## SECTION 12: DATA DRIFT VS CONCEPT DRIFT — DEEP DIVE

### Data Drift (Covariate Shift)

Data drift means the distribution of input features has changed.

**Examples in fraud:**
- Average claim amount increases due to inflation
- New policy types enter the portfolio
- Seasonal spikes in certain claim categories

**Detection:**
- PSI on top features
- KS test on feature distributions
- Monitoring feature means, medians, and percentiles over time

**Response:**
- Investigate whether the shift is a data pipeline issue or a real business change
- If real and sustained, retrain the model on recent data
- If temporary, wait before retraining

### Concept Drift

Concept drift means the relationship between features and the target has changed.

**Examples in fraud:**
- Fraudsters learn which claim patterns trigger flags and adjust behavior
- New fraud scheme emerges that the model has never seen
- Regulatory change makes previously suspicious behavior normal

**Detection:**
- Performance monitoring on lagged ground truth
- CUSUM charts on prediction error
- KS test on model score distributions
- Manual review of high-confidence misses

**Response:**
- Collect labeled examples of the new pattern
- Engineer features that capture the new behavior
- Retrain and A/B test before rollout

**Why concept drift is harder:**
> "Data drift can often be fixed by retraining on recent data. Concept drift requires understanding the new fraud mechanism and engineering features that capture it. You cannot just retrain blindly and expect improvement."

### Prediction Drift

Prediction drift means the distribution of model outputs has changed even when input features are stable. This can signal upstream data issues or concept drift.

**Interview one-liner:**> "I monitor three types of drift: data drift with PSI on input features, prediction drift with distribution checks on model scores, and concept drift with performance metrics on confirmed labels. Each requires a different response."

---

## SECTION 13: CLOUD PLATFORMS AND MANAGED ML SERVICES

### Google Cloud Platform (GCP)

**Vertex AI:**
- Managed platform for training, tuning, and deploying ML models
- Supports custom containers, pre-trained models, and AutoML
- Integrates with BigQuery for feature storage and model monitoring

**When to use Vertex AI:**
- You want a managed training and serving platform
- You need AutoML for rapid prototyping
- You want integrated model monitoring and explainability

**BigQuery ML:**
- Train models directly inside BigQuery using SQL
- Best for: linear models, logistic regression, k-means, time series
- Limitation: not suitable for complex gradient boosting or deep learning

**Dataproc:**
- Managed Spark and Hadoop service
- Best for: large-scale ETL, feature engineering, distributed training
- You control cluster configuration and can use open-source tools

**When to use Dataproc vs Databricks:**
- Use Dataproc if you are already on GCP and want cost control
- Use Databricks if you want a unified analytics platform with built-in MLflow and Delta Lake

### Amazon Web Services (AWS)

**SageMaker:**
- Managed service for building, training, and deploying models
- Supports notebooks, training jobs, endpoints, and model monitoring

**SageMaker Model Monitor:**
- Monitors data quality, model quality, bias, and feature attribution drift
- Integrates with CloudWatch for alerts

**Lambda / Serverless:**
- Good for: low-volume, event-driven inference
- Bad for: sustained high throughput, low-latency requirements, large models
- Cold start latency can exceed 500ms

**When to use Lambda for ML:**> "Lambda works for occasional batch inference, lightweight pre-processing, or triggering workflows. I avoid Lambda for real-time fraud scoring because cold starts break our 200ms SLA."

### Databricks

**Why Databricks for ML:**
- Unified platform for data engineering, analytics, and ML
- Delta Lake for reliable storage
- MLflow built-in for experiment tracking
- Unity Catalog for data and model governance

**Unity Catalog:**
- Centralized governance for data and AI assets
- Lineage from raw data to features to models
- Access control and audit logging

### Kubeflow

Kubeflow is an open-source platform for orchestrating ML workflows on Kubernetes.

**Components:**
- Kubeflow Pipelines: orchestrate multi-step ML workflows
- Katib: hyperparameter tuning
- KServe: model serving
- Notebooks: collaborative development

**When to use Kubeflow:**
- You want full control over the ML platform
- You already run Kubernetes and have platform engineering support
- You need complex, multi-stage pipelines that go beyond Airflow

**When managed services are better:**
- If you lack platform engineering resources, managed services like SageMaker or Vertex AI reduce operational burden

---

## SECTION 14: SERVERLESS VS KUBERNETES FOR ML SERVING

### Kubernetes Serving

**Strengths:**
- Always-warm pods when minReplicas > 0
- Fine-grained control over resources, scaling, and networking
- Supports complex serving patterns: canary, blue-green, sidecars
- Predictable latency for critical paths

**Weaknesses:**
- Operational overhead
- Need cluster management and capacity planning
- Higher baseline cost because pods run continuously

### Serverless Serving

**Strengths:**
- No cluster management
- Scales to zero when not in use
- Cost-effective for sporadic traffic

**Weaknesses:**
- Cold start latency can be 500ms to several seconds
- Memory and timeout limits
- Less control over environment
- Harder to run sidecars or complex pipelines

### Decision Framework

**Choose Kubernetes when:**
- Latency SLA is under 200ms
- Traffic is sustained or has predictable spikes
- You need canary, blue-green, or shadow deployments
- Model size or dependencies exceed serverless limits

**Choose serverless when:**
- Traffic is infrequent and latency is not critical
- You want to avoid operational overhead
- Cost per request is lower than keeping pods warm

**Interview one-liner:**> "I choose Kubernetes for real-time fraud scoring because we need consistent sub-200ms latency and cannot tolerate cold starts. I would consider serverless only for batch preprocessing or low-frequency internal tools."

---

## SECTION 15: FEATURE STORE DEEPER DIVE

### Why Feature Stores Matter

A feature store eliminates training-serving skew by ensuring the same feature computation logic is used in both training and inference.

### Online Feature Store

**Purpose:** Serve pre-computed features to real-time models with low latency.

**Characteristics:**
- Low-latency reads, typically under 5ms
- Stores only features needed for real-time inference
- Refreshed by batch jobs at regular intervals
- Uses TTL to handle staleness

**Redis best practices:**
- Use pipelining for bulk writes
- Partition keys to avoid hot spots
- Monitor memory usage and eviction policies
- Use Redis Cluster for horizontal scaling

### Offline Feature Store

**Purpose:** Store historical feature values for training and batch scoring.

**Characteristics:**
- Stores point-in-time correct features
- Supports time travel to reconstruct training datasets
- Uses Delta Lake or similar versioning

**Point-in-time correctness:**
> "For training, features must be as they existed at the time of prediction, not as they are today. Using current feature values creates leakage. The offline feature store records feature values by timestamp so we can reconstruct any historical view."

### Training-Serving Skew Prevention

**Sources of skew:**
- Different code paths for training and serving
- Different libraries or versions
- Different handling of missing values
- Different aggregation windows

**Controls:**
- Shared feature computation code in a library used by both pipelines
- Schema validation at inference
- Integration test comparing offline and online predictions
- Feature store as the single source of truth

**Interview one-liner:**> "I prevent training-serving skew by putting feature computation in a shared library, validating schemas at inference, and running integration tests that compare offline training predictions against online serving predictions on the same data."

---

## SECTION 16: SECURITY, PII, AND COMPLIANCE IN ML PIPELINES

### Handling PII and PHI

**Anonymization and pseudonymization:**
- Remove or mask direct identifiers before model training
- Replace identifiers with tokens or hashes
- Use k-anonymity or differential privacy where required

**In the RAG/NLP pipeline:**
- Remove names, addresses, phone numbers, medical record numbers, and policy numbers before sending text to LLM APIs
- Use NER-based masking tools like Microsoft Presidio
- Store only masked or tokenized versions in vector databases

### Audit and Lineage

**Model lineage:**
- Track training data version, feature version, code commit, hyperparameters, and model artifact
- MLflow and Databricks Unity Catalog provide this

**Audit trail:**
- Log every prediction, model version, and feature values used
- Required for regulated industries like insurance and healthcare
- Retain logs according to compliance requirements

### Access Control

- Restrict access to production models and training data
- Use role-based access control (RBAC)
- Separate development, staging, and production environments
- Encrypt data at rest and in transit

**Interview one-liner:**> "For insurance fraud, I anonymize claim notes before any LLM processing, maintain full audit logs of predictions and model versions, and restrict production model access through role-based controls. No raw PII is stored in model artifacts or vector indexes."

---

## SECTION 17: MODEL ROLLBACK AND INCIDENT RESPONSE

### Rollback Mechanics

**MLflow-based rollback:**
- Transition the previous model version back to the Production stage
- Update the model URI in the serving configuration
- Kubernetes pods restart and load the previous model

**Time to rollback:**
- With automated pipelines: 2 to 5 minutes
- Manual rollback should also be documented and tested quarterly

### Incident Response Playbook

**Detection:**
- Automated alerts on error rate, latency, or prediction distribution
- Manual reports from business users or SIU team

**Triage:**
- Determine if the issue is infrastructure, data pipeline, or model behavior
- Check recent deployments, data pipeline changes, and external dependencies

**Mitigation:**
- If model is the cause: rollback to previous version
- If data pipeline is the cause: halt retraining until fixed
- If external dependency is down: activate fallback rules

**Post-incident:**
- Root cause analysis
- Update runbooks and monitoring
- Add tests to prevent recurrence

**Interview one-liner:**> "My rollback procedure is: revert the MLflow Production stage to the previous version, update the serving configmap, and restart pods. Total time is under 5 minutes. I also maintain a fallback rule-based system so fraud detection continues even if the model is unavailable."

---

## SECTION 18: MLOPS INTERVIEW SCENARIOS

### "How do you deploy an ML model to production?"

> "I package the model and feature pipeline as a Python wheel and a Docker image. The serving layer is a FastAPI endpoint that loads the model from MLflow registry. I deploy to Kubernetes with a Deployment, HPA, readiness probe, and liveness probe. For rollout I use canary or blue-green deployment, monitor guardrail metrics, and have an automatic rollback path."

### "How do you monitor models in production?"

> "I use three layers. Input monitoring with PSI and KS tests on top features. Output monitoring with prediction distribution drift and latency/error metrics. Performance monitoring with precision, recall, and PR-AUC on lagged ground truth. Alerts go to Slack or PagerDuty based on severity."

### "What is the difference between data drift and concept drift?"

> "Data drift is a change in the input feature distribution. Concept drift is a change in the relationship between inputs and the target. Data drift can be detected with PSI on features. Concept drift can only be detected with labeled outcomes, such as SIU feedback. They require different responses."

### "How do you version training data?"

> "I store training data in Delta Lake and record the table version or timestamp in the MLflow run metadata. To reproduce a model, I check out the model by run_id and read the training data at the recorded version. This gives full reproducibility."

### "What do you do when the model fails in production?"

> "First, I check whether the issue is infrastructure, data, or model behavior. If it is the model, I roll back to the previous MLflow version. If it is a data pipeline issue, I halt retraining and fix the pipeline. In all cases, a fallback rule-based system keeps fraud detection running. After the incident, I run a root cause analysis and add preventive tests."

---

## SECTION 19: ML CI/CD PIPELINE — STAGE BY STAGE

### How ML CI/CD Differs from Software CI/CD

Software CI/CD validates code correctness. ML CI/CD must also validate data, features, model performance, and reproducibility.

### CI/CD Stages for ML

| Stage | Purpose | Checks |
|---|---|---|
| Code commit | Trigger pipeline | Lint, type checks, unit tests |
| Data validation | Ensure training data is healthy | Schema checks, null rate, distribution checks, freshness |
| Feature validation | Ensure feature logic is consistent | Training-serving parity test, IV/VIF checks |
| Model training | Train challenger model | Reproducibility, hyperparameter tracking |
| Model evaluation | Validate model quality | PR-AUC, precision, recall, calibration, bias checks |
| Integration test | Validate end-to-end pipeline | Offline vs online prediction comparison |
| Staging deploy | Deploy to staging environment | Smoke tests, load tests |
| Production gate | Human or automated approval | Performance must exceed champion by defined margin |
| Production deploy | Canary/blue-green rollout | Monitor guardrail metrics |
| Continuous monitoring | Detect drift and degradation | PSI, KS, business metrics, latency/error SLOs |

### Model Evaluation Gates

Before a model moves to production, it must pass:
- **Performance gate:** challenger beats champion by pre-defined margin on primary metric
- **Guardrail gate:** no degradation on fairness, latency, calibration, or error rate
- **Business gate:** meets operational constraints like false positive budget
- **Reproducibility gate:** training can be reproduced from recorded data and code versions

### Automated vs Manual Promotion

**Automated promotion is appropriate when:**
- The change is a routine retrain with no architecture changes
- A/B test results pass predefined statistical criteria
- Guardrail metrics are all green

**Manual approval is appropriate when:**
- New feature types are introduced
- Model architecture changes
- High-stakes deployment with regulatory exposure

### Interview One-Liner

> "ML CI/CD has more gates than software CI/CD because we must validate data, features, model performance, and reproducibility, not just code. Every promotion requires passing performance, guardrail, business, and reproducibility gates before production rollout."

---

## SECTION 20: EXPERIMENT TRACKING VS MODEL REGISTRY

### Experiment Tracking

Experiment tracking records the research process: hyperparameters, metrics, code version, artifacts, and notes from many training runs.

**Tools:** MLflow Tracking, Weights & Biases, Neptune, TensorBoard

### Model Registry

A model registry is a lifecycle management system for production model versions. It tracks stages (Staging, Production, Archived) and metadata required for deployment.

**Key differences:**

| Aspect | Experiment Tracking | Model Registry |
|---|---|---|
| Purpose | Research and comparison | Production lifecycle management |
| Users | Data scientists | Data scientists, ML engineers, platform teams |
| Content | All experiments, including failures | Only validated, deployable models |
| Stages | None — free-form tags | Defined stages: Staging, Production, Archived |
| Governance | Light | Strict — production versions are immutable |

### What the Model Registry Stores

- Model artifact location
- Training data version / timestamp
- Feature version
- Code commit hash
- Hyperparameters and metrics
- Model card and known limitations
- Approval history

### Interview One-Liner

> "Experiment tracking is for research and comparison across many trials. The model registry is the production system of record that governs which model version is staged, in production, or archived. A model moves from experiment tracking to the registry only after passing evaluation gates."

---

## SECTION 21: DATA VERSIONING AND DVC

### Why Data Versioning Matters

Code versioning is not enough. A model is a function of code + data + hyperparameters. To reproduce a model, you must version all three.

### DVC (Data Version Control)

DVC versions large data files and models by storing metadata in Git while keeping actual data in remote storage (S3, GCS, Azure Blob).

**What DVC tracks:**
- Raw datasets
- Processed features
- Trained model artifacts
- Intermediate outputs

**DVC vs Delta Lake:**

| Use Case | Tool |
|---|---|
| Version raw files, models, and artifacts | DVC |
| Version structured tables with time travel | Delta Lake / Iceberg |
| Experiment tracking and model lifecycle | MLflow |

**Best practice:**
Use DVC for large artifacts and model binaries. Use Delta Lake for structured feature tables. Use MLflow to link code, data version, and model artifact together in a registry entry.

### Interview One-Liner

> "I version code with Git, structured data with Delta Lake, and large artifacts with DVC. MLflow ties them together by recording the data version, code commit, and model artifact for every training run, making reproduction possible."

---

## SECTION 22: SCHEMA EVOLUTION AND DATA CONTRACTS

### Why Schema Evolution Matters

Upstream data producers change schemas. If a feature pipeline silently breaks, the model may receive wrong inputs.

### Data Contracts

A data contract defines the expected schema, semantics, and quality rules for data exchanged between systems.

**Elements of a data contract:**
- Column names and types
- Nullability rules
- Allowed value ranges
- Freshness requirements
- Ownership and contact information

### Handling Schema Changes

**Additive changes (new column):**
- Usually safe; schema validation can warn and continue

**Breaking changes (column removed, type changed):**
- Pipeline should fail fast and alert
- Model should fall back to a safe mode

### Schema Validation at Inference

- Validate incoming feature payload against expected schema
- Reject or impute based on a feature availability map
- Log schema violations for investigation

### Interview One-Liner

> "I treat data schemas as contracts. The feature pipeline validates every input against the contract. Additive changes trigger warnings; breaking changes fail fast. At inference, schema validation ensures the model only receives expected features."

---

## SECTION 23: SLOs, SLIs, AND ERROR BUDGETS FOR ML SERVICES

### Definitions

- **SLI (Service Level Indicator):** A measurable metric, e.g., p99 latency
- **SLO (Service Level Objective):** A target for the SLI, e.g., p99 latency < 200ms
- **SLA (Service Level Agreement):** A contract with consequences, often customer-facing
- **Error budget:** The allowable amount of unreliability before you must pause changes

### Example SLOs for Fraud Scoring

| SLI | SLO |
|---|---|
| p99 latency | < 200ms |
| Error rate | < 0.1% |
| Availability | 99.9% |
| Prediction drift PSI | < 0.1 |
| Feature freshness | < 5 minutes |

### Error Budget Policy

If the error budget is consumed, the team must:
- Pause non-critical deployments
- Focus on reliability work
- Resolve root causes before resuming feature work

### Interview One-Liner

> "I define SLOs for latency, error rate, availability, and prediction drift. An error budget tells us how much unreliability we can tolerate in a window. If we exhaust the budget, we stop feature work and focus on reliability."

---

## SECTION 24: SCHEDULED VS EVENT-DRIVEN RETRAINING

### Scheduled Retraining

Retrain on a fixed cadence: daily, weekly, monthly.

**Pros:**
- Predictable cost and compute schedule
- Easy to plan around business cycles
- Simpler to monitor

**Cons:**
- May retrain when nothing has changed
- Slow response to sudden drift

### Event-Driven Retraining

Retrain when a specific event occurs: PSI exceeds threshold, performance drops, new fraud pattern identified.

**Pros:**
- Faster response to real changes
- Avoids wasted compute

**Cons:**
- Harder to schedule and cost-control
- Can trigger too often if thresholds are too tight

### Hybrid Approach

Most production systems use both:
- A scheduled lightweight retrain (e.g., weekly) for incremental updates
- Event-driven triggers for significant drift or new patterns

### Interview One-Liner

> "I use a hybrid approach: scheduled retraining for incremental model refresh and event-driven retraining when drift or performance degradation crosses thresholds. This balances stability with responsiveness."

---

## SECTION 25: MODEL RETIREMENT

### Why Retire Models

Models accumulate technical debt:
- Dependencies become unsupported
- Training data becomes obsolete
- Performance degrades beyond acceptable levels
- Business use case changes

### Retirement Process

1. **Identify candidate:** model has not been used in production for N months or fails SLOs
2. **Notify stakeholders:** announce retirement date and migration path
3. **Archive artifacts:** store model, data version, and documentation
4. **Remove from serving:** stop deployment and monitoring
5. **Retain audit logs:** keep prediction history per compliance requirements
6. **Delete or decommission:** free storage and compute after retention period

### Interview One-Liner

> "Model retirement is part of the ML lifecycle. I archive the model, data, and documentation, notify stakeholders, remove it from serving, and retain audit logs for compliance before decommissioning."

---

## SECTION 26: ADVANCED DRIFT METRICS AND TOOLS

### Beyond PSI and KS

| Metric | Use Case |
|---|---|
| Jensen-Shannon divergence | Symmetric measure of distribution similarity, bounded between 0 and 1 |
| KL divergence | Measures information loss when using training distribution to approximate production distribution |
| Wasserstein distance | Measures how much distribution mass must move to align two distributions |
| Population Stability Index (PSI) | Industry standard, easy to interpret |
| KS test | Statistical test for distribution equality |

**When to use Jensen-Shannon:**
- When you want a bounded, symmetric metric
- When comparing distributions with zero bins where KL would diverge

### Monitoring Tools

| Tool | Strengths |
|---|---|
| Evidently AI | Open-source, comprehensive reports for data drift, model quality, target drift |
| NannyML | Specialized for estimating model performance when labels are delayed |
| Whylabs | Lightweight, real-time data quality monitoring |
| MLflow Model Monitoring | Integrated with Databricks ecosystem |
| SageMaker Model Monitor | Managed, integrates with AWS |

### Interview One-Liner

> "I use PSI and KS for operational drift alerts. For deeper analysis, I use Jensen-Shannon divergence because it is bounded and symmetric. Tools like Evidently and NannyML help generate drift reports and estimate performance when labels are delayed."

---

## SECTION 27: MODEL SERIALIZATION RISKS

### Why pickle Is Risky

Pickle can execute arbitrary code during deserialization. Loading an untrusted pickle file is a security vulnerability.

**Safer alternatives:**
- ONNX for model artifacts
- Joblib for scikit-learn models in trusted environments
- Native model formats: XGBoost JSON/binary, LightGBM text/binary
- MLflow model flavors with validated signatures

### Best Practice

In production, prefer framework-native formats or ONNX. Avoid pickle for models received from external sources or stored in shared locations.

### Interview One-Liner

> "I avoid pickle in production because deserialization can execute arbitrary code. I use ONNX, framework-native formats, or MLflow model flavors with validated signatures."

---

## SECTION 28: MULTI-TENANT ML PLATFORM OPS

### Tenant Isolation in Production

| Isolation Level | Cost | Customization | Operational Complexity |
|---|---|---|---|
| Shared model | Low | Low | Low |
| Tenant-tuned heads | Medium | Medium | Medium |
| Fully isolated models | High | High | High |

### Operational Concerns

- **Resource quotas:** prevent one tenant from consuming all serving capacity
- **Cost attribution:** track compute, storage, and API costs per tenant
- **Data isolation:** ensure no cross-tenant data leakage in training or serving
- **Per-tenant monitoring:** track latency, error rate, and drift per tenant
- **Per-tenant rollback:** ability to roll back one tenant without affecting others

### Interview One-Liner

> "For a multi-tenant ML platform, I define tenant isolation levels, enforce resource quotas, attribute costs per tenant, monitor per-tenant metrics, and ensure data isolation in both training and serving."

---

## SECTION 29: ADDITIONAL MLOPS INTERVIEW SCENARIOS

### "What is the difference between experiment tracking and a model registry?"

> "Experiment tracking captures all my research trials, including failures, with hyperparameters and metrics. The model registry is the production system of record that manages lifecycle stages like Staging, Production, and Archived. Only validated models move from experiment tracking to the registry."

### "How do you decide when to retrain a model?"

> "I use a hybrid trigger model. Scheduled retraining gives incremental refresh. Event-driven retraining fires when PSI exceeds 0.2, PR-AUC drops more than 3%, or SIU identifies a new fraud pattern. This avoids wasted compute while responding to real drift."

### "How do you set SLOs for an ML service?"

> "I define SLIs for latency, error rate, availability, feature freshness, and prediction drift, then set SLOs based on business requirements. I also define an error budget so the team knows when to pause feature work and focus on reliability."

### "What do you do when schema changes in upstream data?"

> "I validate every incoming dataset against a data contract. Additive changes trigger warnings. Breaking changes fail the pipeline and alert the owner. At inference, schema validation ensures the model receives only expected features, with a fallback for missing fields."

### "How do you retire a model?"

> "I identify models that are unused or below SLO, notify stakeholders, archive the model and data artifacts, remove it from serving and monitoring, and retain audit logs for compliance before decommissioning storage."

---

## SECTION 30: LLM / GENAAI PLATFORM MLOPS — PRODUCTION PATTERNS (FROM AXTRIA PROD.TXT)

> Standard MLOps covers traditional ML models. This section covers the additional MLOps layer required for production LLM systems — the patterns I built at Axtria for an enterprise multi-tenant GenAI platform.

---

### The Key Differences: Traditional ML MLOps vs LLM MLOps

| Concern | Traditional ML MLOps | LLM / GenAI MLOps |
|---|---|---|
| Artifact to version | Model weights (.pkl, .whl) | Model weights + Prompt templates + Tool definitions |
| Training trigger | Data drift / label availability | Rarely retrained; prompt changes are the primary "update" |
| Evaluation | Offline AUC, RMSE on labeled test set | LLM-as-judge, human annotation, behavioral assertions |
| Monitoring | PSI, feature drift, prediction drift | Token usage, cost, latency, quality scores, hallucination rate |
| Serving | REST API with deterministic output | WebSocket streaming with non-deterministic output |
| Failures | Model returns wrong score | Agent loops, hallucination, context overflow, cost blowout |
| Observability | Prometheus/Grafana metrics | LLM trace platforms (Langfuse, LangSmith) |

---

### Prompt Versioning — Treating Prompts Like Code

**Why prompts must be versioned:**
- A single word change in a system prompt can drastically change agent behavior
- Without versioning, you cannot reproduce a specific agent run or debug a regression
- Prompt changes are the most frequent "deployment" in a production LLM system

**How to version prompts:**
- Store prompt templates in Git with semantic versioning (v1.2.0)
- Every agent execution logs the prompt version alongside the trace ID
- Never hardcode prompts inline in Python — load from a versioned template store
- On every prompt change: run the full evaluation suite on the fixed test set before merging

**Prompt Registry Pattern:**
```
prompt_registry/
  ├── fraud_agent_system_prompt/
  │     ├── v1.0.0.txt    ← original
  │     ├── v1.1.0.txt    ← added citation requirement
  │     └── v2.0.0.txt    ← breaking change, new output schema
  └── sql_agent_system_prompt/
        └── v1.0.0.txt
```

**Interview One-Liner:**
> "I treat prompts as code: versioned in Git, loaded from a template registry, evaluated on a fixed test set before every merge. When an agent behavior changes, the first question is: which prompt version was running?"

---

### LLM Evaluation CI/CD — Quality Gates Before Deployment

**The Problem:** Deploying a new prompt or swapping to a new model version (e.g., GPT-4o → GPT-4o-mini) can degrade quality silently. Without automated evaluation, you discover the regression when users complain.

**The Solution — Evaluation as a CI/CD Gate:**

```
Prompt Change Commit
    → CI Trigger
    → Load Fixed Evaluation Test Set (e.g., 200 representative queries)
    → Run all queries through new prompt version
    → Score each output with LLM-as-Judge (Langfuse automated scoring):
        - Completeness ≥ 0.85
        - Helpfulness ≥ 0.80
        - Trajectory ≤ 5 steps (agent efficiency)
        - Faithfulness ≥ 0.90 (no hallucination)
    → Compare against baseline scores
    → If any dimension regresses by > 5%: BLOCK deployment, alert team
    → If all pass: promote to staging, then production
```

**Why LLM-as-Judge over human evaluation at every CI run:**
- Human evaluation is accurate but slow (hours per run) and expensive
- LLM-as-Judge runs in seconds, scales infinitely, and is consistent
- Human annotation is reserved for periodic deep reviews and catching systematic judge errors

---

### Token Budget Enforcement — Cost as a First-Class SLO

**Why token budgets matter:**
- Unconstrained agents can loop and accumulate thousands of dollars in API costs (the "$47K LangChain incident")
- Token costs scale linearly with usage — a 10x traffic spike means 10x cost without budgets

**Budget enforcement layers at Axtria:**

| Layer | Control | Enforcement Point |
|---|---|---|
| Per-task token cap | Max tokens per single agent execution | LangGraph state check at each node |
| Per-session cost cap | Max $ spend per user session | Redis counter incremented per LLM call |
| Daily cost alert | Alert when daily spend > threshold | Langfuse cost tracking + alerting |
| Model routing | Cheap model for extraction, expensive for reasoning | LangGraph node-level model config |

**Model Routing Strategy (reduce cost without sacrificing quality):**

| Agent Step | Model Choice | Reason |
|---|---|---|
| Intent classification | GPT-4o-mini / small model | Binary classification, cheap |
| Simple extraction | GPT-4o-mini | Structured output, no deep reasoning |
| SQL generation | GPT-4o | Accuracy critical, schema-aware reasoning |
| Complex multi-step reasoning | GPT-4o / Claude | Full capability needed |
| Reflection / Judge | GPT-4o | Accuracy of critique is critical |

**Interview One-Liner:**
> "I treat token cost as a first-class SLO. I enforce per-task token caps in the state machine, per-session cost caps tracked in Redis, and route simple steps to cheap models and complex reasoning to expensive models. Without this, a looping agent can generate thousands of dollars of API spend overnight."

---

### Secrets Management — Vault over Environment Variables

**The problem with environment variables:**
- Environment variables are dumped in crash reports, visible in Kubernetes pod descriptions, and often logged accidentally
- Rotating a secret requires a pod restart (downtime) or a complex rolling update
- No audit trail of who read the secret or when

**HashiCorp Vault at Axtria:**
- All API keys (OpenAI, Anthropic), DB passwords, and integration credentials stored in Vault
- Vault injects secrets at pod startup via sidecar or Kubernetes Secrets Operator
- Secrets are short-lived and auto-rotate — compromised keys expire within hours
- Full audit log: every secret read is logged with the requesting service identity
- Secret rotation does NOT require pod restart — Vault pushes the new value dynamically

**Interview One-Liner:**
> "I use HashiCorp Vault for all secrets. Keys are injected at runtime, auto-rotate, and every read is audited. This means a compromised API key expires within hours and I have a full record of which service accessed it and when."

---

### WebSocket Serving — MLOps Considerations

**Traditional ML serving:** stateless REST, request → response, easy to load balance and monitor.

**LLM WebSocket serving:** stateful streaming connection — different operational requirements:

**Challenges and solutions at Axtria:**

| Challenge | Solution |
|---|---|
| Connection persistence across pod restarts | Load balancer with sticky sessions (same pod for session duration) |
| Monitoring streaming latency | Track time-to-first-token (TTFT) and token-per-second rate via Langfuse |
| Dead connections from slow generation | Async keepalive ping every 15 seconds |
| Backpressure when client is slow | Async queue on server side; don't block LLM generation |
| Session memory across pod failures | Redis-backed memory; any pod can resume any session |

**Key Metrics for WebSocket LLM Serving:**
- **TTFT (Time to First Token):** Target < 500ms. This is user-perceived latency.
- **TPS (Tokens per Second):** Average throughput of token delivery
- **Session drop rate:** % of sessions that disconnect before completion
- **Memory hit rate:** % of sessions that successfully load prior Redis context

---

### Multi-Tenant LLM Platform — Operational Runbook

**Scenario 1: Tenant reports their RAG is returning wrong documents**
1. Pull Langfuse traces for the affected user's session
2. Check tenant_id filter on the vector retrieval call — is it correctly scoped?
3. Check chunking configuration for that tenant's documents — are chunks too large/small?
4. Check embedding model version — if it changed, re-embed the tenant's document corpus
5. Check RRF fusion weights — if BM25 is dominating, semantic matches may be suppressed

**Scenario 2: Agent costs spike 10x overnight**
1. Pull Langfuse cost dashboard — identify which agent type is responsible
2. Check for looping agents: look for sessions with > N LLM calls
3. Check loop detection guards — did they fire? If not, why?
4. Check if a new prompt version was deployed — did it introduce an ambiguous instruction causing re-tries?
5. Tighten per-task token cap and per-session cost cap; roll back suspect prompt version

**Scenario 3: Quality scores drop after model version update**
1. Compare Langfuse quality scores for 7 days pre vs post model version change
2. Identify which dimension regressed (completeness? faithfulness? trajectory?)
3. Run fixed evaluation test set on old and new model versions to isolate
4. If regressed: pin to previous model version; refine prompt for new version compatibility
5. Never treat a model upgrade as a no-op — always gate on evaluation scores

**Scenario 4: New tenant onboarding**
1. Insert tenant record into tenants table (triggers RLS policy automatically)
2. Create tenant's document storage partition in vector store
3. Run document ingestion pipeline for tenant's initial corpus
4. Provision JWT credentials and test all 6 AI surfaces in staging with tenant context
5. Enable in production; monitor first 48 hours of Langfuse traces for anomalies

---

### Interview Q&A: LLM MLOps

**Q: "How do you monitor a production LLM system?"**
> "I layer two types of monitoring. Infrastructure monitoring via Prometheus/Grafana: API latency, error rates, WebSocket connection counts, Redis memory usage. LLM-specific monitoring via Langfuse: token usage and cost per task, quality scores (completeness, helpfulness, trajectory, faithfulness), hallucination rate from the reflection layer, and human escalation rate. When quality scores drop, I compare the current prompt version and model version against the baseline to isolate the cause."

**Q: "What is a prompt regression and how do you prevent it?"**
> "A prompt regression is when a change to a prompt template causes the agent to perform worse on previously-working tasks — typically silent, discovered only when users complain. I prevent it by treating prompts as code with Git versioning, running every prompt change through an automated evaluation suite on a fixed test set, scoring with LLM-as-judge across completeness, helpfulness, and faithfulness dimensions, and blocking deployment if any dimension regresses by more than 5%."

**Q: "How do you control LLM API costs at scale?"**
> "Three layers. First, model routing: cheap models (GPT-4o-mini) for classification and extraction, expensive models only for complex reasoning. Second, per-task token caps enforced in the state machine — the agent physically cannot exceed the budget regardless of looping. Third, per-session and daily cost caps tracked in Redis, with Langfuse alerting when thresholds are exceeded. The goal is treating cost as an SLO, not an afterthought."

**Q: "How do you ensure data isolation in a multi-tenant LLM platform?"**
> "Four enforcement points. Database layer: PostgreSQL Row-Level Security — physically impossible for one tenant's query to return another's data. Vector retrieval layer: every query carries a mandatory tenant_id filter at the retrieval step, not in application code. LLM context: tenant data is never co-mingled in a single prompt. Secrets: Vault-managed credentials with per-tenant API key scoping where supported. Defense in depth — no single layer can be the sole isolation guarantee."
