# Day 201: Titan Phase 4: The Observability Mesh
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 29: Capstone Project Part 1

---

> **🎯 Focus Area:** We have 1000 requests per second hitting the Scoring Service. Is the model accurate? Is latency degrading? Is the User ID propagating? **Titan Observability** links Signals across Boundaries.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Instrument** the KServe Predictor to emit custom Prometheus Metrics (`accuracy_score`, `model_drift_value`).
2.  **Configure** OpenTelemetry Collector to gather Traces from the Istio Mesh (Sidecars) and Application Code.
3.  **Construct** a Grafana A/B Testing Dashboard comparing v1 (Control) vs v2 (Canary).
4.  **Implement** an AlertManager rule for "Response Distribution Drift" (e.g., Model starts returning only 0s).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-inf`.

### Software Environment
- `helm install prometheus-stack`.
- `helm install tempo`.

---

## 📖 Theoretical Foundation

### 1. The 4 Signals of ML
Customizing the Google SRE "Golden Signals" for ML:
*   **Latency:** Time to Inference.
*   **Traffic:** Requests per second.
*   **Errors:** HTTP 500s.
*   **Saturation:** GPU Memory Usage.
*   **Accuracy (New):** Feedback Loop score.
*   **Drift (New):** How different is Live Data from Training Data?

### 2. Context Propagation in Mesh
Istio automatically injects `x-b3-traceid` headers.
However, Python code must **Forward** these headers when making downstream calls (e.g., to Feature Store).
If you drop the header, the Trace breaks.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Enable Tracing in Istio

Global Mesh Config.

#### 📁 `manifests/istio-tracing.yaml`
```yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-tracing
spec:
  meshConfig:
    enableTracing: true
    defaultConfig:
      tracing:
        sampling: 10.0 # Sample 10%
        zipkin:
          address: "zipkin.istio-system:9411" # Send to Tempo (Zipkin compatible)
```

### 👨‍💻 Core Implementation: Custom ML Metrics (Python)

Inside the KServe Predictor code.

#### 📁 `src/predictor.py`
```python
from kserve import Model
from prometheus_client import Summary, Histogram, Counter

# Define Metrics
PREDICTION_LATENCY = Histogram("model_latency_seconds", "Time spent predicting")
PREDICTION_VALUE = Histogram("model_prediction_value", "Distribution of output scores")
DRIFT_WARNING = Counter("model_drift_detected", "Number of drifty inputs")

class MyModel(Model):
    def __init__(self, name):
        super().__init__(name)

    async def predict(self, payload):
        with PREDICTION_LATENCY.time():
            inputs = payload["inputs"]
            
            # Check Drift (Simple Statistical check)
            if inputs.mean() > 5.0: 
                DRIFT_WARNING.inc()
                
            # Predict
            result = self.model.predict(inputs)
            
            # Record Distribution (for A/B testing)
            PREDICTION_VALUE.observe(result[0])
            
            return {"predictions": result.tolist()}
```

### 👨‍💻 Infrastructure: ServiceMonitor

Tell Prometheus to scrape the KServe Pods.

#### 📁 `manifests/servicemonitor.yaml`
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: kserve-scraper
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      serving.kserve.io/inferenceservice: sklearn-iris
  endpoints:
  - port: user-port # KServe exposes user metrics here
    path: /metrics
    interval: 5s
```

### 👨‍💻 Core Implementation: The A/B Dashboard (PromQL)

Grafana Panel Queries.

**Panel 1: Latency Comparison (Line Chart)**
```promql
rate(model_latency_seconds_sum{model="v1"}[5m]) / rate(model_latency_seconds_count{model="v1"}[5m])
vs
rate(model_latency_seconds_sum{model="v2"}[5m]) / rate(model_latency_seconds_count{model="v2"}[5m])
```

**Panel 2: Prediction Distribution (Heatmap)**
```promql
sum(rate(model_prediction_value_bucket[5m])) by (le, model)
```
*Visual Check: If v1 predicts mostly 0.8 and v2 predicts mostly 0.2, something is wrong.*

---

## 🔬 Lab Exercise: "The Broken Chain"

### Task
Debug a missing trace.
1.  **Scenario:** `Gateway -> Model -> FeatureStore`.
2.  **Observation:** Trace shows `Gateway -> Model`. Trace ID changes for `Model -> FeatureStore` (Disconnected).
3.  **Cause:** The Model code did not copy headers.
4.  **Fix:**
    ```python
    # Before
    requests.get("http://feature-store/get")
    
    # After (Propagate Headers)
    headers = {"x-request-id": request.headers.get("x-request-id")}
    requests.get("http://feature-store/get", headers=headers)
    ```
5.  **Verify:** Trace is now contiguous.

---

## 📖 Advanced Theory: Outlier Detection
How do you detect "Weird" data without labels?
**Alibi Detect (VAE):**
*   Train an Auto-Encoder on Training Data.
*   Inference: Run input through VAE. Calculate Reconstruction Error.
*   If Error > Threshold, input is "Out of Distribution" (Drift).
*   Flag metric `model_drift_detected` for Alerting.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Metrics are Cheap, Logs are Expensive:** For "Drift Usage", increment a Counter metric. Don't log "Drift detected" 1000 times a second.
2.  **Percentiles:** Average Latency lies. Always monitor p99 Latency. If p99 is 5s but Average is 100ms, 1% of your users (10k people) are angry.
3.  **Granularity:** KServe exposes metrics per *Revision*. This is key for Canary Analysis. You can separate `my-model-v1` from `my-model-v2`.

### API Summary
```python
HISTOGRAM.observe(3.5)
```

---

**Day 201 Complete** ✅

*Next: Day 202 - Capstone Part 6 - Security & RBAC.*
