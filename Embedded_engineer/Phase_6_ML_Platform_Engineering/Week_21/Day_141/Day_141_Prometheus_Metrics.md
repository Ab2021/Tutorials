# Day 141: Pulse of the Machine: Prometheus & Metrics
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** Is your model slow because of the Network, the Disk IO, or the GPU Compute? **Prometheus** scrapes metrics from everything to answer this question.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Logs (Text), Metrics (Numbers), and Traces (Time Spans).
2.  **Instrument** a Python Inference Service with `prometheus_client` to expose custom metrics.
3.  **Deploy** Prometheus on Kubernetes using the Prometheus Operator.
4.  **Write** PromQL queries to calculate Rate, Error % and 99th Percentile Latency.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- `pip install prometheus-client`.

---

## 📖 Theoretical Foundation

### 1. The Pull Model
*   **Push (StatsD, Datadog):** App sends UDP packet to Server. (Fire and forget).
*   **Pull (Prometheus):** App exposes HTTP `/metrics`. Prometheus server periodically requests this URL ("Scraping").
    *   *Pros:* App doesn't need to know where server is. Server controls load.

### 2. Metric Types
*   **Counter:** Only goes up (e.g., `requests_total`). Rate = $\Delta C / \Delta t$.
*   **Gauge:** Goes up and down (e.g., `gpu_memory_usage`).
*   **Histogram:** Buckets values (e.g., `request_latency_seconds`). Used to calculate P99.
*   **Summary:** Pre-calculated quantiles (More expensive client-side).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Installing Prometheus (Helm)

Don't install raw Prometheus. Use the **Kube-Prometheus-Stack** (includes Grafana, Node Exporter, AlertManager).

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

kubectl create ns monitoring
helm install prometheus-stack prometheus-community/kube-prometheus-stack -n monitoring
```

### 👨‍💻 Core Implementation: Instrumenting Python Code

Let's instrument a FastAPI Inference Service.

#### 📁 `src/main_metrics.py`
```python
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, Counter, Histogram, Gauge
import time
import random

app = FastAPI()

# 1. Define Metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests", ["model_name", "status"])
LATENCY = Histogram("inference_latency_seconds", "Time spent processing request", ["model_name"])
GPU_UTIL = Gauge("gpu_utilization_percent", "Current GPU Util")

# 2. Middleware / endpoint logic
@app.post("/predict/{model_name}")
def predict(model_name: str):
    start = time.time()
    
    # Simulate GPU load
    gpu_load = random.random() * 100
    GPU_UTIL.set(gpu_load)
    
    try:
        # Simulate processing
        time.sleep(random.uniform(0.1, 0.5))
        
        # Success
        REQUEST_COUNT.labels(model_name=model_name, status="success").inc()
        
    except Exception:
        REQUEST_COUNT.labels(model_name=model_name, status="error").inc()
        raise
        
    finally:
        duration = time.time() - start
        LATENCY.labels(model_name=model_name).observe(duration)
        
    return {"result": "ok"}

# 3. Expose /metrics for Prometheus to scrape
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 👨‍💻 Infrastructure: ServiceMonitor (The Glue)

Tell Prometheus Operator to scrape our Service.

#### 📁 `manifests/servicemonitor.yaml`
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: inference-monitor
  namespace: monitoring
  labels:
    release: prometheus-stack # Must match Prometheus Operator selector
spec:
  selector:
    matchLabels:
      app: inference-service # Matches your Deployment/Service labels
  endpoints:
  - port: http # Name of the port in Service
    path: /metrics
    interval: 15s
```

---

## 🔬 Lab Exercise: "PromQL"

### Task
Query the data.
1.  Port-forward Prometheus: `kubectl port-forward svc/prometheus-stack-kube-prom-prometheus 9090:9090 -n monitoring`.
2.  Open `localhost:9090`.
3.  **Query 1 (Throughput):** Rate of requests per second.
    ```promql
    rate(inference_requests_total[1m])
    ```
4.  **Query 2 (Error Rate):**
    ```promql
    sum(rate(inference_requests_total{status="error"}[5m])) / sum(rate(inference_requests_total[5m]))
    ```
5.  **Query 3 (P99 Latency):**
    ```promql
    histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le))
    ```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Labels (Dimensions):** Use labels like `model_version`, `customer_id` strictly. Avoid high-cardinality labels like `request_id` (this will kill Prometheus RAM).
2.  **Scraping:** If your app crashes, Prometheus sees the scrape fail ("Up" metric = 0). This is the most basic health check.
3.  **Exporters:** You don't instrument MySQL directly. You run `mysqld_exporter` sidecar which converts MySQL stats to Prometheus format.

### API Summary
```python
Counter("name", "help", ["label1"]).inc()
Histogram("name", "help").observe(val)
```

---

**Day 141 Complete** ✅

*Next: Day 142 - Grafana - Visualization that doesn't suck.*
