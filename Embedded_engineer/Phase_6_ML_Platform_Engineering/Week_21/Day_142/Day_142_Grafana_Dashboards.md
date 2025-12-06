# Day 142: The Cockpit: Grafana Dashboards
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** Prometheus collects the dots. **Grafana** connects them. A good dashboard answers "Is the system healthy?" in 5 seconds without requiring a Ph.D. in PromQL.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Connect** Prometheus as a Data Source in Grafana.
2.  **Construct** rich dashboards with Time Series, Gauges, and Bar Charts.
3.  **Implement** Dashboard Variables to switch between Environments (Dev/Prod) or Models.
4.  **Import** pre-built dashboards for Kubernetes (Nodes/Pods) and NVIDIA GPUs (DCGM).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl` (Grafana installed via `kube-prometheus-stack`).

### Software Environment
- Browser access to Grafana.

---

## 📖 Theoretical Foundation

### 1. The Dashboard JSON Model
Every Grafana dashboard is a JSON file.
*   **Panels:** Individual charts.
*   **Rows:** Group layout.
*   **Templating:** Variables (`$namespace`, `$pod`).
*   **Annotations:** Vertical lines on charts (e.g., "Deployment Happened Here").

### 2. Visualization Types
*   **Time Series:** Line chart. Standard for CPU/Memory/Accuracy over time.
*   **Stat:** Single Big Number. "Current Latency: 45ms". Color code Green/Red.
*   **Table:** Sorted list. "Top 5 CPU consuming pods".
*   **Heatmap:** Distribution over time (great for Latency Histograms).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Port Forwarding

Access Grafana.
```bash
# Default user/pass: admin/prom-operator
kubectl port-forward svc/prometheus-stack-grafana 3000:80 -n monitoring
```
Visit `localhost:3000`.

### 👨‍💻 Core Implementation: The ML Dashboard Layout

We will build a dashboard for our Inference Service.
**Panel 1: Global Request Rate**
*   **Visualization:** Stat.
*   **Query:** `sum(rate(inference_requests_total[1m]))`.
*   **Title:** "Total RPS".

**Panel 2: Latency per Model (Variable)**
*   **Visualization:** Time Series.
*   **Query:** `histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket{model_name="$model"}[5m])) by (le))`.
*   **Title:** "P99 Latency - $model".

**Panel 3: GPU Logic**
*   **Visualization:** Gauge.
*   **Query:** `avg(DCGM_FI_DEV_GPU_UTIL)`.
*   **Thresholds:** 80 (Orange), 90 (Red).

### 👨‍💻 Core Implementation: Dashboard as Code (ConfigMap)

We don't click "Save" in UI (because if Pod restarts, it's gone). We deploy Dashboards as K8s ConfigMaps.

#### 📁 `manifests/dashboard-inference.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1" # Label trigger sidecar to import it
data:
  inference-dashboard.json: |
    {
      "id": null,
      "title": "ML Inference Service",
      "tags": ["mlops"],
      "timezone": "browser",
      "panels": [
        {
          "type": "timeseries",
          "title": "Requests per Second",
          "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
          "targets": [
            {
              "expr": "sum(rate(inference_requests_total[1m])) by (model_name)",
              "legendFormat": "{{model_name}}"
            }
          ]
        },
        {
          "type": "stat",
          "title": "P95 Latency",
          "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
          "targets": [
            {
               "expr": "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "thresholds": {
                "steps": [
                  { "color": "green", "value": null },
                  { "color": "red", "value": 0.5 }
                ]
              },
              "unit": "s"
            }
          }
        }
      ],
      "templating": {
        "list": [
          {
            "name": "namespace",
            "type": "query",
            "query": "label_values(namespace)"
          }
        ]
      }
    }
```

---

## 🔬 Lab Exercise: "Annotations"

### Task
Correlate Deployments with Metrics.
1.  Go to Dashboard Settings -> Annotations.
2.  Add Query: `kube_deployment_created`.
3.  **Result:** Vertical dashed lines appear on your CPU graph whenever a `kubectl apply` happened.
4.  **Insight:** If CPU spikes right after the line, the new code is inefficient.

---

## 📖 Advanced Theory: The RED Method
For a Service (HTTP/RPC), visualize:
1.  **R**ate (Requests/sec).
2.  **E**rrors (Failed requests/sec).
3.  **D**uration (Latency distribution).

For Infrastructure (GPU/Node), use **USE**:
1.  **U**tilization (Time busy).
2.  **S**aturation (Queue length).
3.  **E**rrors.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Variables:** Use variables (`$cluster`, `$namespace`, `$pod`) to make one Dashboard support 1000 microservices.
2.  **Alerting:** Don't stare at dashboards. Configure Alerts (Day 143) to page you. Dashboards are for *investigation* after the pager rings.
3.  **Persistence:** Use `grafana-operator` or ConfigMaps to version control dashboards (JSON files) in Git.

### API Summary
```json
{ "panels": [ ... ], "templating": { ... } }
```

---

**Day 142 Complete** ✅

*Next: Day 143 - Alerting - Waking up at 3 AM.*
