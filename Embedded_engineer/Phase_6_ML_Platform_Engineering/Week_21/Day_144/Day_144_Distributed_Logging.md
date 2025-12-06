# Day 144: Parsing the Matrix: Distributed Logging
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** Metrics tell you *that* something is wrong (Error Rate 5%). Logs tell you *why* it is wrong ("NullPointerException at line 42"). But searching 1 TB of text logs with `grep` is impossible. We need **Loki**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Transition** from `print()` to Structured JSON logging (`{"level": "error", "model": "resnet"}`).
2.  **Deploy** the PLG Stack (Promtail, Loki, Grafana) on Kubernetes.
3.  **Query** logs efficiently using LogQL (e.g., filter by label and grep for substring).
4.  **Correlate** Logs with Traces using a `trace_id`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- `pip install python-json-logger`.

---

## 📖 Theoretical Foundation

### 1. The Text vs Structure debate
*   **Unstructured:** `[2023-10-01] Error processing image 123`.
    *   Hard to parse. Parsing rules break if dev changes generic message.
*   **Structured:** `{"timestamp": "...", "level": "error", "image_id": 123, "msg": "processing failed"}`.
    *   Machine readable. Easy to index.

### 2. ELK vs PLG
*   **ELK (Elasticsearch):** Indexes *every word*. Fast text search. Expensive RAM/Disk.
*   **Loki (Grafana):** Indexes *only metadata* (labels). Greps the raw stream at query time. Cheap. Optimized for K8s (Pod Labels).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Structured Logging (Python)

Never use `print`.

#### 📁 `src/logger_setup.py`
```python
import logging
from pythonjsonlogger import jsonlogger
import sys

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Format as JSON
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Usage
# log = get_logger("inference")
# log.info("Prediction success", extra={"model": "v1", "latency": 0.02})
# Output: {"asctime": "...", "levelname": "INFO", "message": "Prediction success", "model": "v1", "latency": 0.02}
```

### 👨‍💻 Infrastructure: Installing Loki (Helm)

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Installs Promtail (Agent) and Loki (Server)
helm upgrade --install loki grafana/loki-stack -n monitoring
```

*   **Promtail:** Runs as DaemonSet (on every node). Reads `/var/log/pods/*`. Tags logs with K8s Metadata (Namespace, Pod, Container). Pushes to Loki.
*   **Loki:** Stores compressed chunks in S3/MinIO.

### 👨‍💻 Core Implementation: LogQL Queries (Grafana Explore)

1.  Open Grafana -> Explore. switch Datasource to **Loki**.
2.  **Basic Selector:**
    ```promql
    {namespace="default", app="inference-service"}
    ```
3.  **Filter (Grep):**
    ```promql
    {app="inference-service"} |= "error"
    ```
4.  **JSON Parser (Extract fields):**
    ```promql
    {app="inference-service"} | json | latency > 0.5
    ```
    *   This parses the JSON line, extracts `latency`, and filters where it is > 500ms.

---

## 🔬 Lab Exercise: "The Needle in the Haystack"

### Task
Debug a 0.1% failure rate.
1.  App logs "Prediction Failed" occasionally.
2.  Query: `{app="inference"} |= "Prediction Failed"`.
3.  Look at the `image_id` field in those logs.
4.  **Observation:** All failures correspond to `image_id` starting with `png_transparent_`.
5.  **Hypothesis:** Model crashes on Alpha Channel (4 channels instead of 3).
6.  **Fix:** Add `image = image.convert('RGB')` in preprocessing.

---

## 📖 Advanced Theory: Context Propagation
If you have Microservice A -> Microservice B -> Microservice C.
If C fails, you see error in C's logs. But who called C?
**Solution:** Pass `X-Request-ID` header.
Logs: `{"request_id": "req-123", "msg": "..."}`.
In Loki, filter `{job="varilogs"} |= "req-123"` to see the entire timeline across all services.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Stdout:** In K8s, apps should write to `stdout`/`stderr`. Do not write to files. Kubelet captures stdout. Promtail reads Kubelet logs.
2.  **Cost:** Logging is the most expensive observability signal (Volume). Don't log DEBUG in Production. Use "Sampling" if necessary.
3.  **PII:** Be careful not to log "User Input" if it contains PII (GDPR violation). Mask sensitive fields in the Logger formatter.

### API Summary
```python
logger.info("msg", extra={key: val})
```

---

**Day 144 Complete** ✅

*Next: Day 145 - Distributed Tracing - Finding the Bottleneck.*
