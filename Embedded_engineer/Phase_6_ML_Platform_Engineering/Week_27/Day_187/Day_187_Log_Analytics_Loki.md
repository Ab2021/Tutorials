# Day 187: Needles in Haystacks: Log Analytics with Loki
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** Your cluster generates 1TB of logs per day. Use `grep`? Too slow. Use Elasticsearch? Too expensive (indexes text). **Grafana Loki** only indexes metadata (labels), making it 10x cheaper and perfect for "grep distributed" at scale.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** the PLG Stack (Promtail, Loki, Grafana) via Helm.
2.  **Write** LogQL queries to filter JSON logs and calculate error rates (`count_over_time`).
3.  **Correlate** Trace IDs from logs to Tempo traces.
4.  **Implement** Structured Logging (JSON) in Python to enable field-level parsing.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `helm install graphics/loki-stack`.
- `pip install python-json-logger`.

---

## 📖 Theoretical Foundation

### 1. Indexed vs Index-Free
*   **Elasticsearch (ELK):** Indexes *every word*. "Search for 'error'" is fast. "Write 1TB" is heavy (CPU/Disk).
*   **Loki:** Indexes *only labels* (`app=inference`, `env=prod`). "Search for 'error'" requires scanning the chunks (Brute force), but writing is essentially free.
*   **Use Case:** Debugging (Scan last 1 hour) fits Loki perfectly.

### 2. LogQL (Log Query Language)
Inspired by PromQL.
*   **Log Stream Selector:** `{app="nginx"}` (Like Prometheus).
*   **Filter Expression:** `|= "error"` (Grep).
*   **Parser:** `| json` (Extract fields).
*   **Metric Query:** `rate({app="nginx"} |= "error" [5m])` (Convert logs to metrics).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Promtail Config

The agent that tails logs and sends to Loki.

#### 📁 `manifests/promtail-config.yaml`
```yaml
scrape_configs:
- job_name: kubernetes-pods
  pipeline_stages:
  - cri: {} # Parse Container Runtime Interface format
  - match:
      selector: '{app="inference-service"}'
      stages:
      - json:
          expressions:
            model_id: model_id # Extract model_id from JSON log
            latency: latency_ms
      - labels:
          model_id: # Add model_id as a Label (Be careful of high cardinality!)
```

### 👨‍💻 Core Implementation: Structured Logging (Python)

Don't print strings. Print JSON.

#### 📁 `src/logger_setup.py`
```python
import logging
from pythonjsonlogger import jsonlogger
import datetime

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            now = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Usage
# logger.info("Inference complete", extra={"model_id": "v1", "latency_ms": 120})
# Output: {"timestamp": "...", "level": "INFO", "message": "Inference complete", "model_id": "v1", "latency_ms": 120}
```

### 👨‍💻 Core Implementation: LogQL Queries

Run these in Grafana Explore.

```bash
# 1. Simple Grep
{namespace="prod"} |= "Exception"

# 2. JSON Parsing & Filtering
{app="inference"} | json | latency_ms > 500

# 3. Generating Metrics (Alerting)
# "Alert if Error Rate > 1% in last 5m"
sum(rate({app="inference"} |= "error" [5m])) 
/ 
sum(rate({app="inference"} [5m])) > 0.01

# 4. High Cardinality Analysis
# "Show me top 5 IP addresses generating errors"
topk(5, sum by (client_ip) (count_over_time({app="gateway"} | json | status=500 [1h])))
```

---

## 🔬 Lab Exercise: "The Needle"

### Task
Find the specific request causing a crash.
1.  **Scenario:** Every 1,000 requests, one crashes with `index out of bounds`.
2.  **Tool:** Grafana + Loki.
3.  **Query:** `{app="model"} |= "IndexError"`.
4.  **Result:** Found 10 logs.
5.  **Context:** Click "Show Context" (Loads 10 lines before/after).
6.  **Observation:** The log *before* the crash shows `input_shape: [3, 224, 224, 4]`.
7.  **Root Cause:** Model expects 3 channels (RGB), user sent 4 (RGBA).
8.  **Fix:** Add input validation.

---

## 📖 Advanced Theory: Context-Aware Logging
Pass a `TraceID` (from OpenTelemetry) into every log message.
Grafana automatically detects `trace_id=...` in Loki logs and provides a link to Tempo/Jaeger.
This allows you to jump from "Log: DB Error" -> "Trace: SQL Query Span".

---

## 📝 Daily Summary

### Key Takeaways
1.  **Labels are Expensive:** In Prometheus/Loki, labels are indexes. Do NOT label `user_id` (millions of values). Label `cluster`, `namespace`, `app`. Put `user_id` inside the JSON log content.
2.  **Retention:** S3 storage is cheap. Keep logs for 30 days. Use Lifecycle Policies to delete older ones.
3.  **Live Tailing:** `logcli query --tail` is better than `kubectl logs -f` because it aggregates logs from ALL replicas of the service, not just one pod.

### API Summary
```bash
logcli query '{app="foo"}'
```

---

**Day 187 Complete** ✅

*Next: Day 188 - Distributed Tracing - The Request Journey.*
