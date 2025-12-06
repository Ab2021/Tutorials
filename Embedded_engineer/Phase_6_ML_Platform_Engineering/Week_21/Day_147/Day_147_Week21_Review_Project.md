# Day 147: Week 21 Review & Project - The Glass Cockpit
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** We have individual tools. Now we combine them. A single incident ("Prediction Failed") should let you click from **Alert** -> **metric** -> **Trace** -> **Log** -> **Code**. This is the Holy Grail of Observability.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** a complete LGTM Stack (Loki, Grafana, Tempo, Prometheus) using Docker Compose.
2.  **Instrument** a multi-service app (Gateway -> Model) to emit correlated signals.
3.  **Debug** a complex failure scenario (Latency Spike caused by Memory Leak) using the dashboard.
4.  **Create** a Runbook that guides an operator from "Pager Ringing" to "Problem Solved".

---

## 📚 Week 21 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 141 | Prometheus | "I can calculate the exact Error Rate over 5 minutes." |
| 142 | Grafana | "One dashboard tells me the health of 50 pods." |
| 143 | Alerting | "I only get paged if the user is actually affected." |
| 144 | Logging | "I can grep logs across the entire cluster instantly." |
| 145 | Tracing | "The bottleneck was actually Redis, not the GPU." |
| 146 | Monitoring | "The model thinks everything is a 'Cat'. Drift Alert!" |

---

## 🏗️ Final Project: "Obs-in-a-Box"

### Scenario
You are the SRE for "VisionAPI".
**Architecture:**
1.  **Gateway:** Receives Request. Auth check (Mock). Forward to Model.
2.  **Model Service:** ResNet50 Inference. Randomly fails or slows down.
3.  **Redis:** Cache.

### Step 1: The Stack (Docker Compose)

#### 📁 `project/docker-compose.yaml`
```yaml
version: '3.4'
services:
  # 1. The App
  gateway:
    build: ./gateway
    ports: ["8080:8080"]
    environment:
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://tempo:4317"
      
  model:
    build: ./model
    environment:
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://tempo:4317"

  # 2. Observability Backend
  prometheus:
    image: prom/prometheus
    volumes: [ "./config/prom.yaml:/etc/prometheus/prometheus.yml" ]
    
  loki:
    image: grafana/loki
    
  tempo:
    image: grafana/tempo
    command: [ "-config.file=/etc/tempo.yaml" ]
    
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
```

### Step 2: The Application (Gateway)

#### 📁 `project/gateway/main.py`
```python
from fastapi import FastAPI, HTTPException
from opentelemetry import trace
import httpx
import logging_config # Custom JSON Logger

app = FastAPI()
tracer = trace.get_tracer(__name__)
logger = logging_config.get_logger("gateway")

@app.get("/classify")
async def classify(image_id: str):
    with tracer.start_as_current_span("gateway_process") as span:
        logger.info("Received request", extra={"image_id": image_id})
        
        # Call Model Service (Trace Context propagates automatically via AutoInstrumentation)
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://model:8000/predict?id={image_id}")
            
        if resp.status_code != 200:
            logger.error("Model failure", extra={"status": resp.status_code})
            raise HTTPException(status_code=500)
            
        return resp.json()
```

### Step 3: The Dashboard (Grafana)

**Goal:** Link Traces to Logs.
1.  **DataSource Config:**
    *   Configure Loki.
    *   Derived Fields: Regex `trace_id=(\w+)`. Link to **Tempo**.
2.  **Usage:**
    *   See Log error: `Model failure trace_id=abc12345`.
    *   Click `abc12345`.
    *   Opens Tempo split-screen.
    *   See Waterfall. Model span is Red.
    *   Click Model span.
    *   See Model Logs: `CUDA Out of Memory`.

### Step 4: The Chaos Script

#### 📁 `project/chaos.py`
```python
import time
import requests
import random

def traffic_gen():
    while True:
        # 1. Normal Traffic
        requests.get("http://localhost:8080/classify?image_id=1")
        
        # 2. Occasional Error
        if random.random() < 0.05:
            requests.get("http://localhost:8080/classify?image_id=poison")
            
        time.sleep(0.1)

if __name__ == "__main__":
    traffic_gen()
```

---

## 🔬 Lab Exercise: "The Incident"

### Task
Roleplay.
1.  Run `chaos.py`.
2.  **Alert Fires:** `HighErrorRate` (Slack Notification).
3.  **Investigate:**
    *   Open Grafana.
    *   See "RPS" is steady, but "Errors" spiked.
    *   Filter Logs by `level="error"`.
    *   Find Log: `Model failure`. Trace ID `1a2b3c`.
    *   Open Trace.
    *   Observe: Model service took 5s then failed.
    *   Look at Model metrics: `gpu_memory_util > 99%`.
4.  **Root Cause:** "poison" image triggers OOM.
5.  **Resolution:** Scale up GPU or fix memory leak.

---

## 📝 Success Criteria
1.  **Correlation:** Accessing a Trace ID from a Log line works.
2.  **Completeness:** Dashboard shows metrics from *both* Gateway and Model.
3.  **Usability:** New developer can find the root cause of a 500 error in < 2 minutes.

---

**Week 21 Complete** ✅
**Phase 6D In Progress**

*Next Phase: Data & Model Quality - The Garbage In, Garbage Out problem.*
