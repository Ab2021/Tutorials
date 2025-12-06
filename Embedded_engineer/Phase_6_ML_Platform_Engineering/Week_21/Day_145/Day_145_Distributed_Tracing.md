# Day 145: Following the Thread: Distributed Tracing
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** Your user says "It's slow". You have a Load Balancer, an API Gateway, an Auth Service, a Feature Store, and a Model Server. Which one is the bottleneck? **Distributed Tracing** visualizes the request lifecycle as a Gantt chart.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Instrument** a Python application with **OpenTelemetry (OTel)**.
2.  **Propagate** context (Trace ID) across HTTP boundaries.
3.  **Visualize** Spans in Jaeger or Grafana Tempo.
4.  **Identify** "Cold Start" variance vs "Network Latency" using trace attributes.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp`.

---

## 📖 Theoretical Foundation

### 1. The Trace and the Span
*   **Trace:** A single user request (e.g., "Predict Customer Churn"). Has a unique `TraceID`.
*   **Span:** A unit of work within a trace (e.g., "Fetch Features from Redis"). Has a `SpanID`, Start Timestamp, Duration, and Parent `SpanID`.

### 2. Context Propagation
To link spans across services (microservices), we inject headers into HTTP requests:
*   `traceparent`: `00-<trace_id>-<span_id>-01`
*   Service B reads this header and creates a child span linked to Service A's span.

### 3. OpenTelemetry (OTel)
The industry standard. Vendor-neutral. You instrument with OTel SDK, and you can export to Jaeger, Zipkin, Datadog, or Honeycomb without changing code.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Running Jaeger (All-in-One)

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.30
```
UI: `localhost:16686`.

### 👨‍💻 Core Implementation: Auto-Instrumentation (FastAPI)

OTel has magic packages that patch libraries (FastAPI, Requests, Boto3) to generate spans automatically.

#### 📁 `src/tracing_app.py`
```python
from fastapi import FastAPI
import time
import requests

# OTel Imports
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 1. Setup Provider
provider = TracerProvider()
# Export to Jaeger via OTLP (gRPC)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

app = FastAPI()

# 2. Instrument Frameworks
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument() # Patches `requests`

tracer = trace.get_tracer(__name__)

@app.get("/predict")
def predict():
    # FastAPI creates the Root Span automatically
    
    with tracer.start_as_current_span("feature_fetch"):
        fetch_features()
        
    with tracer.start_as_current_span("model_inference"):
        result = run_inference()
        
    return {"result": result}

def fetch_features():
    time.sleep(0.1)
    # If we call external service, RequestsInstrumentor adds headers
    # requests.get("http://feature-store/get")

def run_inference():
    time.sleep(0.2)
    return "churn"
```

### 👨‍💻 Core Implementation: Manual Attributes

Adding rich metadata to spans helps debugging.

```python
with tracer.start_as_current_span("db_query") as span:
    span.set_attribute("db.payload.size", 1024)
    span.set_attribute("customer_id", "123")
    
    try:
        query_db()
        span.set_status(trace.Status(trace.StatusCode.OK))
    except Exception as e:
        span.set_status(trace.Status(trace.StatusCode.ERROR))
        span.record_exception(e)
        raise
```

---

## 🔬 Lab Exercise: "The Waterfall"

### Task
Analyze Latency.
1.  Run the app. Hit `/predict`.
2.  Open Jaeger UI (`localhost:16686`). Find trace.
3.  **Visualization:** You see a Waterfall.
    *   `/predict` (Total: 300ms)
        *   `feature_fetch` (100ms)
        *   `model_inference` (200ms)
4.  **Optimization:** If `feature_fetch` and `model_inference` bars don't overlap, they are sequential. Can we parallelize them?
5.  **Action:** Use `asyncio.gather` in Python.
6.  **Verify:** New trace shows spans overlapping. Total time drops to `max(100, 200) = 200ms`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Sampling:** Tracing overhead is high. We sample traces (e.g., 1%). Head-based sampling (decision at start) vs Tail-based sampling (keep only if error occurs).
2.  **Instrumentation:** Start with **Auto-Instrumentation**. Manually instrument only complex internal logic loops.
3.  **Databases:** OTel automatically sanitizes SQL queries (replaces values with `?`) to prevent PII leaks in traces.

### API Summary
```python
with tracer.start_as_current_span("name"):
    ...
```

---

**Day 145 Complete** ✅

*Next: Day 146 - Model Performance Monitoring - Drift & degradation.*
