# Day 188: Where did the time go? Distributed Tracing with Tempo
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** "The API is slow (500ms)." "Why?" "I don't know." **Distributed Tracing** visualizes the request lifecycle across 20 Microservices, DBs, and Queues to pinpoint the exact 450ms bottleneck.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Instrument** a Python application using OpenTelemetry (OTel) Auto-Instrumentation.
2.  **Propagate** Context (Traceparent Headers) across HTTP and Kafka boundaries.
3.  **Implement** "Tail Sampling" to keep only 1% of success traces but 100% of error traces.
4.  **Visualize** a Waterfall Trace in Grafana Tempo.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `helm install tempo`.
- `pip install opentelemetry-distro opentelemetry-exporter-otlp`.

---

## 📖 Theoretical Foundation

### 1. Spans & Traces
*   **Span:** A single unit of work (e.g., "SQL Query", "HTTP GET"). Has Start/End time.
*   **Trace:** A tree of Spans sharing a `TraceID`. Represents one user request.
*   **Context Propagation:** Service A calls Service B. A sends `trace-id=123` in the HTTP Header. B uses `trace-id=123`. This links the graph.

### 2. Sampling
*   **Head Sampling:** Decide at the start (Gateway). "Sample 1%". Efficient, but misses rare errors.
*   **Tail Sampling:** Collect ALL spans in a buffer. At the end, check: "Was there an error? If yes, Keep. If no, keep 1%". (Requires more memory).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: OTel Collector

The agent that receives spans and sends to Tempo.

#### 📁 `manifests/otel-collector.yaml`
```yaml
apiVersion: opentelemetry.io/v1alpha1
kind: OpenTelemetryCollector
metadata:
  name: otel
spec:
  config: |
    receivers:
      otlp:
        protocols:
          grpc:
          http:
    processors:
      batch:
      memory_limiter:
        limit_mib: 400
    exporters:
      otlp:
        endpoint: "tempo:4317" # Send to Tempo
        tls:
          insecure: true
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [otlp]
```

### 👨‍💻 Core Implementation: Auto-Instrumentation (No Code Change)

Python Wrapper.

```bash
# Install Distro
opentelemetry-bootstrap -a install

# Run App with Tracing enabled
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
export OTEL_SERVICE_NAME="inference-service"

opentelemetry-instrument python app.py
```

### 👨‍💻 Core Implementation: Manual Instrumentation (Custom Spans)

When you need to trace inside a generic function.

#### 📁 `src/custom_span.py`
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def heavy_computation():
    with tracer.start_as_current_span("matrix_multiplication") as span:
        span.set_attribute("matrix.size", "1024x1024")
        # Do work...
        try:
            result = 1 / 0
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
```

### 👨‍💻 Core Implementation: Baggage (Context Propagation)

Passing User ID deep into the stack.

#### 📁 `src/baggage_demo.py`
```python
from opentelemetry import baggage, context

# Service A
def handler(request):
    # Set User ID in context
    ctx = baggage.set_baggage("user_id", "12345")
    token = context.attach(ctx)
    call_service_b()

# Service B (Automatically receives baggage via HTTP Header)
def service_b_handler():
    # Read User ID (Even though it wasn't an argument)
    uid = baggage.get_baggage("user_id")
    print(f"Processing for user {uid}")
```

---

## 🔬 Lab Exercise: "The 300ms Delay"

### Task
Find the bottleneck.
1.  **Scenario:** `GET /predict` takes 500ms.
2.  **Trace Analysis:**
    *   Span 1: `ingress` (500ms).
    *   Span 2: `app.predict` (490ms).
    *   Span 3: `db.query` (10ms).
    *   Span 4: `redis.get` (**450ms**).
3.  **Insight:** Redis is slow?
4.  **Deep Dive:** Redis Span shows `command="KEYS *"`.
5.  **Root Cause:** Developer used `KEYS *` (O(N) Scan) instead of `SCAN` or `GET`.
6.  **Fix:** Change to `GET`. Latency drops to 50ms.

---

## 📖 Advanced Theory: Trace-Metric Correlation
If a Trace is slow, was the CPU high?
Tempo links to Prometheus.
Grafana Config: "Node Graph".
*   Click on Span -> "Logs for this Span" (Loki).
*   Click on Span -> "Host Metrics" (Prometheus).
This "Exemplar" integration means you don't hunt for timestamps manually.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Don't Trace Everything:** Sampling 100% of production traffic is too expensive (Storage/Network). 1-10% is usually enough for statistics.
2.  **Async Boundaries:** Tracing breaks at Queues (Kafka/SQS) if you don't handle it. The Producer must inject TraceContext into the Message Headers. The Consumer must extract it.
3.  **Overhead:** OTel adds ~1ms latency per span. Negligible for HTTP APIs, significant for High Frequency Trading.

### API Summary
```python
tracer.start_as_current_span("name")
```

---

**Day 188 Complete** ✅

*Next: Day 189 - Week 27 Review & Project - The War Room.*
