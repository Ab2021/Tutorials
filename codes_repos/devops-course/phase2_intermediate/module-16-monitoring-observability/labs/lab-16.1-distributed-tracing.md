# Lab 16.1: Distributed Tracing with Jaeger

## 🎯 Objective

Find the bottleneck. In a microservices architecture, a single user request might touch 10 different services. If the request is slow, which service is to blame? **Distributed Tracing** visualizes the entire lifecycle of a request.

## 📋 Prerequisites

-   Docker & Docker Compose.
-   Python installed.

## 📚 Background

### Concepts
-   **Trace**: The journey of a request through the system.
-   **Span**: A single unit of work (e.g., "DB Query", "HTTP Call").
-   **Context Propagation**: Passing the Trace ID from Service A to Service B (usually via HTTP Headers).

### Tools
-   **Jaeger**: The UI and backend for storing traces.
-   **OpenTelemetry**: The standard library for generating traces.

---

## 🔨 Hands-On Implementation

### Part 1: The Infrastructure (Jaeger) 🕵️‍♂️

1.  **Create `docker-compose.yml`:**
    ```yaml
    version: '3'
    services:
      jaeger:
        image: jaegertracing/all-in-one:1.38
        ports:
          - "16686:16686" # UI
          - "14268:14268" # Collector HTTP
          - "4317:4317"   # OTLP gRPC
    ```

2.  **Run:**
    ```bash
    docker-compose up -d
    ```

3.  **Verify:**
    Open `http://localhost:16686`. You should see the Jaeger UI.

### Part 2: The Microservices 🕸️

We will simulate Service A calling Service B.

1.  **Install Libs:**
    ```bash
    pip install flask opentelemetry-distro opentelemetry-exporter-otlp
    opentelemetry-bootstrap -a install
    ```

2.  **Create `service-b.py` (The Backend):**
    ```python
    from flask import Flask
    import time
    import random

    app = Flask(__name__)

    @app.route("/data")
    def data():
        # Simulate DB query
        time.sleep(random.uniform(0.1, 0.5))
        return "Data from B"

    if __name__ == "__main__":
        app.run(port=5001)
    ```

3.  **Create `service-a.py` (The Frontend):**
    ```python
    from flask import Flask
    import requests

    app = Flask(__name__)

    @app.route("/")
    def home():
        # Call Service B
        resp = requests.get("http://localhost:5001/data")
        return f"Service A got: {resp.text}"

    if __name__ == "__main__":
        app.run(port=5000)
    ```

### Part 3: Auto-Instrumentation 🪄

We don't need to change the code! OpenTelemetry can auto-instrument Flask and Requests.

1.  **Run Service B:**
    ```bash
    export OTEL_SERVICE_NAME=service-b
    export OTEL_TRACES_EXPORTER=otlp
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    opentelemetry-instrument python service-b.py
    ```

2.  **Run Service A (New Terminal):**
    ```bash
    export OTEL_SERVICE_NAME=service-a
    export OTEL_TRACES_EXPORTER=otlp
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    opentelemetry-instrument python service-a.py
    ```

### Part 4: Generate Traffic & Analyze 📉

1.  **Hit Service A:**
    `curl http://localhost:5000` (Run this a few times).

2.  **Check Jaeger:**
    -   Go to `http://localhost:16686`.
    -   Service: `service-a`.
    -   Click **Find Traces**.
    -   Click on a trace.

3.  **Analyze:**
    -   You see a Gantt chart.
    -   Top bar: `service-a` (Total time).
    -   Child bar: `service-b` (Time spent in B).
    -   *Conclusion:* If A takes 500ms and B takes 490ms, the problem is in B.

---

## 🎯 Challenges

### Challenge 1: Manual Spans (Difficulty: ⭐⭐⭐)

**Task:**
In `service-b.py`, add a manual span to track a specific "heavy calculation".
*Hint:*
```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("heavy_calc"):
    time.sleep(0.1)
```
*Note:* You need to import `trace` properly.

### Challenge 2: Error Tagging (Difficulty: ⭐⭐)

**Task:**
Make Service B return a 500 error randomly.
Verify that the span in Jaeger turns **Red**.
*Hint:* OpenTelemetry auto-detects HTTP 5xx codes.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Requires standard `opentelemetry-api` imports. The auto-instrumentation wrapper handles the setup, so `trace.get_tracer` works.

**Challenge 2:**
```python
if random.random() < 0.2:
    return "Error", 500
```
</details>

---

## 🔑 Key Takeaways

1.  **Context Propagation**: The magic happens because `requests` library injects a `traceparent` header into the HTTP call to Service B.
2.  **Sampling**: In production, you don't trace 100% of requests (too expensive). You sample 1% or 0.1%.
3.  **Standardization**: OpenTelemetry (OTEL) is the industry standard. It works with Jaeger, Zipkin, Datadog, NewRelic, etc.

---

## ⏭️ Next Steps

We can see traces. Now let's define what "Good" performance means.

Proceed to **Lab 16.2: Custom Metrics & SLOs**.
