# Lab: Day 37 - Distributed Tracing with Jaeger

## Goal
See a trace in action.

## Prerequisites
- Docker (Jaeger).
- `pip install opentelemetry-distro opentelemetry-exporter-otlp flask requests`

## Step 1: Start Jaeger
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
  jaegertracing/all-in-one:1.50
```
*   **UI**: `http://localhost:16686`

## Step 2: The App (`app.py`)

```python
from flask import Flask
import requests
import time

app = Flask(__name__)

@app.route("/")
def home():
    # Simulate work
    time.sleep(0.1)
    # Call external service (Google)
    requests.get("https://www.google.com")
    return "Hello OTel"

if __name__ == "__main__":
    app.run(port=8000)
```

## Step 3: Run with Auto-Instrumentation
We don't modify the code. We wrap it.

1.  **Install Bootstrap**:
    ```bash
    opentelemetry-bootstrap -a install
    ```

2.  **Run**:
    ```bash
    export OTEL_SERVICE_NAME=my-flask-app
    export OTEL_TRACES_EXPORTER=otlp
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    
    opentelemetry-instrument python app.py
    ```

## Step 4: Generate Traffic
Visit `http://localhost:8000`.

## Step 5: View Trace
1.  Open Jaeger UI (`http://localhost:16686`).
2.  Select Service: `my-flask-app`.
3.  Click "Find Traces".
4.  Click on a trace.
    *   You should see spans for `GET /` and `GET google.com`.

## Challenge
Create a second service (`service_b.py`).
Call Service B from Service A.
See the trace connect them (Distributed Trace).
