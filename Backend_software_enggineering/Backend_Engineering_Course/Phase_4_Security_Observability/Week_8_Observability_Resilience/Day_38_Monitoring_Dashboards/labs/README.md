# Lab: Day 38 - Prometheus & Grafana

## Goal
Build a monitoring stack.

## Prerequisites
- Docker Compose.
- `pip install prometheus-client flask`

## Step 1: The App (`app.py`)

```python
from flask import Flask
from prometheus_client import Counter, Histogram, generate_latest
import time
import random

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total Requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Latency', ['endpoint'])

@app.route("/")
def home():
    start = time.time()
    REQUEST_COUNT.labels(method='GET', endpoint='/').inc()
    
    # Simulate latency
    time.sleep(random.uniform(0.1, 0.5))
    
    REQUEST_LATENCY.labels(endpoint='/').observe(time.time() - start)
    return "Hello Monitoring"

@app.route("/metrics")
def metrics():
    return generate_latest()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

## Step 2: Prometheus Config (`prometheus.yml`)

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'my-app'
    static_configs:
      - targets: ['app:5000'] # 'app' is the docker service name
```

## Step 3: Docker Compose (`docker-compose.yml`)

```yaml
version: '3'
services:
  app:
    build: . # Dockerfile for app.py
    ports:
      - "5000:5000"
  
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## Step 4: Run It
1.  Create a `Dockerfile` for the app.
2.  `docker-compose up`.
3.  Generate traffic: `curl localhost:5000` (loop it).

## Step 5: Visualize
1.  **Prometheus**: `http://localhost:9090`. Query `http_requests_total`.
2.  **Grafana**: `http://localhost:3000` (admin/admin).
    *   Add Data Source -> Prometheus (`http://prometheus:9090`).
    *   Create Dashboard -> Add Panel -> `rate(http_requests_total[1m])`.

## Challenge
Add an **Error Rate** panel.
Modify `app.py` to randomly return 500 errors.
Query: `sum(rate(http_requests_total{status="500"}[1m])) / sum(rate(http_requests_total[1m]))`.
