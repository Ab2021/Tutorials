# Lab 9.5: Monitoring Capstone Project

## 🎯 Objective

Build a **Full Observability Stack**. You will instrument a Python application with Prometheus metrics, run it alongside Prometheus and Grafana, and build a dashboard to monitor its health.

## 📋 Prerequisites

-   Completed Module 9.
-   Docker & Docker Compose.

## 📚 Background

### The Goal
1.  **App**: Python Flask app exposing `/metrics`.
2.  **Collector**: Prometheus scraping the app.
3.  **Visualizer**: Grafana displaying the metrics.

---

## 🔨 Hands-On Implementation

### Step 1: The Instrumented App 🐍

1.  **Create `app/main.py`:**
    ```python
    from flask import Flask
    from prometheus_client import make_wsgi_app, Counter, Histogram
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    import time
    import random

    app = Flask(__name__)

    # Metrics
    REQUEST_COUNT = Counter('app_request_count', 'Total request count')
    REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')

    @app.route('/')
    def hello():
        REQUEST_COUNT.inc()
        with REQUEST_LATENCY.time():
            time.sleep(random.uniform(0.1, 0.5)) # Simulate work
            return "Hello Monitoring!"

    # Add prometheus wsgi middleware to route /metrics requests
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        '/metrics': make_wsgi_app()
    })

    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000)
    ```

2.  **Create `app/Dockerfile`:**
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    RUN pip install flask prometheus_client
    COPY main.py .
    CMD ["python", "main.py"]
    ```

### Step 2: Prometheus Config ⚙️

1.  **Create `prometheus.yml`:**
    ```yaml
    global:
      scrape_interval: 5s

    scrape_configs:
      - job_name: 'flask_app'
        static_configs:
          - targets: ['app:5000']
    ```

### Step 3: Docker Compose 🎼

1.  **Create `docker-compose.yml`:**
    ```yaml
    version: '3'
    services:
      app:
        build: ./app
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
        depends_on:
          - prometheus
    ```

### Step 4: Execution & Visualization 🚀

1.  **Run:**
    ```bash
    docker-compose up --build -d
    ```

2.  **Generate Traffic:**
    Reload `http://localhost:5000` multiple times.

3.  **Grafana Setup:**
    -   Login (admin/admin).
    -   Add Data Source: `http://prometheus:9090`.
    -   Create Dashboard.
    -   **Panel 1 (Traffic)**: `rate(app_request_count_total[1m])`.
    -   **Panel 2 (Latency)**: `rate(app_request_latency_seconds_sum[1m]) / rate(app_request_latency_seconds_count[1m])` (Avg Latency).

---

## 🎯 Challenges

### Challenge 1: Error Rate (Difficulty: ⭐⭐⭐)

**Task:**
1.  Add a route `/error` to the app that returns 500 and increments an `ERROR_COUNT` metric.
2.  Add a Grafana panel showing "Error Rate %".
    Formula: `rate(errors) / rate(total_requests) * 100`.

### Challenge 2: Persistent Dashboards (Difficulty: ⭐⭐⭐)

**Task:**
Configure Grafana in `docker-compose.yml` to load dashboards from a file on startup (Provisioning), so you don't lose them when the container restarts.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Python:
```python
ERROR_COUNT = Counter('app_error_count', 'Total errors')
@app.route('/error')
def error():
    ERROR_COUNT.inc()
    return "Error", 500
```

**Challenge 2:**
Map a volume to `/etc/grafana/provisioning/dashboards` and `/etc/grafana/provisioning/datasources`.
</details>

---

## 🔑 Key Takeaways

1.  **Instrumentation**: You must modify code to get custom metrics (business logic).
2.  **Blackbox vs Whitebox**:
    -   **Blackbox**: Pinging from outside (Nagios).
    -   **Whitebox**: App reporting its own internals (Prometheus). Whitebox is better for debugging.
3.  **Observability**: Metrics + Logs + Traces = Observability.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 9.
You can now monitor your infrastructure.

Proceed to **Module 10: Cloud Fundamentals (AWS)**.
