# Lab 16.2: Custom Metrics & SLOs

## 🎯 Objective

Move beyond "CPU Usage". Business cares about "User Experience". You will define **SLIs** (Indicators), **SLOs** (Objectives), and **SLAs** (Agreements), and implement them using Prometheus and Python.

## 📋 Prerequisites

-   Completed Lab 9.5 (Prometheus basics).

## 📚 Background

### The SRE Terminology
1.  **SLI (Service Level Indicator)**: What we measure. (e.g., "Request Latency").
2.  **SLO (Service Level Objective)**: The goal. (e.g., "99% of requests < 200ms").
3.  **SLA (Service Level Agreement)**: The contract. (e.g., "If we miss the SLO, we pay you back").
4.  **Error Budget**: `100% - SLO`. The amount of time we are allowed to be broken.

---

## 🔨 Hands-On Implementation

### Part 1: The App with Custom Metrics 🐍

We need a metric that counts "Good" vs "Bad" requests.

1.  **Create `app.py`:**
    ```python
    from flask import Flask
    from prometheus_client import start_http_server, Counter
    import time
    import random

    app = Flask(__name__)

    # Metrics
    TOTAL_REQS = Counter('http_requests_total', 'Total Requests')
    FAST_REQS = Counter('http_requests_fast_total', 'Requests under 200ms')

    @app.route("/")
    def home():
        TOTAL_REQS.inc()
        latency = random.uniform(0.1, 0.3) # 100ms to 300ms
        time.sleep(latency)
        
        if latency < 0.2:
            FAST_REQS.inc()
            return "Fast"
        return "Slow"

    if __name__ == "__main__":
        start_http_server(8000) # Metrics on port 8000
        app.run(port=5000)      # App on port 5000
    ```

2.  **Run:**
    ```bash
    python app.py
    ```

### Part 2: Prometheus Config ⚙️

1.  **`prometheus.yml`:**
    ```yaml
    scrape_configs:
      - job_name: 'slo_app'
        static_configs:
          - targets: ['localhost:8000']
    ```

2.  **Run Prometheus:**
    (Use Docker as in previous labs).

### Part 3: Calculating the SLI (PromQL) 🧮

1.  **Generate Traffic:**
    Hit `localhost:5000` 10-20 times.

2.  **Query in Prometheus:**
    We want the % of requests that are fast.
    
    **Formula:** `Fast Requests / Total Requests`
    
    **PromQL:**
    ```promql
    rate(http_requests_fast_total[1m]) / rate(http_requests_total[1m])
    ```
    *Result:* Should be around 0.5 (50%).

### Part 4: Alerting on SLO Breach 🚨

We want to alert if the success rate drops below 90% (0.9).

1.  **Create Alert Rule:**
    ```yaml
    groups:
    - name: slo_alerts
      rules:
      - alert: HighLatency
        expr: (rate(http_requests_fast_total[1m]) / rate(http_requests_total[1m])) < 0.9
        for: 1m
        labels:
          severity: page
        annotations:
          summary: "SLO Breach: Too many slow requests"
    ```

---

## 🎯 Challenges

### Challenge 1: Error Budget Burn Rate (Difficulty: ⭐⭐⭐⭐)

**Task:**
Advanced SRE concept. Instead of alerting when you break the SLO, alert when you are *burning* the budget too fast.
*Concept:* If you burn 2% of your monthly budget in 1 hour, page the on-call engineer.
*Research:* Look up "Google SRE Multiwindow Burn Rate Alerts".

### Challenge 2: Histogram SLO (Difficulty: ⭐⭐⭐)

**Task:**
Instead of a custom `FAST_REQS` counter, use a standard **Histogram**.
Use `histogram_quantile(0.99, ...)` to see the 99th percentile latency.
SLO: "P99 must be < 250ms".

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
Python:
```python
LATENCY = Histogram('http_request_duration_seconds', 'Latency', buckets=[0.1, 0.2, 0.5])
```
PromQL (P99):
```promql
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```
Alert: `... > 0.25`
</details>

---

## 🔑 Key Takeaways

1.  **Business Value**: SLOs align Dev and Ops. If you have Error Budget left, Devs can push features. If Budget is exhausted, Devs must stop and fix reliability.
2.  **Golden Signals**: Latency, Traffic, Errors, Saturation.
3.  **Alert Fatigue**: Don't alert on every error. Alert on Symptom (SLO Breach), not Cause (CPU High).

---

## ⏭️ Next Steps

We have Observability. Now let's handle Logs at scale.

Proceed to **Module 17: Logging & Log Management**.
