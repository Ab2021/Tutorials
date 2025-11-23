# Lab 16.1: Prometheus Advanced

## Objective
Implement advanced Prometheus monitoring with custom metrics and alerting.

## Learning Objectives
- Configure Prometheus scraping
- Create custom metrics
- Write PromQL queries
- Set up alerting rules

---

## Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'app'
    static_configs:
      - targets: ['localhost:8080']
    
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

## Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Counter
requests_total = Counter('http_requests_total', 'Total HTTP requests')

# Histogram
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Gauge
active_users = Gauge('active_users', 'Number of active users')

@app.route('/')
def index():
    requests_total.inc()
    with request_duration.time():
        return "Hello"

start_http_server(8000)
```

## PromQL Queries

```promql
# Rate of requests
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

## Alerting Rules

```yaml
# alerts.yml
groups:
  - name: example
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="500"}[5m]) > 0.05
        for: 10m
        annotations:
          summary: "High error rate detected"
```

## Success Criteria
✅ Prometheus scraping metrics  
✅ Custom metrics exposed  
✅ PromQL queries working  
✅ Alerts configured  

**Time:** 45 min
