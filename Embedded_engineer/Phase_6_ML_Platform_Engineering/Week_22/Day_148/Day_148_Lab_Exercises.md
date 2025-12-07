# Days 148-154: Week 22 - Observability Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Observability Quick Reference

### Prometheus Custom Metrics
```python
from prometheus_client import Counter, Histogram, start_http_server

PREDICTIONS = Counter('predictions_total', 'Total predictions', ['model'])
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@LATENCY.time()
def predict(model_name, data):
    PREDICTIONS.labels(model=model_name).inc()
    return model.predict(data)

# Start metrics server
start_http_server(8000)
```

### ServiceMonitor
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-api
spec:
  selector:
    matchLabels:
      app: ml-api
  endpoints:
  - port: metrics
    interval: 30s
```

### Grafana Dashboard (JSON)
```json
{
  "panels": [{
    "title": "Predictions/sec",
    "targets": [{
      "expr": "rate(predictions_total[5m])"
    }]
  }]
}
```

### AlertManager Rules
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-alerts
spec:
  groups:
  - name: ml
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.99, prediction_latency_seconds) > 0.5
      for: 5m
```

---

## 📝 Week 22 Summary
| Day | Topic |
|-----|-------|
| 148 | Prometheus |
| 149 | Metrics |
| 150 | Grafana |
| 151 | Alerting |
| 152 | Dashboards |
| 153 | SLOs |
| 154 | Project |
