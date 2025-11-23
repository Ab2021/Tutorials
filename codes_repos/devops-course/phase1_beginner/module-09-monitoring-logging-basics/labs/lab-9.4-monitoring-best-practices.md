# Lab 9.4: Monitoring Best Practices

## Objective
Implement monitoring best practices for production systems.

## Learning Objectives
- Design monitoring strategy
- Set up alerting rules
- Create dashboards
- Implement SLOs

---

## Monitoring Strategy

```yaml
# Monitoring layers
monitoring_strategy:
  infrastructure:
    - CPU, Memory, Disk
    - Network metrics
    - System logs
  
  application:
    - Request rate
    - Error rate
    - Response time
    - Custom business metrics
  
  user_experience:
    - Page load time
    - Transaction success rate
    - User satisfaction
```

## Alert Rules

```yaml
# Prometheus alert rules
groups:
  - name: production_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for {{ $labels.instance }}"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

## Dashboard Design

```json
{
  "dashboard": {
    "title": "Production Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }]
      }
    ]
  }
}
```

## Success Criteria
✅ Monitoring strategy defined  
✅ Alerts configured  
✅ Dashboards created  
✅ Best practices implemented  

**Time:** 40 min
