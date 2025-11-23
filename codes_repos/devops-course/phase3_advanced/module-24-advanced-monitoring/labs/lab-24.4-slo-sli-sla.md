# Lab 24.4: SLO/SLI/SLA

## Objective
Define and monitor Service Level Objectives.

## Learning Objectives
- Define SLIs and SLOs
- Calculate error budgets
- Monitor SLO compliance
- Implement SLA reporting

---

## Define SLIs

```yaml
# SLI: Availability
availability_sli:
  description: "Percentage of successful requests"
  query: "sum(rate(http_requests_total{status!~'5..'}[5m])) / sum(rate(http_requests_total[5m]))"

# SLI: Latency
latency_sli:
  description: "95th percentile latency"
  query: "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
```

## Define SLOs

```yaml
slos:
  - name: "API Availability"
    sli: availability_sli
    target: 99.9  # 99.9% uptime
    window: 30d
  
  - name: "API Latency"
    sli: latency_sli
    target: 200  # 200ms p95
    window: 30d
```

## Error Budget

```python
# Calculate error budget
slo_target = 0.999  # 99.9%
error_budget = 1 - slo_target  # 0.1%

# Monthly budget (30 days)
total_requests = 1_000_000
allowed_errors = total_requests * error_budget  # 1,000 errors

# Current errors
current_errors = 500
budget_remaining = (allowed_errors - current_errors) / allowed_errors * 100
print(f"Error budget remaining: {budget_remaining}%")
```

## Prometheus Rules

```yaml
groups:
  - name: slo
    rules:
      - record: slo:availability:ratio
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[5m]))
          /
          sum(rate(http_requests_total[5m]))
      
      - alert: SLOBudgetExhausted
        expr: slo:availability:ratio < 0.999
        for: 5m
        annotations:
          summary: "SLO budget exhausted"
```

## Success Criteria
✅ SLIs defined  
✅ SLOs configured  
✅ Error budget calculated  
✅ Alerts firing on violations  

**Time:** 45 min
