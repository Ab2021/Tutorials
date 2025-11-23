# Lab 16.7: SLO/SLI Implementation

## Objective
Define and monitor Service Level Objectives and Indicators.

## Learning Objectives
- Define SLIs
- Set SLOs
- Calculate error budgets
- Monitor SLO compliance

---

## Define SLIs

```yaml
# SLI: Availability
availability_sli:
  query: |
    sum(rate(http_requests_total{status!~"5.."}[5m]))
    /
    sum(rate(http_requests_total[5m]))

# SLI: Latency (p95)
latency_sli:
  query: |
    histogram_quantile(0.95,
      rate(http_request_duration_seconds_bucket[5m])
    )
```

## Set SLOs

```yaml
slos:
  - name: API Availability
    sli: availability_sli
    target: 99.9  # 99.9% uptime
    window: 30d
  
  - name: API Latency
    sli: latency_sli
    target: 200  # 200ms p95
    window: 30d
```

## Error Budget

```python
# Calculate error budget
slo_target = 0.999
error_budget = 1 - slo_target  # 0.1%

# Monthly requests
total_requests = 10_000_000
allowed_errors = total_requests * error_budget  # 10,000

# Current status
current_errors = 5000
budget_remaining = (allowed_errors - current_errors) / allowed_errors
print(f"Error budget: {budget_remaining * 100}% remaining")
```

## Success Criteria
✅ SLIs defined  
✅ SLOs set  
✅ Error budget tracked  
✅ Alerts on violations  

**Time:** 40 min
