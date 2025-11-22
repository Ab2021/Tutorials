# Lab 09: SLO Definition & Alerting

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
- Define Service Level Objectives
- Create Prometheus alerts
- Implement error budgets

## Problem Statement
Define an SLO: \"99% of messages processed within 1 second\". Create Prometheus alerting rules to fire when the SLO is violated (error budget exhausted).

## Starter Code
```yaml
# prometheus-rules.yml
groups:
  - name: slo_alerts
    rules:
      - alert: HighLatency
        expr: ???
        for: 5m
```

## Hints
<details>
<summary>Hint 1</summary>
Use histogram metrics and `histogram_quantile` function.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**SLO Definition:**
- **SLI**: P99 latency < 1s
- **SLO**: 99% availability (error budget: 1%)
- **Measurement Window**: 30 days

**Prometheus Rules:**
```yaml
groups:
  - name: slo_alerts
    rules:
      # SLI: P99 Latency
      - record: sli:latency:p99
        expr: histogram_quantile(0.99, rate(flink_latency_bucket[5m]))
      
      # SLO Violation
      - alert: SLOViolation_Latency
        expr: sli:latency:p99 > 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: \"P99 latency exceeds 1s\"
          description: \"Current P99: {{ $value }}s\"
      
      # Error Budget Burn Rate
      - alert: ErrorBudgetBurnRate
        expr: |
          (
            1 - (
              sum(rate(requests_success[1h])) /
              sum(rate(requests_total[1h]))
            )
          ) > 0.01
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: \"Burning error budget too fast\"
```

**Alertmanager Config:**
```yaml
route:
  receiver: 'slack'
  group_by: ['alertname']
  
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'YOUR_WEBHOOK_URL'
        channel: '#alerts'
```

**Verification:**
```bash
# Check rules
curl http://localhost:9090/api/v1/rules

# Trigger alert (simulate high latency)
# Wait 5 minutes
# Check firing alerts
curl http://localhost:9090/api/v1/alerts
```
</details>
