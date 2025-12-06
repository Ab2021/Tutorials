# Day 143: Don't Wake Me Up: Alerting Best Practices
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** "High CPU Usage" is a bad alert. "Users cannot predict" is a good alert. Learn to define **SLOs (Service Level Objectives)** and configure **AlertManager** to ping Slack during the day and PagerDuty at night.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** SLIs (Indicators) and SLOs (Objectives) for an Inference Service.
2.  **Author** a `PrometheusRule` manifest to detect High Error Rates.
3.  **Configure** AlertManager Routes to dispatch critical alerts to PagerDuty.
4.  **Implement** "Inhibition" rules (Don't alert on "Pod Down" if "Cluster Down" is firing).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- Kube-Prometheus-Stack installed.

---

## 📖 Theoretical Foundation

### 1. The Hierarchy of Reliability
*   **SLI (Indicator):** The metric. "Latency".
*   **SLO (Objective):** The target. "99% of requests < 200ms".
*   **SLA (Agreement):** The contract. "If SLO missed, we pay you back". (Business/Legal).

### 2. The Golden Signals (Google SRE)
Alert on Symptoms, not Causes.
1.  **Latency:** Is it slow?
2.  **Traffic:** Is it zero? (Did the load balancer die?)
3.  **Errors:** Is it 500ing?
4.  **Saturation:** Is disk full?

### 3. Alert Fatigue
If you get 100 emails a day, you ignore them all. Then the site goes down, and you ignore that too.
**Rule:** Every Pager alert must require actionable intelligence.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Defining the Rule (PrometheusRule)

Detect if > 1% of requests are errors for 5 minutes.

#### 📁 `manifests/alert-rules.yaml`
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: inference-alerts
  namespace: monitoring
  labels:
    release: prometheus-stack
spec:
  groups:
  - name: inference.rules
    rules:
    # 1. High Error Rate (Critical)
    - alert: HighErrorRate
      expr: |
        sum(rate(inference_requests_total{status="error"}[5m])) 
        / 
        sum(rate(inference_requests_total[5m])) > 0.01
      for: 2m  # Wait 2m before firing (avoid blips)
      labels:
        severity: critical
      annotations:
        summary: "High Error Rate on {{ $labels.model_name }}"
        description: "Error rate is {{ $value | humanizePercentage }} (Threshold 1%)"

    # 2. High Latency (Warning)
    - alert: HighLatency
      expr: |
        histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le)) > 0.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Slow Response Time"
```

### 👨‍💻 Infrastructure: Routing (AlertManager Config)

Send CRITICAL to PagerDuty. Send WARNING to Slack.

#### 📁 `manifests/alertmanager-config.yaml`
```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'model_name']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'slack-notifications' # Default

  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-ops'

receivers:
- name: 'slack-notifications'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/...'
    channel: '#mlops-alerts'
    send_resolved: true

- name: 'pagerduty-ops'
  pagerduty_configs:
  - service_key: 'YOUR_PD_KEY'
```

---

## 🔬 Lab Exercise: "Silence"

### Task
Maintenance Window.
1.  You are upgrading the cluster. You know alerts will fire.
2.  Go to AlertManager UI (`localhost:9093`).
3.  Create **Silence**.
4.  Matcher: `severity="warning"`. Duration: 2h.
5.  **Result:** Alerts are suppressed. No pager triggers.
6.  **Safety:** Silence expires automatically.

---

## 📖 Advanced Theory: Burn Rate Alerts
Instead of alerting on "Error Rate > 1%", alert on "Error Budget Burn".
If SLO is 99.9%, you have 0.1% Error Budget.
*   **Fast Burn:** If burning 10% of budget per hour -> Page immediately.
*   **Slow Burn:** If burning 1% of budget per day -> Ticket.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Deduplication:** AlertManager groups alerts. If 100 Pods fail, you get ONE email listing all 100 pods, not 100 emails.
2.  **Hysteresis:** Use `for: 5m`. This prevents alerts flapping (On/Off/On) if the metric hovers near threshold.
3.  **Runbooks:** Every Alert should link to a Runbook (Markdown in Git) telling the On-Call engineer what to do. "Check logs, Restart pod, Rollback".

### API Summary
```yaml
kind: PrometheusRule
spec:
  groups:
  - rules:
    - alert: Name
      expr: promql
```

---

**Day 143 Complete** ✅

*Next: Day 144 - Distributed Logging - ELK vs Loki.*
