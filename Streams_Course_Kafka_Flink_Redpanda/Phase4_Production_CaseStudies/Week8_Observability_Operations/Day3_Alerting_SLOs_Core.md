# Alerting & SLOs

## Core Concepts

### SLI/SLO/SLA
- SLI (Indicator): Metric (P99 latency)
- SLO (Objective): Target (< 100ms)
- SLA (Agreement): Contract with penalty

### Alert Types
- Symptom-based: User impact (high latency)
- Cause-based: Root cause (disk full)

### Error Budget
- 99.9% uptime = 43 minutes downtime/month
- Burn rate: How fast we consume budget
