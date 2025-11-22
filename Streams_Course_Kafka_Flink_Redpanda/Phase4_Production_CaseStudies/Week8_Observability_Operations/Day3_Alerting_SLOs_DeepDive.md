# Alerting - Deep Dive

## Internals

### Multi-Window Multi-Burn-Rate
Google SRE approach:
- Fast burn (1h window): Page immediately
- Slow burn (6h window): Ticket

### Alert Fatigue
Avoid:
- Alerts on metrics, not symptoms
- Too sensitive thresholds
- Non-actionable alerts
