# Lab 16.4: Alertmanager

## Objective
Configure Alertmanager for alert routing and notification.

## Learning Objectives
- Set up Alertmanager
- Configure routing rules
- Integrate with Slack/PagerDuty
- Implement alert grouping

---

## Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  slack_api_url: 'https://hooks.slack.com/services/XXX'

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'

receivers:
  - name: 'slack'
    slack_configs:
      - channel: '#alerts'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
  
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_KEY'
```

## Alert Routing

```yaml
routes:
  - match:
      team: frontend
    receiver: frontend-team
  - match:
      team: backend
    receiver: backend-team
  - match_re:
      severity: critical|warning
    receiver: oncall
```

## Silencing Alerts

```bash
# Create silence
amtool silence add alertname=HighCPU --duration=2h --comment="Maintenance"

# List silences
amtool silence query

# Expire silence
amtool silence expire <ID>
```

## Success Criteria
✅ Alertmanager receiving alerts  
✅ Slack notifications working  
✅ Routing rules configured  
✅ Alert grouping functional  

**Time:** 40 min
