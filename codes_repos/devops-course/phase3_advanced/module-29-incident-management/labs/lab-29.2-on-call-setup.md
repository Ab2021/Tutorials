# Lab 29.2: On-Call Setup

## Objective
Set up on-call rotation and incident response.

## Learning Objectives
- Configure on-call schedules
- Set up escalation policies
- Implement runbooks
- Track incident metrics

---

## PagerDuty Setup

```bash
# Create service
curl -X POST https://api.pagerduty.com/services \
  -H 'Authorization: Token token=YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "service": {
      "name": "Production API",
      "escalation_policy": {
        "id": "POLICY_ID"
      }
    }
  }'
```

## Escalation Policy

```yaml
# escalation-policy.yaml
name: "Production Escalation"
escalation_rules:
  - escalation_delay_in_minutes: 0
    targets:
      - type: user
        id: USER_ID_1
  - escalation_delay_in_minutes: 15
    targets:
      - type: user
        id: USER_ID_2
  - escalation_delay_in_minutes: 30
    targets:
      - type: schedule
        id: SCHEDULE_ID
```

## On-Call Schedule

```yaml
# schedule.yaml
name: "Primary On-Call"
time_zone: "America/New_York"
schedule_layers:
  - name: "Weekly Rotation"
    start: "2024-01-01T00:00:00"
    rotation_virtual_start: "2024-01-01T00:00:00"
    rotation_turn_length_seconds: 604800  # 1 week
    users:
      - user: USER_1
      - user: USER_2
      - user: USER_3
```

## Runbook

```markdown
# Incident: API Down

## Severity: P1

## Steps:
1. Check status page: https://status.example.com
2. Verify monitoring: https://grafana.example.com
3. Check recent deployments: `kubectl rollout history deployment/api`
4. Rollback if needed: `kubectl rollout undo deployment/api`
5. Notify stakeholders in #incidents Slack channel

## Escalation:
- 15 min: Page backend team lead
- 30 min: Page engineering manager
- 60 min: Page CTO
```

## Success Criteria
✅ On-call schedule configured  
✅ Escalation policy working  
✅ Runbooks documented  
✅ Incident tracking enabled  

**Time:** 40 min
