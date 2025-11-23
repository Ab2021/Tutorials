# Lab 29.3: Incident Response with PagerDuty

## Objective
Implement incident response workflows with PagerDuty.

## Learning Objectives
- Configure PagerDuty
- Set up on-call schedules
- Create escalation policies
- Integrate with monitoring tools

---

## PagerDuty Setup

```python
import requests

PAGERDUTY_API_KEY = 'your-api-key'
PAGERDUTY_EMAIL = 'your-email@example.com'

headers = {
    'Authorization': f'Token token={PAGERDUTY_API_KEY}',
    'Accept': 'application/vnd.pagerduty+json;version=2',
    'Content-Type': 'application/json',
    'From': PAGERDUTY_EMAIL
}

# Create service
service_data = {
    'service': {
        'name': 'Production API',
        'description': 'Main production API service',
        'escalation_policy': {
            'id': 'ESCALATION_POLICY_ID',
            'type': 'escalation_policy_reference'
        }
    }
}

response = requests.post(
    'https://api.pagerduty.com/services',
    headers=headers,
    json=service_data
)
```

## Trigger Incident

```python
def trigger_incident(title, description, severity='high'):
    incident_data = {
        'incident': {
            'type': 'incident',
            'title': title,
            'service': {
                'id': 'SERVICE_ID',
                'type': 'service_reference'
            },
            'urgency': severity,
            'body': {
                'type': 'incident_body',
                'details': description
            }
        }
    }
    
    response = requests.post(
        'https://api.pagerduty.com/incidents',
        headers=headers,
        json=incident_data
    )
    return response.json()

# Trigger incident
trigger_incident(
    'High CPU Usage',
    'CPU usage exceeded 90% for 5 minutes',
    'high'
)
```

## Escalation Policy

```json
{
  "escalation_policy": {
    "name": "Production Escalation",
    "escalation_rules": [
      {
        "escalation_delay_in_minutes": 15,
        "targets": [
          {
            "id": "PRIMARY_ONCALL_SCHEDULE_ID",
            "type": "schedule_reference"
          }
        ]
      },
      {
        "escalation_delay_in_minutes": 30,
        "targets": [
          {
            "id": "SECONDARY_ONCALL_SCHEDULE_ID",
            "type": "schedule_reference"
          }
        ]
      }
    ]
  }
}
```

## Success Criteria
✅ PagerDuty configured  
✅ On-call schedules created  
✅ Escalation policies working  
✅ Incidents triggered and resolved  

**Time:** 40 min
