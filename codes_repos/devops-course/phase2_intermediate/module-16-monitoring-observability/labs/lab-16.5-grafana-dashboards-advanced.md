# Lab 16.5: Grafana Dashboards Advanced

## Objective
Create advanced Grafana dashboards with complex visualizations.

## Learning Objectives
- Build custom dashboards
- Use variables and templating
- Create alerts
- Share and export dashboards

---

## Dashboard with Variables

```json
{
  "dashboard": {
    "title": "Application Metrics",
    "templating": {
      "list": [
        {
          "name": "environment",
          "type": "query",
          "query": "label_values(up, environment)",
          "multi": true
        },
        {
          "name": "instance",
          "type": "query",
          "query": "label_values(up{environment=~\"$environment\"}, instance)"
        }
      ]
    },
    "panels": [
      {
        "title": "CPU Usage",
        "targets": [
          {
            "expr": "rate(cpu_usage{instance=\"$instance\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

## Alert Rules

```json
{
  "alert": {
    "name": "High CPU",
    "conditions": [
      {
        "evaluator": {
          "params": [80],
          "type": "gt"
        },
        "query": {
          "model": {
            "expr": "cpu_usage"
          }
        }
      }
    ],
    "notifications": [
      {
        "uid": "slack-notifier"
      }
    ]
  }
}
```

## Success Criteria
✅ Custom dashboards created  
✅ Variables working  
✅ Alerts configured  
✅ Dashboards exported  

**Time:** 45 min
