# Lab 10.8: CloudWatch Monitoring

## Objective
Monitor AWS resources with CloudWatch.

## Learning Objectives
- Create CloudWatch alarms
- Use CloudWatch Logs
- Create custom metrics
- Set up dashboards

---

## CloudWatch Alarms

```bash
# CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=InstanceId,Value=i-1234567890
```

## CloudWatch Logs

```bash
# Create log group
aws logs create-log-group --log-group-name /aws/app/myapp

# Put log events
aws logs put-log-events \
  --log-group-name /aws/app/myapp \
  --log-stream-name app-stream \
  --log-events timestamp=1234567890000,message="Application started"
```

## Custom Metrics

```bash
# Put custom metric
aws cloudwatch put-metric-data \
  --namespace MyApp \
  --metric-name ActiveUsers \
  --value 42 \
  --dimensions Environment=production
```

## Dashboard

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/EC2", "CPUUtilization"]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1"
      }
    }
  ]
}
```

## Success Criteria
✅ Alarms configured  
✅ Logs collected  
✅ Custom metrics published  
✅ Dashboard created  

**Time:** 40 min
