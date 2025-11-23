# Lab 28.4: Rightsizing

## Objective
Optimize resource sizing for cost efficiency.

## Learning Objectives
- Analyze resource utilization
- Identify oversized resources
- Implement rightsizing
- Measure savings

---

## Analyze Utilization

```bash
# Get CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-1234567890 \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-31T23:59:59Z \
  --period 3600 \
  --statistics Average
```

## Rightsizing Recommendations

```python
import boto3

ce = boto3.client('ce')

response = ce.get_rightsizing_recommendation(
    Service='AmazonEC2',
    PageSize=100
)

for rec in response['RightsizingRecommendations']:
    current = rec['CurrentInstance']
    recommended = rec['ModifyRecommendationDetail']['TargetInstances'][0]
    
    print(f"Instance: {current['ResourceId']}")
    print(f"Current: {current['InstanceType']}")
    print(f"Recommended: {recommended['InstanceType']}")
    print(f"Estimated savings: ${rec['EstimatedMonthlySavings']}")
```

## Implement Changes

```bash
# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890

# Modify instance type
aws ec2 modify-instance-attribute \
  --instance-id i-1234567890 \
  --instance-type t3.small

# Start instance
aws ec2 start-instances --instance-ids i-1234567890
```

## Success Criteria
✅ Utilization analyzed  
✅ Oversized resources identified  
✅ Rightsizing implemented  
✅ Cost savings measured  

**Time:** 45 min
