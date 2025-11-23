# Lab 28.3: Resource Tagging

## Objective
Implement comprehensive resource tagging for cost allocation.

## Learning Objectives
- Design tagging strategy
- Enforce tagging policies
- Generate cost reports by tags
- Automate tag compliance

---

## Tagging Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Deny",
    "Action": ["ec2:RunInstances"],
    "Resource": "*",
    "Condition": {
      "StringNotLike": {
        "aws:RequestTag/Environment": ["dev", "staging", "prod"],
        "aws:RequestTag/Owner": "*",
        "aws:RequestTag/CostCenter": "*"
      }
    }
  }]
}
```

## Auto-Tagging Lambda

```python
import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')
    
    # Get instance ID from event
    instance_id = event['detail']['instance-id']
    
    # Auto-tag with creator
    creator = event['detail']['userIdentity']['principalId']
    
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            {'Key': 'Creator', 'Value': creator},
            {'Key': 'CreatedAt', 'Value': event['time']}
        ]
    )
```

## Cost Report by Tags

```bash
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=TAG,Key=Team \
  --group-by Type=TAG,Key=Environment
```

## Success Criteria
✅ Tagging policy enforced  
✅ Auto-tagging working  
✅ Cost reports by tags  
✅ Tag compliance >95%  

**Time:** 40 min
