# Lab 28.2: Cloud Cost Monitoring

## Objective
Implement comprehensive cloud cost monitoring and alerting.

## Learning Objectives
- Set up cost monitoring
- Create cost budgets
- Implement cost alerts
- Analyze spending trends

---

## AWS Cost Explorer

```python
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')

# Get cost and usage
response = ce.get_cost_and_usage(
    TimePeriod={
        'Start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        'End': datetime.now().strftime('%Y-%m-%d')
    },
    Granularity='DAILY',
    Metrics=['UnblendedCost'],
    GroupBy=[
        {'Type': 'DIMENSION', 'Key': 'SERVICE'}
    ]
)

for result in response['ResultsByTime']:
    print(f"Date: {result['TimePeriod']['Start']}")
    for group in result['Groups']:
        service = group['Keys'][0]
        cost = group['Metrics']['UnblendedCost']['Amount']
        print(f"  {service}: ${float(cost):.2f}")
```

## Cost Budgets

```python
# Create budget
budgets = boto3.client('budgets')

response = budgets.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'Monthly-Budget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST'
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80,
                'ThresholdType': 'PERCENTAGE'
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'team@example.com'
                }
            ]
        }
    ]
)
```

## Cost Anomaly Detection

```python
# Get cost anomalies
response = ce.get_anomalies(
    DateInterval={
        'StartDate': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'EndDate': datetime.now().strftime('%Y-%m-%d')
    }
)

for anomaly in response['Anomalies']:
    print(f"Anomaly detected:")
    print(f"  Impact: ${anomaly['Impact']['TotalImpact']}")
    print(f"  Service: {anomaly['RootCauses'][0]['Service']}")
```

## Success Criteria
✅ Cost monitoring configured  
✅ Budgets created  
✅ Alerts working  
✅ Anomalies detected  

**Time:** 40 min
