# Lab 28.5: Reserved Instances Strategy

## Objective
Implement reserved instances strategy for cost optimization.

## Learning Objectives
- Analyze usage patterns
- Calculate RI savings
- Purchase reserved instances
- Monitor RI utilization

---

## Usage Analysis

```python
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')

# Get RI recommendations
response = ce.get_reservation_purchase_recommendation(
    Service='Amazon Elastic Compute Cloud - Compute',
    LookbackPeriodInDays='SIXTY_DAYS',
    TermInYears='ONE_YEAR',
    PaymentOption='NO_UPFRONT'
)

for recommendation in response['Recommendations']:
    details = recommendation['RecommendationDetails']
    print(f"Instance Type: {details['InstanceDetails']['EC2InstanceDetails']['InstanceType']}")
    print(f"Recommended quantity: {details['RecommendedNumberOfInstancesToPurchase']}")
    print(f"Estimated monthly savings: ${details['EstimatedMonthlySavingsAmount']}")
    print(f"Estimated savings percentage: {details['EstimatedSavingsPercentage']}%")
```

## RI Purchase

```python
# Purchase RI
ec2 = boto3.client('ec2')

response = ec2.purchase_reserved_instances_offering(
    InstanceCount=10,
    ReservedInstancesOfferingId='offering-id',
    DryRun=False
)

print(f"Reserved Instances ID: {response['ReservedInstancesId']}")
```

## RI Utilization Monitoring

```python
# Get RI utilization
response = ce.get_reservation_utilization(
    TimePeriod={
        'Start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        'End': datetime.now().strftime('%Y-%m-%d')
    },
    Granularity='MONTHLY'
)

for result in response['UtilizationsByTime']:
    utilization = result['Total']
    print(f"RI Utilization: {utilization['UtilizationPercentage']}%")
    print(f"Unused hours: {utilization['TotalActualHours'] - utilization['TotalRunningHours']}")
```

## Savings Plan

```python
# Get Savings Plan recommendations
response = ce.get_savings_plans_purchase_recommendation(
    SavingsPlansType='COMPUTE_SP',
    TermInYears='ONE_YEAR',
    PaymentOption='NO_UPFRONT',
    LookbackPeriodInDays='SIXTY_DAYS'
)

for recommendation in response['SavingsPlansPurchaseRecommendation']['SavingsPlansPurchaseRecommendationDetails']:
    print(f"Hourly commitment: ${recommendation['HourlyCommitmentToPurchase']}")
    print(f"Estimated monthly savings: ${recommendation['EstimatedMonthlySavingsAmount']}")
```

## Success Criteria
✅ Usage analyzed  
✅ RI recommendations generated  
✅ RIs purchased  
✅ Utilization monitored  

**Time:** 45 min
