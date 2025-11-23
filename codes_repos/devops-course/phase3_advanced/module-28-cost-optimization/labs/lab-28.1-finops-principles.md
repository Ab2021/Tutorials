# Lab 28.1: FinOps Principles

## Objective
Implement FinOps practices for cloud cost optimization.

## Learning Objectives
- Understand FinOps framework
- Track cloud spending
- Implement cost allocation
- Optimize resources

---

## Cost Visibility

```bash
# AWS Cost Explorer
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

## Tagging Strategy

```hcl
# Terraform tagging
locals {
  common_tags = {
    Environment = var.environment
    Team        = var.team
    CostCenter  = var.cost_center
    Project     = var.project
  }
}

resource "aws_instance" "web" {
  ami           = "ami-12345"
  instance_type = "t2.micro"
  tags          = local.common_tags
}
```

## Cost Alerts

```bash
# Create budget
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "Monthly-Budget",
    "BudgetLimit": {
      "Amount": "1000",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "team@example.com"
    }]
  }]'
```

## Success Criteria
✅ Cost visibility implemented  
✅ Tagging strategy applied  
✅ Budgets and alerts configured  
✅ Cost reports generated  

**Time:** 40 min
