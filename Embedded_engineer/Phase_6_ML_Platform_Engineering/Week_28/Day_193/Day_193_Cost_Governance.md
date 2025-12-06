# Day 193: The Bill Comes Due: Cost Governance & FinOps
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 28: Platform Engineering Practices

---

> **🎯 Focus Area:** "We spent $50,000 on cloud this month. Who did it?" If you cannot answer this question, you are failing as a Platform Engineer. **FinOps** is the practice of attributing cost to teams and driving accountability.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** Kubecost to visualize Kubernetes Spend by Namespace/Label.
2.  **Calculate** Unit Economics (e.g., Cost per Trained Model, Cost per Prediction).
3.  **Implement** a Tagging Strategy that links AWS resources to Cost Centers.
4.  **Create** Budget Alarms that trigger PagerDuty when spend exceeds 80% forecast.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `helm install kubecost`.

---

## 📖 Theoretical Foundation

### 1. Showback vs Chargeback
*   **Showback:** "Team A, you spent $10k." (Information only. Shame factor).
*   **Chargeback:** "Team A, $10k has been deducted from your department budget." (Financial transaction).
*   **Platform Goal:** Enable Showback first. Chargeback is a culture change.

### 2. Unit Economics
Total Cloud Bill is meaningless if business grows.
*   **Bad Metric:** "Total Cost went up 10%."
*   **Good Metric:** "Cost per User dropped 5%."
*   **ML Metric:** "Cost per Training Hour".

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Deploy Kubecost

The standard for K8s cost visibility.

```bash
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm upgrade --install kubecost kubecost/cost-analyzer \
    --namespace kubecost --create-namespace \
    --set kubecostToken="ZGemo...demo" # Free token
```

### 👨‍💻 Infrastructure: AWS Tagging Policy (Terraform)

Enforce tags on ALL resources using a Provider Default.

#### 📁 `providers.tf`
```hcl
provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {
      Environment = "Production"
      Owner       = "PlatformTeam"
      CostCenter  = "AI-Labs-101"
      ManagedBy   = "Terraform"
    }
  }
}
```

### 👨‍💻 Core Implementation: AWS Budget Alarm

Stop the bleeding early.

#### 📁 `budgets.tf`
```hcl
resource "aws_budgets_budget" "ml_budget" {
  name              = "monthly-ml-budget"
  budget_type       = "COST"
  limit_amount      = "1000"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2023-01-01_00:00"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["manager@myorg.com"]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}
```

### 👨‍💻 Core Implementation: Cost Calculator Script

Estimate Job cost before running it.

#### 📁 `src/cost_estimator.py`
```python
# Pricing Data (Simplified)
PRICING = {
    "g4dn.xlarge": 0.526, # $/hr
    "p3.2xlarge": 3.06,
    "p4d.24xlarge": 32.77
}

def estimate_training(instance_type, hours, spot=False):
    rate = PRICING.get(instance_type, 0)
    if spot:
        rate *= 0.3 # Approx 70% discount
        
    total = rate * hours
    print(f"--- Cost Estimate ---")
    print(f"Instance: {instance_type}")
    print(f"Duration: {hours} hours")
    print(f"Type:     {'Spot' if spot else 'On-Demand'}")
    print(f"Total:    ${total:.2f}")
    return total

# Usage: "Can I run this 48hr job on P3?"
estimate_training("p3.2xlarge", 48, spot=True)
# Result: $44.06. Approved.
```

---

## 🔬 Lab Exercise: "The Zombie Hunt"

### Task
Find wasted money.
1.  **Metrics:** Open Kubecost.
2.  **Filter:** `namespace=dev-notebooks`.
3.  **Sort By:** Idle Cost.
4.  **Finding:** "Jupyter-Bob" has requested 4 GPUs but used 0% GPU for 7 days.
5.  **Cost:** 4 * $3/hr * 24 * 7 = $2,016 wasted.
6.  **Action:** Kill the pod. Setup "Auto-Termination" script (Day 168) to kill idle notebooks after 4 hours.

---

## 📖 Advanced Theory: Spot Market
Spot instances are spare capacity.
*   **Price:** Fluctuates based on supply/demand.
*   **Interruption:** AWS gives 2 minute warning (`SIGTERM`).
*   **Strategy:** Use Spot for Training (Checkpointing saves you). Use On-Demand for Inference (Reliability).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Tag Everything:** If it's not tagged, it's "Unattributed Cost" (The CFO hates this). Tag Policies (Day 191) prevent untagged resources from launching.
2.  **Right-Sizing:** Developers always request `4 GPUs` because "just in case". Show them the data: "You strictly used 1 GPU. Next time request 1."
3.  **Storage Costs:** Data lakes (S3) are "Written once, read never". Use S3 Intelligent Tiering to automatically move old data to Glacier (cheaper).

### API Summary
```bash
aws ce get-cost-and-usage --time-period Start=2023-01-01,End=2023-02-01 --granularity DAILY
```

---

**Day 193 Complete** ✅

*Next: Day 194 - Disaster Recovery - Backup & Restore.*
