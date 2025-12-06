# Day 167: Show Me the Money: FinOps Dashboards
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** Engineers ignore costs until the CFO yells. **FinOps Dashboards** put cost metrics right next to system metrics like latency, making "Spend" a first-class citizen in the engineering decision loop.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Extract** daily spend data using the AWS Cost Explorer API (Programmatic Access).
2.  **Deploy** **Kubecost** to visualize spend per Namespace, Deployment, and Service.
3.  **Build** a Grafana Panel showing "Accumulated Cost This Month" vs "Budget".
4.  **Configure** Real-time Anomaly Detection alerts (e.g., "Spend velocity doubled in 1 hour").

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install boto3`.
- Kubecost Helm Chart.

---

## 📖 Theoretical Foundation

### 1. Cost Allocation
*   **AWS Bill:** Shows "EC2: $5000".
*   **The Problem:** Which team spent it? Was it the `payment-model` or `fraud-model`?
*   **Kubecost:** Connects to AWS Pricing API + K8s Metrics API.
    *   Pod A used 2 CPUs for 1 hour. Price of 2 CPUs on this node is $0.10. Cost = $0.10.

### 2. Anomaly Detection
*   **Static Threshold:** "Alert if > $1000/day". (Fails on Black Friday).
*   **Trend Based:** "Alert if Cost > 30-day Moving Average + 3 Sigma". (Catch runaway loops).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Installing Kubecost

```bash
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm install kubecost kubecost/cost-analyzer \
    --namespace kubecost --create-namespace \
    --set kubecostToken="ZG5hbWU9a3ViZWNvc3Q=" 
```
Access UI: `kubectl port-forward svc/kubecost-cost-analyzer 9090:9090 -n kubecost`.

### 👨‍💻 Core Implementation: AWS Cost Export to Grafana

If you don't use Kubecost, pull data into Prometheus via a custom exporter.

#### 📁 `src/cost_exporter.py`
```python
import boto3
import time
from prometheus_client import start_http_server, Gauge
from datetime import datetime, timedelta

# Metric
DAILY_COST = Gauge('aws_daily_cost_usd', 'Daily spend', ['service'])

ce = boto3.client('ce')

def fetch_cost():
    now = datetime.utcnow()
    start = (now - timedelta(days=1)).strftime('%Y-%m-%d')
    end = now.strftime('%Y-%m-%d')
    
    # Get Cost by Service
    resp = ce.get_cost_and_usage(
        TimePeriod={'Start': start, 'End': end},
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )
    
    for group in resp['ResultsByTime'][0]['Groups']:
        service_name = group['Keys'][0]
        amount = float(group['Metrics']['UnblendedCost']['Amount'])
        DAILY_COST.labels(service=service_name).set(amount)
        print(f"{service_name}: ${amount:.2f}")

if __name__ == "__main__":
    start_http_server(8000)
    print("Cost Exporter running on :8000")
    while True:
        fetch_cost()
        time.sleep(3600) # Update hourly (API charges apply if too frequent)
```

### 👨‍💻 Infrastructure: Budget Alarm (Terraform)

Stop the bleeding automatically.

#### 📁 `infra/budget.tf`
```hcl
resource "aws_budgets_budget" "ml_budget" {
  name              = "monthly-ml-budget"
  budget_type       = "COST"
  limit_amount      = "1000"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["manager@example.com"]
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED" # AI Prediction
    subscriber_email_addresses = ["manager@example.com"]
  }
}
```

---

## 🔬 Lab Exercise: "The Leak"

### Task
Simulate a Cost Anomaly.
1.  **Baseline:** Daily spend $50.
2.  **Action:** Launch a script that creates 1TB S3 objects in a loop.
3.  **Observation:**
    *   Hour 1: Spend normal.
    *   Hour 4: AWS Budget Forecast triggers (if configured for intra-day).
    *   **Kubecost:** View "Savings" tab. It detects "Over-provisioned storage".
4.  **Dashboard:** Grafana Panel "Projected End of Month Bill" jumps from $1500 to $50,000.
5.  **Alert:** PagerDuty rings. "Projected Budget Exceeded".
6.  **Resolution:** Kill script. Delete bucket.

---

## 📖 Advanced Theory: Spot Savings Analysis
You need to prove Spot was worth it.
*   **Metric:** `spot_savings_usd = (on_demand_price - spot_price) * hours`.
*   **Dashboard:** Show "Money Saved This Year". This number justifies the Engineering Effort spent on building fault-tolerant training loops (Day 163).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Visibility:** Developers don't waste money on purpose. They waste it because they don't see the price tag. Show cost in the PR ("This change increases estimated cost by $5/day").
2.  **Rightsizing:** Kubecost tells you "You requested 4 CPUs but used 0.1 CPUs". Fix your `requests`.
3.  **Tagging Hygiene:** If untagged resources exist, the dashboard shows "Unallocated Cost". This should be $0.

### API Summary
```python
ce.get_cost_and_usage(Granularity='DAILY')
```

---

**Day 167 Complete** ✅

*Next: Day 168 - Week 24 Review & Project - The FinOps Optimization.*
