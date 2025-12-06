# Day 162: The Bill Arrives: Cloud Cost Analysis
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** You built a brilliant model. But it costs \$50 to train and generates \$5 of revenue. You are essentially burning venture capital. **FinOps** is the discipline of mapping Cloud Spend to Business Value.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Breakdown** the AWS Bill by Service (EC2 vs S3 data transfer) and Tag (Team/Project).
2.  **Calculate** Unit Economics: "Cost per Training Run" and "Cost per 1M Predictions".
3.  **Identify** Zombie Resources (Unattached EBS Volumes, Idle Load Balancers).
4.  **Implement** Budget Alerts to prevent "Cloud Bill Shock".

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install boto3 pandas`.

---

## 📖 Theoretical Foundation

### 1. The Pricing Matrix
*   **On-Demand:** Pay by the second. Most expensive. Flexible.
*   **Reserved Instances (RI):** Commit to 1-3 years. 40-60% discount. Good for DBs.
*   **Savings Plans:** Commit to \$10/hr spend. Flexible across instance families.
*   **Spot Instances:** Bid on spare capacity. 90% discount. Can be interrupted. Good for Training.

### 2. The Hidden Costs
*   **Data Transfer:** Ingress is free. Egress (S3 to Internet) is expensive. Cross-AZ (Zone A to Zone B) costs money.
*   **API Requests:** `s3:ListObjects` costs money. Infinite loops in code can bankrupt you.
*   **EBS IOPS:** Provisioned IOPS (io1/io2) are very expensive.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Cost Allocation Tags

You cannot optimize what you cannot measure. Force tags via Terraform.

```hcl
provider "aws" {
  default_tags {
    tags = {
      Environment = "Production"
      CostCenter  = "ML-Platform"
      Owner       = "DataScienceTeam"
    }
  }
}
```

### 👨‍💻 Core Implementation: Identifying Zombie Resources

Python script to find waste.

#### 📁 `src/find_waste.py`
```python
import boto3
from datetime import datetime, timedelta

ec2 = boto3.client('ec2')
cloudwatch = boto3.client('cloudwatch')

def find_idle_instances():
    # 1. List all Instances
    instances = ec2.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
    
    for r in instances['Reservations']:
        for i in r['Instances']:
            instance_id = i['InstanceId']
            
            # 2. Check CPU Utilization (Last 7 Days)
            stats = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=datetime.utcnow() - timedelta(days=7),
                EndTime=datetime.utcnow(),
                Period=3600,
                Statistics=['Average']
            )
            
            # Logic: If Max Avg CPU < 5% for 7 days -> Zombie
            data_points = [p['Average'] for p in stats['Datapoints']]
            if data_points and max(data_points) < 5.0:
                 print(f"IDLE INSTANCE FOUND: {instance_id} ({i['InstanceType']})")

def find_unattached_volumes():
    volumes = ec2.describe_volumes(Filters=[{'Name': 'status', 'Values': ['available']}])
    for v in volumes['Volumes']:
        print(f"ORPHAN VOLUME: {v['VolumeId']} ({v['Size']} GB) - Costing money!")

if __name__ == "__main__":
    find_idle_instances()
    find_unattached_volumes()
```

### 👨‍💻 Core Implementation: Cost per Prediction Calculator

Measuring Unit Economics.

#### 📁 `src/unit_economics.py`
```python
# Constants (Approx AWS Pricing)
COST_G4DN_XLARGE_HR = 0.526
COST_LOAD_BALANCER_HR = 0.0225
COST_S3_GB_MONTH = 0.023

def calculate_inference_cost(daily_requests, avg_latency_ms, autoscaling=True):
    # Throughput per instance (Requests per Second per Instance)
    # Assume 1 instance handles: 1000ms / latency * concurrency
    concurrency = 4
    rps_per_instance = (1000 / avg_latency_ms) * concurrency
    
    total_reqs_per_hr = daily_requests / 24
    
    if autoscaling:
        # Scale instances based on load
        required_instances = total_reqs_per_hr / (rps_per_instance * 3600)
    else:
        # Provision for Peak (assume 2x Avg)
        required_instances = (total_reqs_per_hr * 2) / (rps_per_instance * 3600)
    
    # Ensure min 1
    required_instances = max(1, required_instances)
    
    daily_compute_cost = required_instances * COST_G4DN_XLARGE_HR * 24
    daily_lb_cost = COST_LOAD_BALANCER_HR * 24
    
    total_daily = daily_compute_cost + daily_lb_cost
    cost_per_1k = (total_daily / daily_requests) * 1000
    
    return total_daily, cost_per_1k

# Scenario: 1M reqs/day, 100ms latency
daily_cost, cpk = calculate_inference_cost(1_000_000, 100)
print(f"Daily Bill: ${daily_cost:.2f}")
print(f"Cost per 1k predictions: ${cpk:.4f}")
```

---

## 🔬 Lab Exercise: "The Forgotton Notebook"

### Task
Simulate real waste.
1.  Launch a `g4dn.xlarge` Notebook.
2.  Use it for 1 hour.
3.  Close the browser tab. (Do NOT Stop the instance).
4.  **Result:** It runs for 30 days. Bill = $0.526 * 24 * 30 = $378.
5.  **Fix:** Implement a "Reaper Script" (Lambda) that stops any Notebook tagged `Dev` at 7 PM every day.

---

## 📖 Advanced Theory: GPU Multi-Instance GPU (MIG)
An A100 GPU costs $4/hr.
If your model only needs 10% of it, you waste $3.60/hr.
**MIG:** Slices 1 Physical A100 into 7 Virtual GPUs.
You can run 7 small models on 1 A100.
**Impact:** Increases utilization from 10% to 70%.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Ownership:** Every resource must have an `Owner` tag. If a resource has no tag, delete it (after warning).
2.  **Spot:** Use Spot Instances for **Stateless** workloads (Training, Batch Inference). Never use Spot for a Database or Single-Replica API.
3.  **Storage:** Delete old StepFunction logs, S3 buckets, and ECR images using Lifecycle Policies.

### API Summary
```python
boto3.client('ce').get_cost_and_usage(...)
```

---

**Day 162 Complete** ✅

*Next: Day 163 - Living on the Edge: Spot Instances.*
