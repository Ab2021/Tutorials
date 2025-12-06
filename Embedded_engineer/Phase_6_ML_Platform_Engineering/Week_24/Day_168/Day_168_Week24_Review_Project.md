# Day 168: Week 24 Review & Project - The Cost Cutting Hackathon
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** We have learned the tools (Spot, Autoscaling, Dashboards). Now we apply them. Your mission: Take a bloated, expensive ML platform and cut the bill by 50% without affecting performance.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Analyze** a legacy infrastructure to find specific cost leaks.
2.  **Refactor** Terraform to implement Mixed Instance Policies and Tags.
3.  **Deploy** a "Reaper" bot that cleans up abandoned resources.
4.  **Present** a "Before vs After" cost report demonstrating ROI.

---

## 📚 Week 24 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 162 | Cost Analysis | "Leaving a notebook open cost us \$378." |
| 163 | Spot Instances | "Training is 90% cheaper if we handle interruptions." |
| 164 | Autoscaling | "Scale to Zero is the ultimate money saver." |
| 165 | Multi-Tenancy | "Sharing GPUs (Time-slicing) quadrupled our dev capacity." |
| 166 | Serverless | "For low traffic, Lambda beats EC2." |
| 167 | Dashboards | "I can see exactly how much 'Team Risk' spent yesterday." |

---

## 🏗️ Final Project: "Project Squeeze"

### Scenario
"Startup.ai" has a bill of **\$10,000/month**.
**Audit Findings:**
1.  **Training:** Running on `p3.2xlarge` On-Demand. (\$3.06/hr). Utilization 20%.
2.  **Inference:** Fixed cluster of 5 `g4dn.xlarge`. Traffic is zero at night.
3.  **Storage:** 50 EBS volumes "Available" (Unattached) from failed experiments. 
4.  **Dev:** Developers leave persistent workspaces running 24/7.

### Step 1: Justification Report

| Resource | Current State | Proposed Optimizations | Est. Savings |
| :--- | :--- | :--- | :--- |
| **Training** | On-Demand P3 | Spot G4dn (Mixed Policy) + Ray Checkpointing | 90% |
| **Inference** | Fixed Cluster (5 nodes) | KEDA Autoscaling (0-5 nodes) | 60% |
| **Storage** | 10TB Orphaned EBS | Delete Unattached Volumes | 100% |
| **Dev** | 24/7 Instances | Auto-Stop at 7 PM (Lambda Reaper) | 65% |

### Step 2: The Reaper Script

#### 📁 `project/reaper.py`
```python
import boto3
from datetime import datetime

ec2 = boto3.client('ec2')

def cleanup_orphaned_volumes():
    print("Scanning for orphaned volumes...")
    volumes = ec2.describe_volumes(Filters=[{'Name': 'status', 'Values': ['available']}])
    
    total_gb = 0
    for v in volumes['Volumes']:
        vid = v['VolumeId']
        size = v['Size']
        print(f"Deleting {vid} ({size} GB)")
        ec2.delete_volume(VolumeId=vid)
        total_gb += size
    
    print(f"Reclaimed {total_gb} GB of storage.")

def stop_dev_instances():
    print("Checking for Dev instances running after hours...")
    # Find running instances tagged Environment=Dev
    instances = ec2.describe_instances(Filters=[
        {'Name': 'instance-state-name', 'Values': ['running']},
        {'Name': 'tag:Environment', 'Values': ['Dev']}
    ])
    
    for r in instances['Reservations']:
        for i in r['Instances']:
            iid = i['InstanceId']
            print(f"Stopping {iid}...")
            ec2.stop_instances(InstanceIds=[iid])

if __name__ == "__main__":
    cleanup_orphaned_volumes()
    stop_dev_instances()
```

### Step 3: The Optimized Infrastructure (Training)

#### 📁 `project/infra/training_node_group.tf`
```hcl
resource "aws_eks_node_group" "training_spot" {
  cluster_name    = var.cluster_name
  node_group_name = "training-spot"
  
  # Scaling Config
  scaling_config {
    desired_size = 0 # Scale to 0 when not training
    max_size     = 10
    min_size     = 0
  }

  # Instance Types (Diversified)
  instance_types = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge"]
  
  # Spot Config
  capacity_type = "SPOT"
  
  labels = {
    "role" = "training-worker"
    "karpenter.sh/capacity-type" = "spot"
  }
}
```

### Step 4: The Autoscaling Inference (KEDA)

#### 📁 `project/manifests/inference-scaler.yaml`
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: inference-scaler
spec:
  scaleTargetRef:
    name: inference-deployment
  minReplicaCount: 1 # Keep 1 warm for low latency
  maxReplicaCount: 5
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server
      metricName: gpu_util
      threshold: "80"
      query: score_gpu_utilization
```

---

## 🔬 Lab Exercise: "The Board Meeting"

### Task
Simulate ROI Presentation.
1.  **Bill Calculation:**
    *   Old: $10,000.
    *   New:
        *   Training: Spot (10% of cost) = $300.
        *   Inference: Scaled down 12h/day = $600.
        *   Storage: Cleaned = $50.
        *   Dev: Stopped at night = $1000.
    *   **Total New:** ~$2,000.
2.  **Savings:** $8,000 / month ($96k / year).
3.  **Metric:** "We extended the startup runway by 1 month just by fixing the Terraform config."

---

## 📝 Success Criteria
1.  **Tag Coverage:** 100% of resources have `Owner` and `CostCenter` tags.
2.  **Zero Waste:** No unattached volumes or IPs exist for > 24h.
3.  **Elasticity:** Clusters shrink when jobs finish.

---

**Week 24 Complete** ✅
**Phase 6D Completed**

*Next Phase: Phase 6E - Advanced Topics & Capstone.*
