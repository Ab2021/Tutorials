# Day 70: Week 10 Review & Project - Cloud Architecture Design
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 10: Cloud Platforms for ML

---

> **🎯 Focus Area:** You are now a Cloud Architect. Synthesize your knowledge of AWS/GCP/Azure, Cost Optimization, and HA to design a production-grade infrastructure for an AI Unicorn.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Hybrid Cloud architecture that balances Performance, Cost, and Compliance.
2.  **Select** appropriate Compute, Storage, and Networking services across providers.
3.  **Create** Terraform/IaC snippets for the proposed infrastructure.
4.  **Defend** architectural decisions (e.g., "Why EKS over DIY on EC2?").

---

## 📚 Week 10 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 64 | AWS ML | "EFA is required for Multi-Node Training performance." |
| 65 | GCP ML | "TPUs are powerful but require XLA compatibility." |
| 66 | Azure ML | "InfiniBand in the cloud is unique to Azure." |
| 67 | Cost Opt | "Spot instances save 70%, but I need to handle preemption." |
| 68 | HA | "Zone failures happen. Spread your pods." |
| 69 | Hybrid | "Don't move the Petabyte. Move the Compute." |

---

## 🏗️ Final Project: "NeuroScale Infrastructure"

### Scenario
**Startup:** NeuroScale.
**Product:** Real-time Medical Imaging Analysis (HIPAA compliant).
**Constraints:**
1.  **Data:** 5 PB of Patient Data resides in a physical Secure Data Center (SDC) in Boston. Cannot leave premises.
2.  **Training:** Needs 100+ GPUs to retrain models weekly. SDC has no space/power for this.
3.  **Inference:** Doctors worldwide (US, EU, Asia) need <100ms latency.
4.  **Budget:** Startups need to burn slowly.

### Solution Design

#### 1. Hybrid Training Pipeline (AWS + Direct Connect)
*   **Architecture:**
    *   **Connectivity:** AWS Direct Connect (10Gbps) between SDC and `us-east-1`.
    *   **Data Loading:** Use **Amazon FSx for Lustre**. Connect it to the SDC via NFS/VPN. It caches the "Hot" dataset for the weekly run.
    *   **Compute:** **EKS** Cluster with **Karpenter**.
    *   **Instances:** `p4d.24xlarge` (Spot if possible, but reliability is key for weekly runs).
*   **Workflow:**
    1.  Job starts.
    2.  FSx hydrates caching layer from SDC (burst download).
    3.  Training runs on AWS.
    4.  Enhanced Model Weights pushed to S3.

#### 2. Global Inference (Multi-Region GKE)
*   **Why GCP?** Excellent global networking and GKE Autopilot for low ops.
*   **Regions:** `us-central1`, `europe-west1`, `asia-northeast1`.
*   **Compute:** **GKE Autopilot** (Scale to zero support).
*   **Storage:** Model weights pulled from S3 (using Multi-Cloud bucket replication or CDNs).
*   **Traffic:** Google Cloud Load Balancer (Anycast IP) routes doctor to nearest cluster.

#### 3. Cost Control (FinOps)
*   **Training:** Use Spot Instances for Hyperparameter Tuning jobs. Use On-Demand for final run.
*   **Inference:** Use **KNA (Knative)** or GKE Autopilot to pay $0 when no doctors are using it (Night time).

### 👨‍💻 Infrastructure as Code (Terraform concept)

#### 📁 `terraform/main.tf`
```hcl
# 1. The Training Cluster (AWS)
module "eks_training" {
  source = "terraform-aws-modules/eks/aws"
  cluster_name = "neuroscale-training-v1"
  cluster_version = "1.27"
  
  vpc_id = module.vpc.vpc_id
  
  # Node Group for heavy lifting
  eks_managed_node_groups = {
    gpu_workers = {
      instance_types = ["p4d.24xlarge"]
      min_size     = 0
      max_size     = 20
      desired_size = 0 # Controlled by Karpenter
      ami_type     = "AL2_x86_64_GPU"
    }
  }
}

# 2. Karpenter for Spot support
resource "helm_release" "karpenter" {
  name       = "karpenter"
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  # ... config ...
}

# 3. Connectivity (Direct Connect Gateway)
resource "aws_dx_gateway" "main" {
  name            = "neuroscale-dx-gw"
  amazon_side_asn = "64512"
}
```

---

## 🔬 Lab Exercise: "Failure Mode Analysis"

### Task
Simulate failures in the design.
1.  **Direct Connect Cut:** A backhoe digs up the fiber in Boston.
    *   *Impact:* Training fails. Inference continues (Weights already in S3).
    *   *Mitigation:* Specific VPN Backup (slower, but functional).
2.  **US-East-1 AWS Outage:**
    *   *Impact:* Training delayed. Inference continues (GCP).
3.  **GCP Europe Outage:**
    *   *Impact:* European doctors routed to US-Central (Latency increases to 150ms, but works). Design resiliency!

---

## 📝 Success Criteria
1.  **Compliance:** No patient data is stored permanently in the cloud (FSx cache is ephemeral).
2.  **Performance:** Inference is local to the user.
3.  **Cost:** Expensive GPUs run only when needed.

---

**Week 10 Complete** ✅

*Next Week: Week 11 - Container Optimization - Making your images smaller, faster, and more secure.*
