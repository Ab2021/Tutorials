# Day 64: The 800lb Gorilla: ML on AWS
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 10: Cloud Platforms for ML

---

> **🎯 Focus Area:** Taking Kubernetes to the Cloud. Understanding the specific components of **Amazon Web Services (AWS)** required for High-Performance Distributed Training and Inference.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** the correct EC2 instance family (`p4d` vs `g5`) for Training vs Inference.
2.  **Deploy** an EKS (Elastic Kubernetes Service) cluster with GPU Node Groups using `eksctl`.
3.  **Explain** the role of EFA (Elastic Fabric Adapter) in multi-node training.
4.  **Configure** FSx for Lustre to accelerate data loading from S3.
5.  **Use** IRSA (IAM Roles for Service Accounts) to secure Pod access to S3.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- AWS Account (with Quota for GPU instances).
- *Warning:* `p4d` instances cost $30+/hr. We will simulate configuration.

### Software Environment
- `aws` CLI.
- `eksctl` CLI.
- `kubectl`.

---

## 📖 Theoretical Foundation

### 1. The Instance Zoo
*   **P-Series (Training):**
    *   `p3.2xlarge`: 1x V100 (Old reliable).
    *   `p4d.24xlarge`: 8x A100 (400Gbps networking). The standard for LLM training.
*   **G-Series (Inference):**
    *   `g4dn`: T4 GPU. Cheap inference.
    *   `g5`: A10G. Better inference/small training.

### 2. Networking (EFA)
Standard Ethernet on AWS is ~25Gbps. Distributed Training needs 400Gbps+ with low latency.
**EFA (Elastic Fabric Adapter):** AWS implementation of OS-Bypass networking (like InfiniBand). Requires specific drivers and nccl-plugin.

### 3. Storage (FSx for Lustre)
Loading 1M images from S3 directly is slow (high latency per file).
**FSx for Lustre:** A high-performance file system that "lazy loads" from S3.
*   First read: Slow (Hydrates from S3).
*   Second read: Fast (Sub-millisecond from Lustre disk).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: EKS Cluster Config

We use `eksctl` to provision the control plane and node groups via Infrastructure as Code.

#### 📁 `aws/eks-gpu-cluster.yaml`
```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ml-platform
  region: us-east-1
  version: "1.27"

managedNodeGroups:
  # 1. CPU Pool (System pods, Controllers)
  - name: system-pool
    instanceType: m5.large
    minSize: 2
    maxSize: 5
    
  # 2. Inference Pool (Cheap GPUs)
  - name: inference-pool
    instanceType: g4dn.xlarge # 1x T4
    minSize: 0 # Scale to 0 to save money
    maxSize: 10
    labels:
      accelerator: nvidia-t4
    tags:
      k8s.io/cluster-autoscaler/enabled: "true"
      
  # 3. Training Pool (A100s) - Simulated Config
  # In real life, needs specific subnets + EFA security groups
  - name: training-pool
    instanceType: p4d.24xlarge
    minSize: 0
    maxSize: 4
    labels:
      accelerator: nvidia-a100
      networking: efa
    efs: true # Mount EFS/FSx
```

### 👨‍💻 Core Implementation: IRSA (IAM for Pods)

Never put AWS Access Keys in K8s Secrets. Use OIDC.

1.  **Create IAM Policy:** Allow `s3:GetObject` on `s3://my-dataset`.
2.  **Create IAM Role:** Trust the EKS OIDC provider.
3.  **Annotate ServiceAccount:**
    
    ```yaml
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: s3-reader
      annotations:
        eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/S3ReaderRole
    ```
4.  **Pod Usage:**
    ```yaml
    spec:
      serviceAccountName: s3-reader
    ```
    *Result:* AWS injects `AWS_ROLE_ARN` and `AWS_WEB_IDENTITY_TOKEN_FILE` env vars. `boto3` picks them up automatically.

---

## 🔬 Lab Exercise: "Instance Pricing Calculator"

### Task
Calculate the cost of training Llama-2-70B.
*   Requires: ~1024 A100 GPUs for 21 days.
*   Instance: `p4d.24xlarge` (8x A100) = $32/hr.
*   Count: 128 Instances.
*   Cost per Hour: $32 * 128 = $4,096.
*   Total: $4,096 * 24 * 21 = **$2,064,384**.

### Insight
Cloud Platform Engineering is largely **FinOps**.
*   Using **Spot Instances** (60-90% discount) for fault-tolerant workloads (e.g., Checkpointing frequent runs) is essential.
*   Using `g5.48xlarge` instead of `p4d` for simple debugging can save thousands.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Managed Nodes:** EKS Managed Node Groups handle OS patching and Driver installation (mostly). Use the `AL2_x86_64_GPU` AMI type.
2.  **Storage Tiers:** S3 (Cheap, Slow-ish) -> EFS (Expensive, POSIX) -> FSx Lustre (Expensive, Fast Cache for S3).
3.  **Networking:** If you don't enable EFA on `p4d` instances, you are paying for a Ferrari and driving it in a school zone.

### API Summary
```bash
eksctl create cluster -f config.yaml
aws eCR get-login-password | docker login ...
```

---

**Day 64 Complete** ✅

*Next: Day 65 - GCP for ML Workloads - GKE, TPUs, and how Google does it.*
