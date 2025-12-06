# Day 198: Titan Phase 1: The Data Plane Foundation
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 29: Capstone Project Part 1

---

> **🎯 Focus Area:** An architecture diagram is nice, but it doesn't run code. Today we build the physical foundation of **Titan**: The AWS VPCs, the EKS Clusters, and the Network Mesh that connects them.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Provision** a Multi-Region AWS Network (US-East-1 + EU-Central-1) using Terraform Modules.
2.  **Establish** VPC Peering (or Transit Gateway) to allow private cross-region communication.
3.  **Boot** 3 EKS Clusters (Control, Train, Serve) with correct Managed Node Groups.
4.  **Install** the Base System Addons (CoreDNS, VPC CNI, EBS CSI) via GitOps.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- AWS Account (Free Tier not sufficient).

### Software Environment
- `terraform`, `aws-cli`.

---

## 📖 Theoretical Foundation

### 1. Network Topology
*   **Mesh:** All-to-All. Good for small number of regions (3).
*   **Hub-and-Spoke:** Transit Gateway. Good for large scale (10+).
*   **Titan Choice:** Mesh (VPC Peering) for simplicity/cost between 2 regions.

### 2. EKS Architecture
*   **Control Plane:** Managed by AWS.
*   **Data Plane:** Self-Managed Node Groups (Simulated) or Managed Node Groups.
*   **Networking:** Pods get real VPC IPs (AWS VPC CNI).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: The Network Module

A reusable Terraform module.

#### 📁 `iac/modules/vpc/main.tf`
```hcl
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  name   = var.name
  cidr   = var.cidr

  azs             = ["${var.region}a", "${var.region}b"]
  private_subnets = [cidrsubnet(var.cidr, 8, 1), cidrsubnet(var.cidr, 8, 2)]
  public_subnets  = [cidrsubnet(var.cidr, 8, 101), cidrsubnet(var.cidr, 8, 102)]

  enable_nat_gateway = true
  single_nat_gateway = true # Save cost
  
  tags = {
    Terraform   = "true"
    Environment = "dev"
    Project     = "Titan"
  }
}
```

### 👨‍💻 Infrastructure: The Global Stack

Deploying 2 Regions.

#### 📁 `iac/live/main.tf`
```hcl
provider "aws" {
  alias  = "us"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu"
  region = "eu-central-1"
}

# 1. US Network
module "vpc_us" {
  source    = "../modules/vpc"
  providers = { aws = aws.us }
  name      = "titan-us"
  cidr      = "10.1.0.0/16"
  region    = "us-east-1"
}

# 2. EU Network
module "vpc_eu" {
  source    = "../modules/vpc"
  providers = { aws = aws.eu }
  name      = "titan-eu"
  cidr      = "10.2.0.0/16"
  region    = "eu-central-1"
}

# 3. Peering (The Bridge)
resource "aws_vpc_peering_connection" "us_eu" {
  provider      = aws.us
  vpc_id        = module.vpc_us.vpc_id
  peer_vpc_id   = module.vpc_eu.vpc_id
  peer_region   = "eu-central-1"
  auto_accept   = false
}

resource "aws_vpc_peering_connection_accepter" "eu_accepter" {
  provider                  = aws.eu
  vpc_peering_connection_id = aws_vpc_peering_connection.us_eu.id
  auto_accept               = true
}
```

### 👨‍💻 Infrastructure: The EKS Clusters

#### 📁 `iac/live/clusters.tf`
```hcl
module "eks_control" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 19.0"
  cluster_name    = "titan-control"
  cluster_version = "1.28"
  vpc_id          = module.vpc_us.vpc_id
  subnet_ids      = module.vpc_us.private_subnets
  
  eks_managed_node_groups = {
    general = {
      min_size     = 2
      max_size     = 3
      desired_size = 2
      instance_types = ["t3.large"]
    }
  }
}

module "eks_gpu" {
  source       = "terraform-aws-modules/eks/aws"
  cluster_name = "titan-gpu"
  # ...
  eks_managed_node_groups = {
    gpu_workers = {
      min_size     = 0
      max_size     = 10
      desired_size = 1
      instance_types = ["g4dn.xlarge"]
      ami_type       = "AL2_x86_64_GPU"
    }
  }
}
```

### 👨‍💻 Core Implementation: Bootstrapping ArgoCD

The "Seed" that grows the rest of the platform.

```bash
# Connect to Control Cluster
aws eks update-kubeconfig --name titan-control

# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Apply the Root App of Apps
kubectl apply -f gitops/root-app.yaml
```

---

## 🔬 Lab Exercise: "Ping Across the Pond"

### Task
Verify VPC Peering.
1.  **Launch Pod US:** In `us-east-1` (IP 10.1.1.5).
2.  **Launch Pod EU:** In `eu-central-1` (IP 10.2.1.5).
3.  **Action:** `kubectl exec pod-us -- ping 10.2.1.5`.
4.  **Expected:** Ping succeeds (Latency ~80ms).
5.  **Troubleshooting:** If fails, check Route Tables.
    *   US Route Table must have: `10.2.0.0/16 -> pcx-xxxxxx` (Peering Connection).
    *   EU Route Table must have: `10.1.0.0/16 -> pcx-xxxxxx`.

---

## 📖 Advanced Theory: Split Horizon DNS
When Pod US wants to talk to Service EU, how does it resolve the name?
*   **CoreDNS StubDomains:** Configure CoreDNS to forward `*.eu.svc.cluster.local` to the CoreDNS IP of the EU cluster.
*   **ExternalDNS:** Writes A-records to Route53 (`service-eu.titan.internal`). This is more robust than StubDomains.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Modules:** Use Terraform Modules. Do not copy paste resource blocks.
2.  **CIDR Management:** Plan your IP space carefully. `10.1.0.0/16` and `10.2.0.0/16` do not overlap. If they overlap, Peering fails.
3.  **State Separation:** Keep Control Plane state separate from Data Plane state. If you destroy the GPU cluster, the Control Plane should survive.

### API Summary
```bash
terraform apply -target=module.vpc_us
```

---

**Day 198 Complete** ✅

*Next: Day 199 - Capstone Part 3 - The Training Pipeline.*
