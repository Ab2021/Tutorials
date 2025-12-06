# Day 139: The Blueprint: Infrastructure as Code (IaC)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** Clicking around in the AWS Console is acceptable for a hobby. It is unacceptable for a platform. **Terraform** allows you to provision specific GPU instance types across 3 regions with a single command.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** HCL (HashiCorp Configuration Language) to provision an S3 Bucket and EC2 GPU Instance.
2.  **Manage** Terraform State (Remote State in S3 with Locking).
3.  **Modularize** infrastructure (Create a reusable "ML Workspace" module).
4.  **Destroy** the entire stack to save costs when not iterating.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- AWS Account (or LocalStack).

### Software Environment
- `pip install localstack`.
- Terraform CLI installed.

---

## 📖 Theoretical Foundation

### 1. The Declarative Model
*   **Procedural (Bash):** "Create instance. Wait. Attach Volume. Wait."
*   **Declarative (Terraform):** "I want an instance with a volume attached. Figure out how."
*   **Graph:** Terraform builds a Dependency Graph. It knows it must create the VPC before the Subnet, and the Subnet before the Instance.

### 2. State
Terraform stores the mapping between "My Code" and "Real World ID (i-123456)" in a `terraform.tfstate` file.
*   **Team:** This file must be shared (S3) and locked (DynamoDB) to prevent race conditions.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: The ML Module

We define a reusable module for a standard Data Scientist environment.

#### 📁 `infra/modules/ml_workspace/main.tf`
```hcl
variable "environment" {
  description = "dev/stage/prod"
  type        = string
}

variable "instance_type" {
  default = "g4dn.xlarge" # GPU Instance
}

# 1. S3 Bucket for Data
resource "aws_s3_bucket" "data_bucket" {
  bucket = "ml-platform-${var.environment}-data-${random_id.suffix.hex}"
  
  tags = {
    Environment = var.environment
    Team        = "AI"
  }
}

# 2. IAM Role for EC2 to access S3
resource "aws_iam_role" "ml_role" {
  name = "ml_role_${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3_attach" {
  role       = aws_iam_role.ml_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_instance_profile" "ml_profile" {
  name = "ml_profile_${var.environment}"
  role = aws_iam_role.ml_role.name
}

# 3. GPU Instance (DLAMI)
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }
}

resource "aws_instance" "notebook_server" {
  ami           = data.aws_ami.dlami.id
  instance_type = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.ml_profile.name

  tags = {
    Name = "ML-Notebook-${var.environment}"
  }
}

# 4. Random Suffix to avoid name collision
resource "random_id" "suffix" {
  byte_length = 4
}
```

### 👨‍💻 Infrastructure: The Root Configuration

#### 📁 `infra/main.tf`
```hcl
provider "aws" {
  region = "us-east-1"
}

# Backend (State Storage)
terraform {
  backend "s3" {
    bucket         = "my-terraform-state-bucket"
    key            = "ml-platform/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
  }
}

# Instantiate Dev
module "dev_workspace" {
  source = "./modules/ml_workspace"
  environment = "dev"
  instance_type = "g4dn.xlarge"
}

# Instantiate Prod (Bigger GPU)
module "prod_training" {
  source = "./modules/ml_workspace"
  environment = "prod"
  instance_type = "p3.2xlarge"
}

output "dev_instance_id" {
  value = module.dev_workspace.instance_id
}
```

### 👨‍💻 Core Implementation: Provisioning Pipeline (Bash)

How CI runs terraform.

#### 📁 `scripts/provision.sh`
```bash
#!/bin/bash
set -e

# 1. Format Check
terraform fmt -check

# 2. Validate Syntax
terraform validate

# 3. Plan (Dry Run)
# Saves the plan to a file so Apply executes exactly what was verified
terraform plan -out=tfplan

# 4. Apply (On Merge)
if [ "$1" == "--apply" ]; then
    terraform apply -auto-approve tfplan
fi
```

---

## 🔬 Lab Exercise: "Drift Detection"

### Task
Simulate Configuration Drift.
1.  Run `terraform apply`. The instance is created.
2.  Go to AWS Console. Manually add an "Allow All traffic" Security Group rule to the instance.
3.  Run `terraform plan`.
4.  **Observation:** Terraform detects the change. "Plan: 0 to add, 1 to change, 0 to destroy." It proposes *removing* the manual rule because it's not in the code.
5.  **Insight:** IaC enforces consistency. If you want the rule, add it to `.tf` files.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Immutable infrastructure:** Don't SSH into instances to update Python versions. Update the AMI ID in Terraform and replace the instance.
2.  **Cost Control:** A `terraform destroy` command at 6 PM on Friday saves 60 hours of GPU cost per week.
3.  **Modules:** Stop copying-pasting VPC config. Write a module.

### API Summary
```hcl
resource "type" "name" { ... }
variable "name" { ... }
output "name" { ... }
```

---

**Day 139 Complete** ✅

*Next: Day 140 - Week 20 Review & Project - The End-to-End MLOps Pipeline.*
