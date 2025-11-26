# Day 19: Terraform & Infrastructure as Code

## Table of Contents
1. [Infrastructure as Code Fundamentals](#1-infrastructure-as-code-fundamentals)
2. [Terraform Basics](#2-terraform-basics)
3. [Providers & Resources](#3-providers--resources)
4. [State Management](#4-state-management)
5. [Variables & Outputs](#5-variables--outputs)
6. [Modules](#6-modules)
7. [Workspaces](#7-workspaces)
8. [Best Practices](#8-best-practices)
9. [AWS Infrastructure Example](#9-aws-infrastructure-example)
10. [Summary](#10-summary)

---

## 1. Infrastructure as Code Fundamentals

### 1.1 Traditional vs IaC

**Traditional** (ClickOps):
```
1. Log into AWS Console
2. Click "Launch EC2 Instance"
3. Choose AMI, instance type, networking...
4. Repeat for staging, production

Problems:
- Manual, error-prone
- Not versioned
- Hard to reproduce
```

**Infrastructure as Code**:
```hcl
# main.tf
resource "aws_instance" "web" {
  ami           = "ami-12345"
  instance_type = "t3.micro"
}

# Deploy everywhere consistently
terraform apply  # Creates infrastructure
```

**Benefits**:
- ✅ **Versioned**: Git history of infrastructure changes
- ✅ **Reproducible**: Same code → same infrastructure
- ✅ **Automated**: CI/CD deploys infrastructure
- ✅ **Documentation**: Code IS the documentation

---

## 2. Terraform Basics

### 2.1 Installation

```bash
# macOS
brew install terraform

# Verify
terraform version
```

### 2.2 Basic Workflow

```bash
# 1. Write configuration
# main.tf created

# 2. Initialize (download providers)
terraform init

# 3. Preview changes
terraform plan

# 4. Apply changes
terraform apply

# 5. Destroy resources
terraform destroy
```

### 2.3 First Example

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  
  tags = {
    Name = "HelloWorld"
  }
}
```

```bash
terraform init
terraform plan  # Shows: will create aws_instance.web
terraform apply  # Prompts for confirmation
```

---

## 3. Providers & Resources

### 3.1 Providers

**Provider**: Plugin to interact with APIs (AWS, GCP, Azure, K8s).

```hcl
provider "aws" {
  region = "us-west-2"
}

provider "google" {
  project = "my-project"
  region  = "us-central1"
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}
```

### 3.2 Resources

```hcl
resource "aws_s3_bucket" "data" {
  bucket = "my-unique-bucket-name"
}

resource "aws_db_instance" "postgres" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  username             = "admin"
  password             = var.db_password  # From variable
}
```

### 3.3 Data Sources

**Query existing resources**:
```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical
  
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "web" {
  ami = data.aws_ami.ubuntu.id  # Use queried AMI
  instance_type = "t3.micro"
}
```

---

## 4. State Management

### 4.1 What is State?

**State file** (`terraform.tfstate`): Mapping of resources to real-world  infrastructure.

```json
{
  "version": 4,
  "resources": [
    {
      "type": "aws_instance",
      "name": "web",
      "instances": [{
        "attributes": {
          "id": "i-abc123",
          "public_ip": "54.123.45.67"
        }
      }]
    }
  ]
}
```

**Why important**: Terraform uses state to know what exists.

### 4.2 Remote State (S3)

**Problem**: Local state file → can't collaborate.

**Solution**: Store in S3.

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"  # Locking
    encrypt        = true
  }
}
```

**Setup**:
```bash
# Create S3 bucket + DynamoDB table (one time)
aws s3 mb s3://my-terraform-state
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### 4.3 State Locking

**Prevents concurrent applies**:
```
User A: terraform apply (acquires lock)
User B: terraform apply (waits for lock)
```

---

## 5. Variables & Outputs

### 5.1 Input Variables

```hcl
# variables.tf
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true  # Won't show in logs
}
```

**Use**:
```hcl
# main.tf
resource "aws_instance" "web" {
  instance_type = var.instance_type
  tags = {
    Environment = var.environment
  }
}
```

**Provide values**:
```bash
# Method 1: CLI
terraform apply -var="environment=production"

# Method 2: File
# terraform.tfvars
environment = "production"
instance_type = "t3.small"

terraform apply  # Auto-loads *.tfvars
```

### 5.2 Outputs

```hcl
# outputs.tf
output "instance_ip" {
  description = "Public IP of web server"
  value       = aws_instance.web.public_ip
}

output "db_endpoint" {
  description = "Database connection endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}
```

**View outputs**:
```bash
terraform apply
# Outputs:
# instance_ip = "54.123.45.67"

terraform output instance_ip  # Get specific output
```

---

## 6. Modules

### 6.1 What are Modules?

**Module**: Reusable Terraform configuration.

**Directory structure**:
```
modules/
  vpc/
    main.tf
    variables.tf
    outputs.tf
  ec2/
    main.tf
    variables.tf
    outputs.tf
```

### 6.2 Creating a Module

```hcl
# modules/ec2/main.tf
variable "instance_type" {
  type = string
}

variable "ami_id" {
  type = string
}

resource "aws_instance" "this" {
  ami           = var.ami_id
  instance_type = var.instance_type
}

output "instance_id" {
  value = aws_instance.this.id
}
```

### 6.3 Using a Module

```hcl
# main.tf
module "web_server" {
  source = "./modules/ec2"
  
  instance_type = "t3.micro"
  ami_id        = "ami-12345"
}

output "web_server_id" {
  value = module.web_server.instance_id
}
```

### 6.4 Public Modules

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "my-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
}
```

---

## 7. Workspaces

### 7.1 What are Workspaces?

**Workspaces**: Manage multiple environments (dev, staging, prod) with same code.

```bash
# Create dev workspace
terraform workspace new dev
terraform workspace new staging
terraform workspace new prod

# List workspaces
terraform workspace list
  default
* dev
  staging
  prod

# Switch workspace
terraform workspace select prod
```

### 7.2 Using Workspaces

```hcl
resource "aws_instance" "web" {
  instance_type = terraform.workspace == "prod" ? "t3.large" : "t3.micro"
  
  tags = {
    Environment = terraform.workspace
  }
}
```

**Result**:
```
dev workspace   → t3.micro instance (tagged "dev")
prod workspace  → t3.large instance (tagged "prod")
```

---

## 8. Best Practices

### 8.1 Directory Structure

```
terraform/
  modules/
    vpc/
    ec2/
    rds/
  environments/
    dev/
      main.tf
      variables.tf
      terraform.tfvars
    prod/
      main.tf
      variables.tf
      terraform.tfvars
```

### 8.2 Use Remote State

❌ **Bad**: Local state file
✅ **Good**: S3 backend with locking

### 8.3 Version Providers

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # Pin major version
    }
  }
}
```

### 8.4 Use Variables

❌ **Bad**: Hardcoded values
```hcl
resource "aws_instance" "web" {
  instance_type = "t3.micro"  # Hardcoded
}
```

✅ **Good**: Variables
```hcl
resource "aws_instance" "web" {
  instance_type = var.instance_type
}
```

### 8.5 Use terraform fmt & validate

```bash
terraform fmt  # Format code
terraform validate  # Check syntax
```

---

## 9. AWS Infrastructure Example

### 9.1 Full Stack

```hcl
# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "my-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
}

# Security Group
resource "aws_security_group" "web" {
  vpc_id = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instance
resource "aws_instance" "web" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = module.vpc.public_subnets[0]
  vpc_security_group_ids = [aws_security_group.web.id]
  
  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y nginx
              systemctl start nginx
              EOF
  
  tags = {
    Name = "web-server"
  }
}

# RDS Database
resource "aws_db_instance" "postgres" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  db_name              = "myapp"
  username             = "admin"
  password             = var.db_password
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  
  skip_final_snapshot = true
}

# Outputs
output "web_server_ip" {
  value = aws_instance.web.public_ip
}

output "db_endpoint" {
  value     = aws_db_instance.postgres.endpoint
  sensitive = true
}
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **IaC** - Version infrastructure like code
2. ✅ **Terraform Workflow** - init → plan → apply
3. ✅ **Providers** - Plugin system for clouds
4. ✅ **State** - Tracking real-world resources
5. ✅ **Remote State** - S3 + DynamoDB locking
6. ✅ **Modules** - Reusable components
7. ✅ **Workspaces** - Multi-environment management

### 10.2 Terraform vs Alternatives

| Tool | Approach | Best For |
|:-----|:---------|:---------|
| **Terraform** | Declarative, multi-cloud | Any cloud, modular |
| **AWS CloudFormation** | Declarative, AWS-only | AWS-only stacks |
| **Pulumi** | Imperative (Python/Go/TS) | Developers preferring code |
| **Ansible** | Configuration management | Server config, not infra |

### 10.3 Tomorrow (Day 20): GitOps & Cloud Deployment

- **GitOps principles**: Git as source of truth
- **ArgoCD**: Kubernetes GitOps
- **Flux**: Alternative GitOps tool
- **CI/CD pipelines**: GitHub Actions, GitLab CI
- **Blue/Green deployments**: Zero downtime
- **Canary deployments**: Gradual rollout

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | Terraform & IaC mastered ✅
