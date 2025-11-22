# Module 7: Infrastructure as Code - Introduction

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand Infrastructure as Code (IaC) principles and benefits
- Master Terraform fundamentals and workflow
- Learn CloudFormation basics for AWS
- Compare different IaC tools (Terraform, CloudFormation, Pulumi)
- Manage infrastructure state effectively
- Implement version control for infrastructure
- Apply IaC best practices

---

## 📖 Theoretical Concepts

### 7.1 What is Infrastructure as Code?

Infrastructure as Code (IaC) is the practice of managing and provisioning infrastructure through machine-readable definition files, rather than physical hardware configuration or interactive configuration tools.

#### Core Principles

**Declarative vs Imperative**
- **Declarative:** Define the desired end state (Terraform, CloudFormation)
- **Imperative:** Define the steps to reach the state (scripts, Ansible)

**Benefits of IaC:**
- ✅ **Version Control:** Track infrastructure changes like code
- ✅ **Reproducibility:** Create identical environments consistently
- ✅ **Automation:** Eliminate manual configuration errors
- ✅ **Documentation:** Infrastructure is self-documenting
- ✅ **Collaboration:** Team members can review and contribute
- ✅ **Testing:** Test infrastructure changes before applying
- ✅ **Disaster Recovery:** Rebuild infrastructure quickly

---

### 7.2 Terraform Fundamentals

Terraform is an open-source IaC tool by HashiCorp that supports multiple cloud providers.

#### Terraform Workflow

```
Write → Plan → Apply → Destroy
```

1. **Write:** Define infrastructure in `.tf` files
2. **Plan:** Preview changes before applying
3. **Apply:** Create/update infrastructure
4. **Destroy:** Remove infrastructure when needed

#### Terraform Language (HCL)

```hcl
# Provider configuration
provider "aws" {
  region = "us-east-1"
}

# Resource definition
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {
    Name = "WebServer"
    Environment = "Development"
  }
}

# Output values
output "instance_ip" {
  value = aws_instance.web_server.public_ip
}
```

#### Key Terraform Concepts

**Providers**
- Plugins that interact with cloud platforms
- Examples: AWS, Azure, GCP, Kubernetes

**Resources**
- Infrastructure components (EC2, S3, VPC)
- Defined with `resource` blocks

**Data Sources**
- Query existing infrastructure
- Use with `data` blocks

**Variables**
- Parameterize configurations
- Support different environments

**Outputs**
- Export values for use elsewhere
- Display important information

---

### 7.3 AWS CloudFormation

CloudFormation is AWS's native IaC service using JSON or YAML templates.

#### CloudFormation Template Structure

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Simple EC2 instance'

Parameters:
  InstanceType:
    Type: String
    Default: t2.micro
    AllowedValues:
      - t2.micro
      - t2.small
    Description: EC2 instance type

Resources:
  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: !Ref InstanceType
      Tags:
        - Key: Name
          Value: WebServer

Outputs:
  InstanceId:
    Description: Instance ID
    Value: !Ref WebServer
```

#### CloudFormation Features

**Stacks**
- Collection of AWS resources managed as a single unit
- Create, update, delete entire stacks

**Change Sets**
- Preview changes before applying
- Similar to Terraform plan

**Drift Detection**
- Identify manual changes to resources
- Ensure infrastructure matches template

---

### 7.4 IaC Tool Comparison

| Feature | Terraform | CloudFormation | Pulumi |
|---------|-----------|----------------|--------|
| **Language** | HCL | JSON/YAML | Real programming languages |
| **Cloud Support** | Multi-cloud | AWS only | Multi-cloud |
| **State Management** | External state file | AWS managed | Cloud-based |
| **Learning Curve** | Moderate | Moderate | Depends on language |
| **Community** | Large | AWS-focused | Growing |
| **Cost** | Free (Cloud paid) | Free | Free (Cloud paid) |
| **Best For** | Multi-cloud | AWS-native | Developers |

---

### 7.5 State Management

State is a critical concept in IaC that tracks the current infrastructure.

#### Terraform State

**Local State**
```hcl
# terraform.tfstate (auto-generated)
{
  "version": 4,
  "terraform_version": "1.5.0",
  "resources": [...]
}
```

**Remote State (Recommended)**
```hcl
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
    
    # State locking
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

**State Best Practices:**
- ✅ Use remote state for team collaboration
- ✅ Enable state locking to prevent conflicts
- ✅ Encrypt state files (contain sensitive data)
- ✅ Never edit state files manually
- ✅ Use workspaces for environment separation

---

### 7.6 Variables and Outputs

#### Terraform Variables

```hcl
# variables.tf
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "instance_count" {
  description = "Number of instances"
  type        = number
  default     = 1
}

variable "allowed_ips" {
  description = "Allowed IP addresses"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {
    Project = "MyApp"
  }
}
```

**Using Variables:**
```hcl
resource "aws_instance" "app" {
  count         = var.instance_count
  instance_type = "t2.micro"
  
  tags = merge(
    var.tags,
    {
      Environment = var.environment
    }
  )
}
```

**Variable Files:**
```hcl
# terraform.tfvars
environment    = "production"
instance_count = 3
allowed_ips    = ["10.0.0.0/8"]
```

---

### 7.7 Infrastructure Versioning

Treat infrastructure like application code:

**Git Workflow for IaC:**
```bash
# Feature branch for infrastructure changes
git checkout -b feature/add-load-balancer

# Make changes to .tf files
vim main.tf

# Test locally
terraform plan

# Commit changes
git add .
git commit -m "Add application load balancer"

# Create pull request
git push origin feature/add-load-balancer

# After review, merge to main
# CI/CD pipeline applies changes
```

**Directory Structure:**
```
infrastructure/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
├── modules/
│   ├── vpc/
│   ├── ec2/
│   └── rds/
└── README.md
```

---

### 7.8 Pulumi Introduction

Pulumi allows you to use real programming languages for IaC.

**Python Example:**
```python
import pulumi
import pulumi_aws as aws

# Create an EC2 instance
instance = aws.ec2.Instance('web-server',
    instance_type='t2.micro',
    ami='ami-0c55b159cbfafe1f0',
    tags={
        'Name': 'WebServer',
        'Environment': 'Development'
    }
)

# Export the instance's public IP
pulumi.export('public_ip', instance.public_ip)
```

**TypeScript Example:**
```typescript
import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

const instance = new aws.ec2.Instance("web-server", {
    instanceType: "t2.micro",
    ami: "ami-0c55b159cbfafe1f0",
    tags: {
        Name: "WebServer",
        Environment: "Development"
    }
});

export const publicIp = instance.publicIp;
```

---

## 🔧 Practical Examples

### Example 1: Simple Terraform Configuration

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.project_name}-public-subnet"
  }
}
```

### Example 2: CloudFormation Stack

```yaml
# stack.yaml
AWSTemplateFormatVersion: '2010-09-09'

Parameters:
  ProjectName:
    Type: String
    Default: MyProject

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      Tags:
        - Key: Name
          Value: !Sub '${ProjectName}-vpc'
  
  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${ProjectName}-public-subnet'

Outputs:
  VPCId:
    Value: !Ref VPC
    Export:
      Name: !Sub '${ProjectName}-VPC-ID'
```

---

## 🎯 Best Practices

### 1. Code Organization
- Separate environments (dev, staging, prod)
- Use modules for reusable components
- Keep configurations DRY (Don't Repeat Yourself)

### 2. State Management
- Always use remote state for teams
- Enable state locking
- Encrypt state files
- Regular state backups

### 3. Security
- Never commit secrets to version control
- Use secret management tools (AWS Secrets Manager, Vault)
- Implement least privilege access
- Enable encryption at rest and in transit

### 4. Testing
- Use `terraform plan` before apply
- Implement automated testing (Terratest)
- Test in non-production environments first
- Use CloudFormation change sets

### 5. Documentation
- Comment complex configurations
- Maintain README files
- Document variable purposes
- Keep architecture diagrams updated

---

## 📚 Additional Resources

### Official Documentation
- [Terraform Documentation](https://www.terraform.io/docs)
- [AWS CloudFormation User Guide](https://docs.aws.amazon.com/cloudformation/)
- [Pulumi Documentation](https://www.pulumi.com/docs/)

### Tutorials
- [HashiCorp Learn](https://learn.hashicorp.com/terraform)
- [AWS CloudFormation Workshops](https://catalog.workshops.aws/)

### Tools
- [Terraform Registry](https://registry.terraform.io/)
- [CloudFormation Designer](https://console.aws.amazon.com/cloudformation/designer)
- [Terragrunt](https://terragrunt.gruntwork.io/) - Terraform wrapper

---

## ⏭️ Next Steps

Complete all 10 labs in the `labs/` directory:

1. **Lab 7.1:** IaC Concepts and Benefits
2. **Lab 7.2:** Terraform Installation and Setup
3. **Lab 7.3:** Terraform Basics - First Resources
4. **Lab 7.4:** CloudFormation Introduction
5. **Lab 7.5:** Resource Creation and Management
6. **Lab 7.6:** State Management Deep Dive
7. **Lab 7.7:** Variables and Parameterization
8. **Lab 7.8:** IaC Tool Comparison Exercise
9. **Lab 7.9:** Terraform Plan and Apply Workflow
10. **Lab 7.10:** Infrastructure Versioning with Git

After completing the labs, move on to **Module 8: Configuration Management**.

---

**Master Infrastructure as Code!** 🚀
