# Lab 07.4: Terraform Variables and Outputs

## Objective
Use variables and outputs for flexible, reusable Terraform configurations.

## Learning Objectives
- Define input variables
- Use variable types and validation
- Create output values
- Use locals and data sources

---

## Input Variables

```hcl
# variables.tf
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_count" {
  description = "Number of instances"
  type        = number
  default     = 1
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
```

## Using Variables

```hcl
# main.tf
resource "aws_instance" "web" {
  count         = var.instance_count
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = var.environment == "prod" ? "t3.medium" : "t2.micro"
  
  tags = merge(
    var.tags,
    {
      Name        = "web-${var.environment}-${count.index}"
      Environment = var.environment
    }
  )
}
```

## Outputs

```hcl
# outputs.tf
output "instance_ids" {
  description = "IDs of created instances"
  value       = aws_instance.web[*].id
}

output "public_ips" {
  description = "Public IPs"
  value       = aws_instance.web[*].public_ip
}

output "connection_string" {
  description = "Connection string"
  value       = "ssh ec2-user@${aws_instance.web[0].public_ip}"
  sensitive   = false
}
```

## Locals

```hcl
locals {
  common_tags = {
    Project     = "MyApp"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
  
  instance_name = "${var.environment}-web-server"
}

resource "aws_instance" "web" {
  tags = local.common_tags
}
```

## Variable Files

```hcl
# terraform.tfvars
environment    = "prod"
instance_count = 3
tags = {
  Team = "Platform"
  Cost = "Engineering"
}
```

```bash
# Use specific var file
terraform apply -var-file="prod.tfvars"
```

## Success Criteria
✅ Variables defined with validation  
✅ Outputs working  
✅ Locals used for DRY code  
✅ Multiple environments supported  

**Time:** 40 min
