# Lab 14.2: Terraform Module Composition

## Objective
Create reusable Terraform modules and compose them into complex infrastructure.

## Learning Objectives
- Create custom Terraform modules
- Use module inputs and outputs
- Compose modules for multi-tier architecture
- Version and publish modules

---

## Creating a Module

```hcl
# modules/vpc/main.tf
variable "cidr_block" {
  type = string
}

variable "name" {
  type = string
}

resource "aws_vpc" "main" {
  cidr_block = var.cidr_block
  tags = {
    Name = var.name
  }
}

output "vpc_id" {
  value = aws_vpc.main.id
}
```

## Using the Module

```hcl
# main.tf
module "vpc" {
  source = "./modules/vpc"
  
  cidr_block = "10.0.0.0/16"
  name       = "production-vpc"
}

output "vpc_id" {
  value = module.vpc.vpc_id
}
```

## Module Registry

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "my-vpc"
  cidr = "10.0.0.0/16"
}
```

## Success Criteria
✅ Created custom module  
✅ Used module in root configuration  
✅ Passed variables and outputs  
✅ Used public registry module  

**Time:** 40 min
