# Lab 14.10: IaC Best Practices

## Objective
Implement Infrastructure as Code best practices.

## Learning Objectives
- Structure IaC projects
- Use naming conventions
- Implement security
- Ensure maintainability

---

## Project Structure

```
terraform/
├── modules/
│   ├── vpc/
│   ├── compute/
│   └── database/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
├── .terraform.lock.hcl
└── README.md
```

## Best Practices

```hcl
# Use variables for everything
variable "environment" {}
variable "region" {}

# Use locals for computed values
locals {
  common_tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Use modules for reusability
module "vpc" {
  source = "../../modules/vpc"
  tags   = local.common_tags
}

# Use data sources
data "aws_ami" "latest" {
  most_recent = true
  owners      = ["amazon"]
}

# Use outputs
output "vpc_id" {
  value = module.vpc.id
}
```

## Security

```hcl
# Never commit secrets
# Use sensitive = true
variable "db_password" {
  type      = string
  sensitive = true
}

# Encrypt state
terraform {
  backend "s3" {
    encrypt = true
  }
}
```

## Success Criteria
✅ Project well-structured  
✅ Best practices followed  
✅ Security implemented  
✅ Code maintainable  

**Time:** 40 min
