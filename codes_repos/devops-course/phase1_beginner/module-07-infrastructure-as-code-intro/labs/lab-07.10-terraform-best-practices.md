# Lab 07.10: Terraform Best Practices

## Objective
Implement Terraform best practices for production use.

## Learning Objectives
- Structure Terraform projects
- Use naming conventions
- Implement security best practices
- Optimize performance

---

## Project Structure

```
terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
├── modules/
│   ├── vpc/
│   ├── compute/
│   └── database/
└── global/
    └── s3/
```

## Naming Conventions

```hcl
# Resources: type-environment-name
resource "aws_instance" "web_prod_01" {}

# Variables: descriptive_snake_case
variable "instance_count" {}

# Outputs: resource_attribute
output "vpc_id" {}
```

## Security Best Practices

```hcl
# Never commit secrets
variable "db_password" {
  type      = string
  sensitive = true
}

# Use data sources for sensitive data
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "prod/db/password"
}

# Enable encryption
resource "aws_s3_bucket" "data" {
  bucket = "my-data"
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}
```

## Performance Optimization

```hcl
# Use -target for specific resources
terraform apply -target=aws_instance.web

# Parallelize operations
terraform apply -parallelism=20

# Use depends_on sparingly
resource "aws_instance" "web" {
  depends_on = [aws_security_group.web]
}
```

## Success Criteria
✅ Project well-structured  
✅ Naming conventions followed  
✅ Security best practices implemented  
✅ Performance optimized  

**Time:** 40 min
