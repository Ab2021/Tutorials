# Lab 14.9: Terraform Cloud

## Objective
Use Terraform Cloud for remote operations and collaboration.

## Learning Objectives
- Set up Terraform Cloud
- Configure remote backend
- Use remote execution
- Implement team workflows

---

## Terraform Cloud Setup

```hcl
terraform {
  cloud {
    organization = "my-org"
    
    workspaces {
      name = "production"
    }
  }
}
```

## Remote Execution

```bash
# Login
terraform login

# Initialize
terraform init

# Plan (runs remotely)
terraform plan

# Apply (runs remotely)
terraform apply
```

## Variables in Cloud

```hcl
variable "api_key" {
  type      = string
  sensitive = true
}

# Set in Terraform Cloud UI or:
terraform cloud workspace set-var \
  -workspace=production \
  -var="api_key=secret123" \
  -sensitive
```

## Success Criteria
✅ Terraform Cloud configured  
✅ Remote execution working  
✅ Team collaboration enabled  

**Time:** 40 min
