# Lab 14.8: Terraform Workspaces

## Objective
Use Terraform workspaces for environment management.

## Learning Objectives
- Create and switch workspaces
- Use workspace-specific variables
- Manage multiple environments
- Understand workspace limitations

---

## Workspace Commands

```bash
# List workspaces
terraform workspace list

# Create workspace
terraform workspace new staging

# Switch workspace
terraform workspace select production

# Show current
terraform workspace show

# Delete workspace
terraform workspace delete staging
```

## Workspace-Specific Config

```hcl
locals {
  env = terraform.workspace
  
  instance_type = {
    dev     = "t2.micro"
    staging = "t2.small"
    prod    = "t2.medium"
  }
}

resource "aws_instance" "web" {
  instance_type = local.instance_type[local.env]
  
  tags = {
    Environment = local.env
  }
}
```

## Remote State with Workspaces

```hcl
terraform {
  backend "s3" {
    bucket = "terraform-state"
    key    = "env/${terraform.workspace}/terraform.tfstate"
    region = "us-east-1"
  }
}
```

## Success Criteria
✅ Workspaces created  
✅ Environment-specific configs  
✅ State isolated per workspace  

**Time:** 35 min
