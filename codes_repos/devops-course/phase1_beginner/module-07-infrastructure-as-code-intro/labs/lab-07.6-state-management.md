# Lab 07.5: Terraform State Management

## Objective
Understand and manage Terraform state effectively.

## Learning Objectives
- Understand state file structure
- Use remote state
- Implement state locking
- Perform state operations

---

## Remote State (S3)

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

## State Commands

```bash
# List resources in state
terraform state list

# Show specific resource
terraform state show aws_instance.web

# Move resource
terraform state mv aws_instance.web aws_instance.web_server

# Remove from state (doesn't delete resource)
terraform state rm aws_instance.old

# Pull current state
terraform state pull > terraform.tfstate.backup
```

## Import Existing Resources

```bash
# Import EC2 instance
terraform import aws_instance.web i-1234567890abcdef0

# Import S3 bucket
terraform import aws_s3_bucket.data my-bucket-name
```

## State Locking

```bash
# Create DynamoDB table for locking
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

## Workspace Management

```bash
# Create workspace
terraform workspace new staging

# List workspaces
terraform workspace list

# Switch workspace
terraform workspace select prod

# Show current workspace
terraform workspace show
```

## Success Criteria
✅ Remote state configured  
✅ State locking working  
✅ State operations performed  
✅ Workspaces used  

**Time:** 45 min
