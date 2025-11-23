# Lab 14.5: Terraform State Locking

## Objective
Implement state locking to prevent concurrent modifications.

## Learning Objectives
- Configure remote state with locking
- Use DynamoDB for state locks
- Handle lock conflicts
- Implement state encryption

---

## S3 Backend with Locking

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

## Create DynamoDB Table

```bash
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

## Testing Locks

```bash
# Terminal 1
terraform apply

# Terminal 2 (while apply running)
terraform apply
# Error: state locked
```

## Force Unlock

```bash
terraform force-unlock <LOCK_ID>
```

## Success Criteria
✅ Remote state configured  
✅ DynamoDB locking working  
✅ Tested concurrent access  
✅ State encrypted  

**Time:** 35 min
