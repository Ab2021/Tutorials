# Lab 14.6: Terraform Import

## Objective
Import existing infrastructure into Terraform state.

## Learning Objectives
- Import existing AWS resources
- Generate configuration from state
- Manage legacy infrastructure
- Migrate to IaC

---

## Import EC2 Instance

```bash
# Create resource block
cat > main.tf << 'EOF'
resource "aws_instance" "imported" {
  # Configuration will be added after import
}
EOF

# Import existing instance
terraform import aws_instance.imported i-1234567890abcdef0

# Generate configuration
terraform show -no-color > imported.tf
```

## Import Multiple Resources

```bash
# Import VPC
terraform import aws_vpc.main vpc-12345

# Import subnet
terraform import aws_subnet.public subnet-67890

# Import security group
terraform import aws_security_group.web sg-abcde
```

## Bulk Import Script

```bash
#!/bin/bash
# import-resources.sh

resources=(
  "aws_instance.web:i-abc123"
  "aws_instance.db:i-def456"
  "aws_s3_bucket.data:my-bucket"
)

for resource in "${resources[@]}"; do
  IFS=':' read -r tf_resource aws_id <<< "$resource"
  terraform import "$tf_resource" "$aws_id"
done
```

## Success Criteria
✅ Imported existing resources  
✅ Generated configuration  
✅ State matches reality  
✅ Can manage imported resources  

**Time:** 40 min
