# Lab 10.7: AWS CLI Advanced

## Objective
Master AWS CLI for automation and scripting.

## Learning Objectives
- Use CLI filters and queries
- Automate with scripts
- Use profiles and credentials
- Implement pagination

---

## Queries and Filters

```bash
# Query specific fields
aws ec2 describe-instances \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
  --output table

# Filter by tag
aws ec2 describe-instances \
  --filters "Name=tag:Environment,Values=production" \
  --query 'Reservations[*].Instances[*].InstanceId'

# JMESPath queries
aws ec2 describe-instances \
  --query 'Reservations[].Instances[?State.Name==`running`].InstanceId'
```

## Automation Script

```bash
#!/bin/bash
# backup-instances.sh

# Get all running instances
INSTANCES=$(aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].InstanceId' \
  --output text)

# Create AMI for each
for instance in $INSTANCES; do
  aws ec2 create-image \
    --instance-id $instance \
    --name "backup-$instance-$(date +%Y%m%d)" \
    --no-reboot
done
```

## Profiles

```bash
# Configure profile
aws configure --profile production
AWS Access Key ID: xxx
AWS Secret Access Key: xxx
Default region: us-east-1

# Use profile
aws s3 ls --profile production
```

## Success Criteria
✅ CLI queries working  
✅ Automation scripts functional  
✅ Profiles configured  

**Time:** 40 min
