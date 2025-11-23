# Lab 10.10: AWS Best Practices

## Objective
Implement AWS best practices for security, cost, and reliability.

## Learning Objectives
- Follow Well-Architected Framework
- Implement cost optimization
- Ensure security best practices
- Design for reliability

---

## Security Best Practices

```bash
# Enable MFA for root
# Use IAM roles instead of access keys
# Enable CloudTrail
aws cloudtrail create-trail \
  --name my-trail \
  --s3-bucket-name my-cloudtrail-bucket

# Enable GuardDuty
aws guardduty create-detector --enable

# Encrypt EBS volumes
aws ec2 create-volume \
  --size 100 \
  --encrypted \
  --kms-key-id arn:aws:kms:...
```

## Cost Optimization

```bash
# Use Reserved Instances
# Right-size instances
# Delete unused resources
# Use S3 lifecycle policies
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-bucket \
  --lifecycle-configuration file://lifecycle.json
```

## Reliability

```bash
# Multi-AZ deployments
# Auto Scaling
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name my-asg \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3

# Backup automation
aws backup create-backup-plan \
  --backup-plan file://backup-plan.json
```

## Success Criteria
✅ Security controls implemented  
✅ Cost optimization applied  
✅ High availability configured  

**Time:** 40 min
