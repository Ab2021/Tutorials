# Lab 20.3: Disaster Recovery

## Objective
Implement disaster recovery strategies.

## Learning Objectives
- Configure cross-region replication
- Implement backup strategies
- Test failover procedures
- Calculate RTO/RPO

---

## Cross-Region Replication

```bash
# Enable S3 replication
aws s3api put-bucket-replication \
  --bucket source-bucket \
  --replication-configuration '{
    "Role": "arn:aws:iam::123456789012:role/replication-role",
    "Rules": [{
      "Status": "Enabled",
      "Priority": 1,
      "Destination": {
        "Bucket": "arn:aws:s3:::destination-bucket",
        "ReplicationTime": {
          "Status": "Enabled",
          "Time": {"Minutes": 15}
        }
      }
    }]
  }'
```

## RDS Multi-Region

```bash
# Create read replica in different region
aws rds create-db-instance-read-replica \
  --db-instance-identifier mydb-dr \
  --source-db-instance-identifier arn:aws:rds:us-east-1:123:db:mydb \
  --region us-west-2

# Promote to standalone
aws rds promote-read-replica \
  --db-instance-identifier mydb-dr \
  --region us-west-2
```

## Route 53 Failover

```json
{
  "Changes": [{
    "Action": "CREATE",
    "ResourceRecordSet": {
      "Name": "app.example.com",
      "Type": "A",
      "SetIdentifier": "Primary",
      "Failover": "PRIMARY",
      "AliasTarget": {
        "HostedZoneId": "Z123",
        "DNSName": "primary-lb.us-east-1.elb.amazonaws.com",
        "EvaluateTargetHealth": true
      }
    }
  }]
}
```

## Success Criteria
✅ Cross-region replication configured  
✅ DR site ready  
✅ Failover tested  
✅ RTO/RPO documented  

**Time:** 50 min
