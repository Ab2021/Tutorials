# Lab 19.1: RDS Setup

## Objective
Deploy and manage AWS RDS databases.

## Learning Objectives
- Create RDS instances
- Configure security groups
- Set up read replicas
- Implement automated backups

---

## Create RDS Instance

```bash
aws rds create-db-instance \
  --db-instance-identifier mydb \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password MyPassword123 \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-12345 \
  --db-subnet-group-name my-subnet-group \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00"
```

## Connect to RDS

```bash
psql -h mydb.abc123.us-east-1.rds.amazonaws.com -U admin -d postgres
```

## Create Read Replica

```bash
aws rds create-db-instance-read-replica \
  --db-instance-identifier mydb-replica \
  --source-db-instance-identifier mydb
```

## Success Criteria
✅ RDS instance created  
✅ Connected successfully  
✅ Read replica configured  
✅ Backups automated  

**Time:** 40 min
