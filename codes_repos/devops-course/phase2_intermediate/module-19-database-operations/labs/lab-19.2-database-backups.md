# Lab 19.2: Database Backups

## Objective
Implement database backup and restore strategies.

## Learning Objectives
- Configure automated backups
- Create manual snapshots
- Restore from backups
- Implement point-in-time recovery

---

## Automated Backups

```bash
# Enable automated backups
aws rds modify-db-instance \
  --db-instance-identifier mydb \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --apply-immediately
```

## Manual Snapshots

```bash
# Create snapshot
aws rds create-db-snapshot \
  --db-instance-identifier mydb \
  --db-snapshot-identifier mydb-snapshot-2024

# List snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier mydb
```

## Restore from Snapshot

```bash
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier mydb-restored \
  --db-snapshot-identifier mydb-snapshot-2024
```

## Point-in-Time Recovery

```bash
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier mydb \
  --target-db-instance-identifier mydb-pitr \
  --restore-time 2024-01-15T10:30:00Z
```

## Success Criteria
✅ Automated backups configured  
✅ Manual snapshot created  
✅ Restored from snapshot  
✅ PITR tested  

**Time:** 35 min
