# Database Operations (DBOps)

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Database Operations, including:
- **Reliability**: Designing for High Availability (HA) and Disaster Recovery (DR).
- **Backups**: Understanding RPO/RTO and implementing Point-in-Time Recovery.
- **Scaling**: Using Read Replicas and Connection Pooling to handle load.
- **Migrations**: Managing schema changes safely with tools like **Flyway**.
- **Performance**: analyzing query plans and optimizing indexes.

---

## 📖 Theoretical Concepts

### 1. High Availability & Disaster Recovery

- **High Availability (HA)**: Minimizing downtime during normal operations (e.g., Multi-AZ).
- **Disaster Recovery (DR)**: Recovering from catastrophic failure (e.g., Region outage).
- **RPO (Recovery Point Objective)**: How much data can you lose? (e.g., "5 minutes").
- **RTO (Recovery Time Objective)**: How long until you are back online? (e.g., "1 hour").

### 2. Replication Strategies

- **Synchronous**: Write to Primary, wait for Replica to confirm. Zero data loss. Slower writes.
- **Asynchronous**: Write to Primary, return success immediately. Replica catches up later. Fast writes. Potential data loss on failover.
- **Read Replicas**: Offload `SELECT` queries to replicas. Scale reads horizontally.

### 3. Schema Migrations

Changing the database structure (DDL) is risky.
- **Versioning**: Treat schema changes like code. Version 1 -> Version 2.
- **Tools**: Flyway, Liquibase, Alembic.
- **Zero Downtime**:
  1.  Add column (nullable).
  2.  Write to both old and new columns.
  3.  Backfill data.
  4.  Switch reads to new column.
  5.  Remove old column.

### 4. Performance Tuning

- **Indexing**: B-Tree indexes speed up lookups but slow down writes.
- **Connection Pooling**: Opening a DB connection is expensive. Reuse them. (Tool: PgBouncer).
- **N+1 Problem**: Fetching a list of items and then running a query for *each* item. Use `JOIN`s instead.

---

## 🔧 Practical Examples

### Backup Strategy (PostgreSQL)

```bash
# Dump entire DB
pg_dump -h localhost -U myuser mydb > backup.sql

# Restore
psql -h localhost -U myuser mydb < backup.sql
```

### Connection Pooling (PgBouncer)

```ini
[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = users.txt
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

### Flyway Migration (`V1__Create_Users.sql`)

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE
);
```

---

## 🎯 Hands-on Labs

- [Lab 19.1: Database Backup & Restore (PostgreSQL)](./labs/lab-19.1-db-backup.md)
- [Lab 19.2: Database Replication (Master-Slave)](./labs/lab-19.2-db-replication.md)
- [Lab 19.3: Point In Time Recovery](./labs/lab-19.3-point-in-time-recovery.md)
- [Lab 19.4: Read Replicas](./labs/lab-19.4-read-replicas.md)
- [Lab 19.5: Database Migrations](./labs/lab-19.5-database-migrations.md)
- [Lab 19.6: Connection Pooling](./labs/lab-19.6-connection-pooling.md)
- [Lab 19.7: Database Monitoring](./labs/lab-19.7-database-monitoring.md)
- [Lab 19.8: Performance Tuning](./labs/lab-19.8-performance-tuning.md)
- [Lab 19.9: Disaster Recovery](./labs/lab-19.9-disaster-recovery.md)
- [Lab 19.10: Database Security](./labs/lab-19.10-database-security.md)

---

## 📚 Additional Resources

### Official Documentation
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [AWS RDS User Guide](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html)

### Tools
- [Flyway](https://flywaydb.org/)
- [PgBouncer](https://www.pgbouncer.org/)

---

## 🔑 Key Takeaways

1.  **Backups are Useless**: ...unless you test **Restores**.
2.  **Don't run DBs in K8s**: Unless you have a dedicated DBA team. Managed services (RDS) are usually better.
3.  **Pool Connections**: Database connections are a finite resource.
4.  **Monitor Slow Queries**: Use `pg_stat_statements` to find queries taking > 1s.

---

## ⏭️ Next Steps

1.  Complete the labs to master database resilience.
2.  Proceed to **[Module 20: Cloud Architecture Patterns](../module-20-cloud-architecture-patterns/README.md)** to design scalable systems.
