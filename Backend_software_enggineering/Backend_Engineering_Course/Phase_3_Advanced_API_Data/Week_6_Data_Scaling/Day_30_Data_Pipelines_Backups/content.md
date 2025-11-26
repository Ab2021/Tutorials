# Day 30: Data Pipelines & Backups

## 1. Schema Migrations

You can't just `ALTER TABLE` in production manually.
*   **The Tool**: Alembic (Python), Flyway (Java/SQL).
*   **The Concept**: Version control for your database schema.
    *   `V1__create_users.sql`
    *   `V2__add_email.sql`
*   **CI/CD**: The pipeline runs `flyway migrate` before deploying the app code.

### 1.1 Zero Downtime Migrations
How to rename a column without downtime?
1.  **Add** new column `new_name`.
2.  **Dual Write** to both `old_name` and `new_name`.
3.  **Backfill** old data to `new_name`.
4.  **Switch Reads** to `new_name`.
5.  **Drop** `old_name`.

---

## 2. Backups & Disaster Recovery (DR)

"RAID is not a backup." "Replication is not a backup."
If you `DROP TABLE` on Master, it replicates to Slave instantly. You lose everything.

### 2.1 Types
1.  **Full Backup**: `pg_dump`. Slow, heavy.
2.  **Incremental/Differential**: Only changes since last backup.
3.  **PITR (Point-in-Time Recovery)**: The Holy Grail.
    *   Save the Base Backup + All WAL (Write Ahead Logs).
    *   You can restore to `2023-10-27 14:32:01` exactly.

---

## 3. Data Pipelines (ETL vs ELT)

Moving data from OLTP (Postgres) to OLAP (Snowflake/BigQuery) for analytics.

### 3.1 ETL (Extract, Transform, Load)
*   Extract from Postgres.
*   Transform (Clean, Aggregate) in Python/Spark.
*   Load into Warehouse.
*   *Pros*: Clean data enters warehouse.
*   *Cons*: Slow, rigid.

### 3.2 ELT (Extract, Load, Transform)
*   Extract from Postgres.
*   Load raw JSON into Warehouse.
*   Transform using SQL inside the Warehouse (dbt).
*   *Pros*: Fast, flexible. The modern standard.

---

## 4. Summary

Today we protected our most valuable asset: Data.
*   **Migrations**: Safe schema changes.
*   **Backups**: Your insurance policy.
*   **Pipelines**: Turning data into insights.

**Phase 3 Wrap-Up**:
We have covered:
1.  Advanced API (REST, GraphQL, gRPC).
2.  SQL Performance (Indexes, Partitioning).
3.  NoSQL Patterns (Redis, Mongo).
4.  Vector DBs (AI).
5.  Scaling (Replication, Sharding).

**Next Week (Week 7)**: We enter Phase 4. **Security**. We will learn how to hack our own apps (OWASP) and how to secure them (OAuth2, MFA).
