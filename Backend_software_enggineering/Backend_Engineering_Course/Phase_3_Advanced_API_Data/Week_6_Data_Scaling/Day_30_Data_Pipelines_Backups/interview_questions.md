# Day 30: Interview Questions & Answers

## Conceptual Questions

### Q1: What is "Point-in-Time Recovery" (PITR) and how does it work?
**Answer:**
*   **Mechanism**: You take a periodic Full Backup (e.g., weekly). You also archive all WAL (Write Ahead Log) files continuously (e.g., to S3).
*   **Recovery**: To restore to `Tuesday 10:00 AM`:
    1.  Restore the Sunday Full Backup.
    2.  Replay WAL files until `Tuesday 10:00 AM`.
*   **Benefit**: Granular recovery from accidental deletes.

### Q2: Explain "Change Data Capture" (CDC).
**Answer:**
*   **Definition**: A pattern to identify and capture changes made to data in a database.
*   **Method**: Instead of polling (`SELECT * FROM users WHERE updated_at > last_check`), CDC reads the DB's Transaction Log (WAL/Binlog).
*   **Tools**: Debezium, Kafka Connect.
*   **Use Case**: Streaming DB changes to a Search Index (Elasticsearch) or Cache (Redis) in real-time.

### Q3: How do you handle PII (Personally Identifiable Information) in backups?
**Answer:**
*   **GDPR/CCPA**: You must be able to "Forget" a user. If their data is in a backup from 2 years ago, that's a problem.
*   **Crypto-Shredding**: Encrypt PII with a per-user key. Store keys separately. To "delete" the user, just delete their key. The backup data becomes garbage.
*   **Sanitization**: Scrub PII from backups used for Dev/Staging environments.

---

## Scenario-Based Questions

### Q4: You deployed a migration that locked the `users` table for 10 minutes, causing an outage. What happened and how to prevent it?
**Answer:**
*   **Cause**: You probably ran `ALTER TABLE users ADD COLUMN default 'value'`. In older DB versions, this rewrites the whole table.
*   **Prevention**:
    1.  **Test**: Run migrations on a copy of Prod data first.
    2.  **Non-Blocking**: Use `CONCURRENTLY` (e.g., `CREATE INDEX CONCURRENTLY`).
    3.  **Batching**: If backfilling data, do it in small batches, not one transaction.

### Q5: Your Data Warehouse (Snowflake) is costing too much. Why?
**Answer:**
*   **Compute vs Storage**: Storage is cheap (S3). Compute is expensive.
*   **Bad Queries**: Analysts running `SELECT *` on petabyte tables.
*   **Fix**:
    *   **Partitioning/Clustering**: Organize data so queries scan less.
    *   **Materialized Views**: Pre-calculate common aggregations.
    *   **Resource Monitors**: Kill queries that run > 1 hour.

---

## Behavioral / Role-Specific Questions

### Q6: A developer manually fixed a schema drift in Production without updating the migration script. What do you do?
**Answer:**
*   **Immediate**: Capture the current state (`pg_dump -s`).
*   **Fix**: Create a new migration that reflects the *actual* state (even if it does nothing, just to sync the migration tool's history table).
*   **Process**: Revoke write access to Prod DB for developers. All changes *must* go through the CI/CD pipeline (Flyway/Alembic).
