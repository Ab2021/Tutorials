# Day 4: CDC - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you handle "Hard Deletes" in the source DB?**
    -   *A*: Debezium emits a `DELETE` event (op='d') and a **Tombstone** (Key, Null). The Tombstone tells Kafka to delete the key during compaction.

2.  **Q: What is the "Dual Write" problem and how does CDC fix it?**
    -   *A*: App writes to DB and then tries to publish to Kafka. If app crashes in between, data is inconsistent. CDC ensures consistency by reading the DB log (Source of Truth).

3.  **Q: How do you re-process data if you find a bug in the consumer?**
    -   *A*: Reset consumer offset to 0. But if data was compacted, you lost history.
    -   *Better*: Trigger a new **Ad-hoc Snapshot** in Debezium (using Signals).

### Production Challenges
-   **Challenge**: **WAL Growth**.
    -   *Scenario*: Kafka is down. Debezium stops reading WAL. Postgres keeps WAL segments until confirmed. Disk fills up.
    -   *Fix*: Monitoring! Alert on Replication Slot Lag.

-   **Challenge**: **Sensitive Data**.
    -   *Scenario*: PII in DB log.
    -   *Fix*: Debezium SMT (Single Message Transform) to blacklist columns or hash PII before it hits Kafka.

### Troubleshooting Scenarios
**Scenario**: Debezium connector fails with `invalid authorization specification`.
-   *Cause*: DB password changed.
-   *Fix*: Update connector config (REST API).
