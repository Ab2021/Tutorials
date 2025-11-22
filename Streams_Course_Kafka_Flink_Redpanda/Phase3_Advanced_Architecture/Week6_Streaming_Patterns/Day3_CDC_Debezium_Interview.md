# Day 3: CDC - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the Outbox Pattern?**
    -   *A*: A pattern to ensure transactional consistency between a DB write and a message publication.

2.  **Q: How does Debezium handle initial snapshots?**
    -   *A*: It scans the table (SELECT *) while holding a lock (or using MVCC) to get a consistent snapshot, then switches to log tailing.

3.  **Q: What happens if you delete a row in the DB?**
    -   *A*: Debezium emits a record with `op='d'` (delete) and then a Tombstone (Key, Null) to allow Kafka Log Compaction to remove it.

### Production Challenges
-   **Challenge**: **WAL Disk Usage**.
    -   *Scenario*: Debezium is down, Postgres keeps WAL segments forever. Disk fills up. DB crashes.
    -   *Fix*: Monitor replication slot lag. Set `max_slot_wal_keep_size`.

### Troubleshooting Scenarios
**Scenario**: CDC stream is lagging.
-   *Cause*: DB is under heavy write load. Debezium is single-threaded per connector.
-   *Fix*: Shard the connector (one per table) or optimize DB log I/O.
