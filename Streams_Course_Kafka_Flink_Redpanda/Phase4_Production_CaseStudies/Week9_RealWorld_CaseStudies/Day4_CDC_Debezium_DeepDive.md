# Day 4: CDC - Deep Dive

## Deep Dive & Internals

### Handling Schema Evolution
What happens when `ALTER TABLE` runs on the source?
-   **Debezium**: Detects the change. Updates Schema Registry. Emits a schema change event.
-   **Downstream**:
    -   **Avro**: If compatible (e.g., add optional field), consumer continues.
    -   **Incompatible**: Consumer fails. Requires manual intervention or a "Schema Router" to send bad data to DLQ.

### Snapshotting
Initial load of a 1TB table.
-   **Locking**: Debezium used to lock the table (Global Read Lock). Bad for prod.
-   **Incremental Snapshot**: New algorithm. Interleaves chunks of snapshot reads with log streaming. No locks.

### Ordering Guarantees
-   **Kafka Partitioning**: Must partition by Primary Key. Ensures all updates to `ID=1` go to the same partition (ordered).
-   **Compaction**: Kafka Log Compaction keeps only the latest state for a key. Crucial for CDC to save space.

### Performance Implications
-   **Toast Columns (Postgres)**: Large text/binary fields are stored separately. Debezium might not see them unless `REPLICA IDENTITY FULL` is set (High DB overhead).
