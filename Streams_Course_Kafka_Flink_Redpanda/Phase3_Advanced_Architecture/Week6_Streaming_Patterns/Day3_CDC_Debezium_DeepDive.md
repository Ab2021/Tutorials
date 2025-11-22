# Day 3: CDC - Deep Dive

## Deep Dive & Internals

### The "Outbox Pattern"
Solving the "Dual Write" problem elegantly.
1.  **Transaction**: App writes to `Orders` table AND `Outbox` table in the *same* DB transaction.
2.  **CDC**: Debezium reads the `Outbox` table and pushes events to Kafka.
3.  **Consumer**: Flink reads Kafka.
-   **Benefit**: Events are guaranteed to be published if and only if the transaction committed.

### Schema Evolution in CDC
-   **DDL**: `ALTER TABLE` in DB.
-   **Debezium**: Detects the change and updates the Schema Registry.
-   **Downstream**: Must handle the new schema (e.g., new column).

### Advanced Reasoning
**Postgres WAL vs MySQL Binlog**
-   **MySQL**: Statement-based (unsafe) or Row-based (safe). Debezium needs Row-based.
-   **Postgres**: Logical Decoding plugins (`pgoutput`).
-   **Impact**: WAL slots in Postgres can fill up disk if the consumer (Debezium) is down.

### Performance Implications
-   **Log Volume**: High-churn tables generate massive CDC logs. Filter unnecessary columns or tables.
