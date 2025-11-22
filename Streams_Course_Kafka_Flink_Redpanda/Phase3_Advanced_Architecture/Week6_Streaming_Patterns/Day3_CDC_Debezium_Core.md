# Day 3: Change Data Capture (CDC)

## Core Concepts & Theory

### What is CDC?
Capturing changes (INSERT, UPDATE, DELETE) from a database transaction log and streaming them as events.
-   **Pattern**: DB -> CDC Connector (Debezium) -> Kafka -> Flink.

### Debezium
The standard open-source CDC platform.
-   Reads binary logs (MySQL binlog, Postgres WAL).
-   Guarantees **ordering** and **completeness**.

### Architectural Reasoning
**Why CDC?**
-   **No Dual Writes**: Don't write to DB and Kafka manually (race conditions). Write to DB, let CDC propagate to Kafka.
-   **Legacy Integration**: Turn a monolithic SQL DB into an event stream without changing the app code.

### Key Components
-   **Snapshot**: Initial load of the table.
-   **Streaming**: Tailing the log.
-   **Tombstone**: A null value in Kafka indicating a DELETE.
