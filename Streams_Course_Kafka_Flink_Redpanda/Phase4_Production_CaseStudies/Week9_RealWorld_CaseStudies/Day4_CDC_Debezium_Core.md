# Day 4: CDC with Debezium

## Core Concepts & Theory

### The Use Case
Sync a legacy Monolithic Database (MySQL/Postgres) to a Microservice ecosystem or Data Lake.
-   **Goal**: Zero code change in the monolith.
-   **Latency**: Sub-second.

### Architecture
1.  **Source**: MySQL (Binlog) / Postgres (WAL).
2.  **Connector**: Debezium (running in Kafka Connect).
3.  **Transport**: Kafka.
4.  **Sink**: Flink (Transformation) -> ElasticSearch / Snowflake.

### Key Patterns
-   **The Outbox Pattern**: Reliable messaging. Write to `Outbox` table in DB transaction. Debezium captures it.
-   **Strangler Fig**: Migrate functionality from Monolith to Microservices piece by piece using CDC data.

### Architectural Reasoning
**Log-Based CDC vs Query-Based (JDBC)**
-   **JDBC (Polling)**: `SELECT * FROM table WHERE updated_at > last_check`.
    -   *Cons*: Misses hard deletes. Polls add load. Latency.
-   **Log-Based (Debezium)**: Reads the transaction log.
    -   *Pros*: Captures Deletes. Real-time. No impact on DB query engine.
