# Day 1: Flink SQL & Table API

## Core Concepts & Theory

### Declarative Stream Processing
Instead of writing Java/Python code (DataStream API), you write SQL.
-   **Table API**: Fluent API in Java/Python (`table.select(...)`).
-   **SQL**: Standard ANSI SQL (`SELECT * FROM ...`).

### Dynamic Tables
A stream is a table that is constantly changing.
-   **Stream -> Table**: The stream is interpreted as a changelog.
-   **Continuous Query**: The query runs forever, updating the result table as new rows arrive.
-   **Table -> Stream**: The result table is converted back to a stream (Append-only, Retract, or Upsert).

### Architectural Reasoning
**Why SQL?**
-   **Accessibility**: Analysts can write streaming jobs.
-   **Optimization**: The Catalyst-like optimizer (Calcite) can reorder joins, push down filters, and choose efficient state backends automatically.

### Key Components
-   `StreamTableEnvironment`: The entry point.
-   `CREATE TABLE`: Defines sources/sinks (Kafka, JDBC, Files).
-   `INSERT INTO`: Submits a job.
