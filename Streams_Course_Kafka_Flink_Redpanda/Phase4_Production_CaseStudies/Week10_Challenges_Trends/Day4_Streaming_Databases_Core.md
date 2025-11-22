# Day 4: Streaming Databases

## Core Concepts & Theory

### The Convergence
-   **Database**: Stores state. Queries are transient.
-   **Stream Processor**: Stores queries (Topology). Data is transient.
-   **Streaming Database**: Stores State AND Continuous Queries. (Materialize, RisingWave, ksqlDB).

### Materialized Views
A standard DB view is calculated on read. A **Materialized View** is pre-calculated.
-   **Streaming DB**: Updates the Materialized View incrementally as new data arrives.
-   **Benefit**: Sub-millisecond query latency for complex joins/aggregates.

### Key Players
1.  **ksqlDB**: Kafka-native. Good for simple transformations.
2.  **Materialize**: Postgres-compatible. Uses Differential Dataflow. Strong consistency.
3.  **RisingWave**: Cloud-native. S3-based state.

### Architectural Reasoning
**Flink vs Streaming DB**
-   **Flink**: Imperative (Java/Python) + SQL. Good for complex logic, external calls, pipelines.
-   **Streaming DB**: Pure SQL. Good for "Serving Layer" (powering dashboards/APIs).
-   **Pattern**: Kafka -> Flink (Complex ETL) -> Kafka -> RisingWave (Serving).
