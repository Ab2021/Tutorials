# Day 22: Data Pipelines - Interview Questions

> **Topic**: Data Engineering
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the difference between ETL and ELT?
**Answer:**
*   **ETL**: Extract, Transform, Load. Transform before DB. Good for security/compliance.
*   **ELT**: Extract, Load, Transform. Load raw into Data Lake/Warehouse. Transform there (SQL). Scalable (BigQuery/Snowflake).

### 2. Explain the role of Apache Airflow.
**Answer:**
*   Workflow Orchestration.
*   Defines DAGs (Directed Acyclic Graphs) of tasks.
*   Handles scheduling, retries, dependencies, and monitoring.

### 3. What is a DAG in Airflow?
**Answer:**
*   Collection of tasks with dependencies.
*   Defined in Python.
*   Ensures Task B runs only after Task A succeeds.

### 4. How do you handle "Backfilling" in data pipelines?
**Answer:**
*   Running the pipeline for past dates (e.g., re-computing metrics for last year).
*   Airflow supports this via `catchup=True` or manual CLI runs.
*   Requires **Idempotency**.

### 5. What is Idempotency? Why is it critical?
**Answer:**
*   Running the same task multiple times produces the same result.
*   **Critical**: If a pipeline fails and retries, it shouldn't duplicate data (e.g., `INSERT` vs `UPSERT`).

### 6. What is the difference between Batch and Streaming pipelines?
**Answer:**
*   **Batch**: Process bounded data (files) at intervals (Daily). High latency, high throughput. (Spark).
*   **Streaming**: Process unbounded data (events) continuously. Low latency. (Kafka/Flink).

### 7. Explain the Lambda Architecture.
**Answer:**
*   Hybrid approach.
*   **Batch Layer**: Accurate, comprehensive, slow.
*   **Speed Layer**: Approximate, real-time, fast.
*   **Serving Layer**: Merges results.

### 8. What is the Kappa Architecture?
**Answer:**
*   Simplification of Lambda.
*   Everything is a stream.
*   Batch processing is just streaming through stored history.

### 9. What is Apache Kafka?
**Answer:**
*   Distributed Event Streaming Platform.
*   **Pub/Sub** model.
*   High throughput, fault-tolerant, durable log.
*   Decouples producers and consumers.

### 10. How do you handle "Late Arriving Data" in streaming?
**Answer:**
*   **Watermarks**: Heuristic for "how late can data be?".
*   **Windowing**: Wait for X minutes.
*   Update previous results (Retraction) if data arrives after window closes.

### 11. What is Data Lineage?
**Answer:**
*   Tracking the flow of data. Origin -> Transformations -> Destination.
*   Crucial for debugging ("Why is this metric wrong?") and compliance.

### 12. What is Data Quality Testing? (Great Expectations).
**Answer:**
*   Asserting properties of data.
*   "Nulls < 1%", "Age > 0", "Schema matches".
*   Run before training to prevent GIGO.

### 13. Explain the "Push" vs "Pull" model in pipelines.
**Answer:**
*   **Push**: Upstream triggers downstream. Low latency. Tight coupling.
*   **Pull**: Downstream polls upstream. Decoupled.

### 14. What is Parquet format? Why is it better than CSV?
**Answer:**
*   **Columnar** storage.
*   **Compressed** efficiently (Run-length encoding).
*   **Schema** embedded.
*   Much faster for analytical queries (reading subset of columns).

### 15. How do you optimize a Spark Job?
**Answer:**
*   **Partitioning**: Avoid skew.
*   **Caching**: Cache intermediate results.
*   **Broadcast Join**: Broadcast small table to all nodes to avoid shuffle.
*   **Serialization**: Use Kryo.

### 16. What is "Shuffle" in distributed computing?
**Answer:**
*   Redistributing data across network to group by key.
*   Expensive (Disk I/O + Network).
*   Minimize it.

### 17. How do you handle Schema Evolution?
**Answer:**
*   **Forward Compatibility**: Old reader can read new data.
*   **Backward Compatibility**: New reader can read old data.
*   Use Avro/Protobuf registries.

### 18. What is a Dead Letter Queue (DLQ)?
**Answer:**
*   Queue for messages that failed to process.
*   Prevents blocking the pipeline. Allows manual inspection later.

### 19. What is CDC (Change Data Capture)?
**Answer:**
*   Capturing changes (Insert/Update/Delete) from a DB transaction log (Binlog) and streaming them.
*   Debezium is a popular tool.

### 20. How do you secure a data pipeline?
**Answer:**
*   IAM Roles (Least Privilege).
*   VPC Endpoints (Private network).
*   Encryption (KMS).
*   Audit Logs.
