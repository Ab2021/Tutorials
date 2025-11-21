# Day 22: Data Engineering Pipelines

> **Phase**: 3 - System Design
> **Week**: 5 - Design Principles
> **Focus**: Moving Data at Scale
> **Reading Time**: 50 mins

---

## 1. Pipeline Architectures

Data is the fuel. Pipelines are the pipes.

### 1.1 ETL vs. ELT
*   **ETL (Extract, Transform, Load)**: Old school. Transform data in memory before saving to DB. Good for privacy (scrubbing PII).
*   **ELT (Extract, Load, Transform)**: Modern (Snowflake/BigQuery). Dump raw data into Data Lake. Transform later using SQL. Scales better because storage is cheap and compute is on-demand.

### 1.2 Batch vs. Streaming
*   **Batch (Airflow)**: Run a job every night. Process the last 24h of data.
    *   *Tools*: Apache Spark, dbt.
*   **Streaming (Kafka/Flink)**: Process events as they arrive (ms latency).
    *   *Tools*: Apache Flink, Kafka Streams, Spark Structured Streaming.

---

## 2. Architectural Patterns

### 2.1 Lambda Architecture
*   **Hybrid**: Has a Batch Layer (for accuracy/history) and a Speed Layer (for real-time).
*   **Pros**: Robust.
*   **Cons**: Complexity. You must maintain two codebases (one for batch, one for stream) and merge results.

### 2.2 Kappa Architecture
*   **Stream Only**: Everything is a stream. Batch processing is just "streaming through historical data".
*   **Pros**: Single codebase (e.g., Flink).
*   **Cons**: Harder to implement correctly.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Data Quality & Drift
**Scenario**: The upstream team changes the date format from "MM-DD-YYYY" to "YYYY-MM-DD". Your pipeline crashes or produces garbage.
**Solution**:
*   **Schema Registry**: Enforce strict schemas (Avro/Protobuf) in Kafka. Changes must be backward compatible.
*   **Data Contracts**: Explicit agreements between producers and consumers.
*   **Great Expectations**: A library to validate data quality (e.g., `expect_column_values_to_be_unique`).

### Challenge 2: Backfilling
**Scenario**: You find a bug in your feature engineering logic. You fixed it, but the last 6 months of data are wrong.
**Solution**:
*   **Idempotency**: Ensure that running the pipeline twice produces the same result.
*   **Partitioning**: Data should be partitioned by time (e.g., `s3://bucket/data/date=2023-01-01/`). Rerunning a partition overwrites just that day.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is "Idempotency" in data pipelines?**
> **Answer**: An operation is idempotent if applying it multiple times has the same effect as applying it once.
> *   Example: `UPDATE user SET score = 100` is idempotent.
> *   Example: `UPDATE user SET score = score + 10` is NOT idempotent.
> *   In pipelines, idempotency allows us to safely retry failed jobs without corrupting data (e.g., double counting).

**Q2: Explain the "Watermark" concept in streaming.**
> **Answer**: In event-time processing, events might arrive late (due to network lag). A watermark is a heuristic that says "I believe all events up to time T have arrived." It allows the system to close a window and compute the result, trading off latency vs. completeness.

**Q3: Why use Parquet over CSV?**
> **Answer**:
> 1.  **Columnar**: Allows reading only specific columns (I/O saving).
> 2.  **Compression**: Highly efficient compression (Snappy/Gzip) due to type homogeneity.
> 3.  **Schema**: Stores schema in the footer. No guessing types.

---

## 5. Further Reading
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook)
- [Apache Airflow Concepts](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/index.html)
