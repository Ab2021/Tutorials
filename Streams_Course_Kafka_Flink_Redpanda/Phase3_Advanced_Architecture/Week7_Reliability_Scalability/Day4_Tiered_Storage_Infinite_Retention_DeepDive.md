# Day 4: Tiered Storage - Deep Dive

## Deep Dive & Internals

### Redpanda / Kafka Implementation
-   **Segments**: Kafka logs are split into segments (e.g., 1GB).
-   **Archiver**: A background thread uploads closed segments to S3.
-   **Remote Index**: The broker keeps the index of S3 segments in memory (or local disk).
-   **Fetch**:
    1.  Consumer asks for Offset 0.
    2.  Broker checks local disk. Miss.
    3.  Broker checks Remote Index. Found in S3 Object X.
    4.  Broker downloads Object X (or range read) to local cache.
    5.  Broker serves data to consumer.

### Advanced Reasoning
**Impact on Kappa Architecture**
Tiered Storage is the enabler for Kappa. You can keep years of history.
-   **Backfill**: High-throughput sequential read from S3 is very fast (often saturates network).

### Performance Implications
-   **First Byte Latency**: Reading cold data has higher latency (S3 GET).
-   **Cost**: S3 API costs (GET/PUT) can be high if not optimized (batching).
