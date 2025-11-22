# Day 5: Log Aggregation - Deep Dive

## Deep Dive & Internals

### Parsing & Normalization
Logs are messy text.
-   **Grok**: Regex-based parsing (slow).
-   **Structured Logging**: App logs in JSON (fast).
-   **Vector (Rust)**: High-performance replacement for Logstash. Can parse/transform in transit.

### Sigma Rules (SIEM)
Standard format for security rules.
-   *Example*: "Detect 5 failed logins from same IP in 1 minute".
-   **Implementation**: Flink CEP or Kafka Streams.
-   **State**: Requires keeping counters per IP.

### Cost Optimization
Logging is expensive (Volume).
-   **Sampling**: Log 100% of Errors, but only 1% of Info/Debug.
-   **Dynamic Level**: Change log level at runtime without restart.
-   **Tiered Storage**: Move data to S3 ASAP. Query directly from S3 (e.g., using ChaosSearch or Athena).

### Performance Implications
-   **Index Rate**: ElasticSearch bottleneck.
    -   *Fix*: Bulk API, remove unnecessary fields, increase refresh interval.
-   **Network**: Compressing logs (Zstd) at the Agent level saves massive bandwidth.
