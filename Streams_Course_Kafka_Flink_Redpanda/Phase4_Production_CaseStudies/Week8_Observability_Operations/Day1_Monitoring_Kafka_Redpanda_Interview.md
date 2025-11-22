# Day 1: Monitoring Kafka & Redpanda - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the most important Kafka metric to monitor?**
    -   *A*: `UnderReplicatedPartitions`. If > 0, you risk data loss.

2.  **Q: How do you monitor consumer lag?**
    -   *A*: Use `kafka-consumer-groups --describe` or Burrow, or monitor `records-lag-max` metric.

3.  **Q: What's the difference between JMX and Prometheus metrics?**
    -   *A*: JMX is Java-specific pull model. Prometheus is language-agnostic, time-series optimized.

### Production Challenges
-   **Challenge**: **Metric Explosion**.
    -   *Scenario*: Prometheus OOM due to high cardinality (per-partition metrics).
    -   *Fix*: Aggregate at topic level, drop unnecessary labels.

-   **Challenge**: **Stale Metrics**.
    -   *Scenario*: Broker crashes, Prometheus still shows old data.
    -   *Fix*: Use `up` metric and alerting on staleness.

### Troubleshooting Scenarios
**Scenario**: `UnderReplicatedPartitions` is increasing.
-   *Cause*: Broker down, network partition, or disk full.
-   *Fix*: Check broker logs, disk usage, network connectivity.
