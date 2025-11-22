# Day 2: Kappa Architecture - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the main benefit of Kappa over Lambda?**
    -   *A*: Code reuse. You don't need to maintain separate Batch and Streaming paths.

2.  **Q: How do you handle code bugs in Kappa?**
    -   *A*: Fix the code, deploy a new job reading from the beginning (or a savepoint), and reprocess the data.

3.  **Q: Does Kappa require infinite retention in Kafka?**
    -   *A*: Ideally yes (Tiered Storage). Or you can archive old data to S3 (Parquet) and have Flink read from S3 for history and Kafka for real-time (Hybrid Source).

### Production Challenges
-   **Challenge**: **Resource Contention**.
    -   *Scenario*: Replaying history saturates the cluster, affecting real-time jobs.
    -   *Fix*: Use a separate "Batch" cluster for backfills, or use priority/quotas.

### Troubleshooting Scenarios
**Scenario**: Backfill job is slow.
-   *Cause*: Sink bottleneck (e.g., writing to RDS).
-   *Fix*: Optimize the sink (batch writes) or scale the DB.
