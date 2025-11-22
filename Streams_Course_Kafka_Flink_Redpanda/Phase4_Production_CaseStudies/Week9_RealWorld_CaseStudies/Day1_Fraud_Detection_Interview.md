# Day 1: Fraud Detection - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you update fraud rules without stopping the job?**
    -   *A*: Use **Broadcast State**. Read rules from a side stream (Kafka topic `rules`). Broadcast them to all parallel tasks. Store in `MapState`. Apply active rules to every transaction.

2.  **Q: How do you handle false positives?**
    -   *A*: The system emits a "Probability Score" rather than a binary Block/Allow. If Score > 90, block. If 50-90, SMS verification.

3.  **Q: Why not use a database for this?**
    -   *A*: Latency. Polling a DB for "last 5 transactions" is too slow for 10k TPS. Flink keeps state locally.

### Production Challenges
-   **Challenge**: **Cold Start**.
    -   *Scenario*: New Flink job starts with empty state. It doesn't know the "Average Spend".
    -   *Fix*: **State Bootstrap**. Read historical data from S3/DB and load it into Flink state using the State Processor API before starting the stream.

-   **Challenge**: **Hot Keys**.
    -   *Scenario*: One merchant (e.g., Amazon) has huge volume.
    -   *Fix*: This logic is usually keyed by `CardID`, which is well distributed. If keyed by Merchant, use Salting.

### Troubleshooting Scenarios
**Scenario**: Latency spikes to 2 seconds.
-   *Cause*: Async I/O to Feature Store is timing out.
-   *Fix*: Add local caching (Guava) or scale the Feature Store.
