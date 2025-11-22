# Day 4: Troubleshooting - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: A Kafka consumer is lagging. How do you debug it?**
    -   *A*: Check: Is the consumer CPU bound? (Scale up). Is it I/O bound? (Check DB/Sink). Is it rebalancing? (Check logs). Is there skew? (Check partition lag).

2.  **Q: What is a Zombie Process in Flink?**
    -   *A*: A TaskManager that lost connection to JobManager but is still running. It might write duplicate data to the sink. Fencing (Epochs) prevents this.

3.  **Q: How do you handle a Schema Mismatch in production?**
    -   *A*: The consumer fails to deserialize. Stop the job. Update the schema (if backward compatible) or route bad messages to DLQ.

### Production Challenges
-   **Challenge**: **Cascading Failure**.
    -   *Scenario*: DB slows down -> Flink backpressures -> Kafka fills up -> Disk full -> Broker crash.
    -   *Fix*: Backpressure is good! It protects the DB. But monitor disk space and set retention policies.

-   **Challenge**: **Slow Memory Leak**.
    -   *Scenario*: App crashes every 3 days.
    -   *Fix*: Monitor Heap usage trend. It will look like a "sawtooth" that gradually rises. Take heap dump before crash.

### Troubleshooting Scenarios
**Scenario**: Flink job stuck in `CREATED` state.
-   *Cause*: Not enough slots (Resources) in the cluster.
-   *Fix*: Scale up cluster or cancel other jobs.
