# Day 4: Streaming Databases - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: When would you use ksqlDB over Flink?**
    -   *A*: When the team knows SQL but not Java. When the use case is simple filtering/joining/aggregation on Kafka. For complex state machines or async I/O, use Flink.

2.  **Q: What is the "Read-After-Write" consistency problem in streaming?**
    -   *A*: You write to Kafka, then immediately query the View. The View hasn't updated yet.
    -   *Fix*: Wait for the watermark/offset to catch up, or accept eventual consistency.

3.  **Q: How does a Streaming DB handle backpressure?**
    -   *A*: Same as Flink. Stop reading from Kafka.

### Production Challenges
-   **Challenge**: **State Explosion**.
    -   *Scenario*: `SELECT * FROM events`. Materializing the raw stream.
    -   *Fix*: Only materialize Aggregates or Filtered datasets. Set TTL (Retention) on the view.

-   **Challenge**: **Migration**.
    -   *Scenario*: Changing the SQL query.
    -   *Fix*: Usually requires rebuilding the view from scratch (Replay history).
