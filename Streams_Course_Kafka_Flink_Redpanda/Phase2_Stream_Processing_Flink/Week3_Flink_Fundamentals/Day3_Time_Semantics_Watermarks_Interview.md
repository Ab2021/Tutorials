# Day 3: Time Semantics - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is a Watermark?**
    -   *A*: A timestamp that asserts "all events with timestamp < T have arrived". It triggers event-time timers (windows).

2.  **Q: How do you handle late data?**
    -   *A*: Allowed Lateness (update old result), Side Outputs (save for later), or just drop it.

3.  **Q: What is the difference between Ingestion Time and Event Time?**
    -   *A*: Ingestion is when Flink sees it. Event is when it happened. Event time allows deterministic replay.

### Production Challenges
-   **Challenge**: **Stalled Watermark**.
    -   *Scenario*: Windows are not closing.
    -   *Cause*: One Kafka partition is empty (idle).
    -   *Fix*: Use `withIdleness()`.

### Troubleshooting Scenarios
**Scenario**: Data is dropped unexpectedly.
-   *Cause*: Watermarks are too aggressive (assuming 1s lag when real lag is 5s).
-   *Fix*: Adjust `BoundedOutOfOrderness` strategy.
