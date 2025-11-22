# Day 5: Unified Batch/Stream - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the Lambda Architecture?**
    -   *A*: Parallel Batch and Speed layers. Complex to maintain.

2.  **Q: How do you backfill 1 year of data in a Streaming system?**
    -   *A*:
        1.  Start Flink job from "Earliest".
        2.  It processes historical data (high throughput).
        3.  It catches up to "Now".
        4.  It continues as a streaming job.

3.  **Q: Why use Flink for Batch?**
    -   *A*: Code reuse. Write logic once, run on historical data (backtesting) and real-time data (production).

### Production Challenges
-   **Challenge**: **Backfill Speed**.
    -   *Scenario*: Replaying 1 year takes 1 month.
    -   *Fix*: Increase parallelism for the backfill job. Use Batch Mode (Blocking Shuffle) for efficiency.

-   **Challenge**: **Side Effects**.
    -   *Scenario*: Job sends emails. Replaying history sends 1M emails.
    -   *Fix*: Disable "Sinks" (or switch to "Dry Run" Sink) during backfill.
