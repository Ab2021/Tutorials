# Day 1: Backpressure - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How does Flink handle backpressure?**
    -   *A*: Using Credit-Based Flow Control. It propagates backpressure naturally without needing a special "Rate Limiter" component.

2.  **Q: What is the difference between Backpressure and Lag?**
    -   *A*: Backpressure is the *state* of the system (buffers full). Lag is the *metric* (offset difference) resulting from backpressure.

3.  **Q: How do you debug a "High Backpressure" alert?**
    -   *A*: Check the Flink Web UI. Find the task with "High" backpressure. Look at its *downstream* task. The bottleneck is usually the first task *without* backpressure (it's the one causing it).

### Production Challenges
-   **Challenge**: **GC Pauses causing Lag**.
    -   *Scenario*: Consumer pauses for 10s due to GC. Kafka rebalances.
    -   *Fix*: Tune JVM (G1GC), increase heap, or reduce state object creation.

### Troubleshooting Scenarios
**Scenario**: Lag is increasing, but CPU is low.
-   *Cause*: I/O Bottleneck (Sink is slow, or Network is saturated).
-   *Fix*: Use Async I/O or batch writes to sink.
