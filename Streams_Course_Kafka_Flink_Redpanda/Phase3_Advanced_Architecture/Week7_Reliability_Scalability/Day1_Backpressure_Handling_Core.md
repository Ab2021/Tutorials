# Day 1: Backpressure & Lag

## Core Concepts & Theory

### What is Backpressure?
When a downstream consumer is slower than the upstream producer.
-   **Symptom**: Buffers fill up.
-   **Mechanism**: The slow consumer stops asking for data. The producer's send buffer fills up. The producer blocks (or slows down).
-   **Propagation**: Backpressure propagates upstream all the way to the source (Kafka).

### Consumer Lag
The difference between the "Latest Offset" in Kafka and the "Current Offset" of the consumer group.
-   **Lag > 0**: Normal.
-   **Lag Increasing**: The consumer cannot keep up.

### Architectural Reasoning
**Handling Backpressure**
1.  **Scale Up**: Increase parallelism (more consumers).
2.  **Optimize**: Fix the slow sink or slow processing logic (e.g., async I/O).
3.  **Drop Data**: If real-time is more important than completeness (e.g., metrics), sample/drop data.

### Key Components
-   **Credit-Based Flow Control**: Flink's internal mechanism to handle backpressure between tasks.
-   **Burrow / Lag Exporter**: Tools to monitor Kafka lag.
