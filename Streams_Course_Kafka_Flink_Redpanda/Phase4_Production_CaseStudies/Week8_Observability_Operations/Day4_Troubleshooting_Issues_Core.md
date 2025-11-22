# Day 4: Troubleshooting Streaming Systems

## Core Concepts & Theory

### The Troubleshooting Loop
1.  **Detect**: Alert fires or user complains.
2.  **Isolate**: Is it the Source (Kafka), the Processor (Flink), or the Sink (DB)?
3.  **Mitigate**: Stop the bleeding (Scale up, restart, rollback).
4.  **Diagnose**: Find root cause (Logs, Metrics, Traces).
5.  **Fix**: Apply permanent fix.

### Common Failure Patterns
1.  **The "Poison Pill"**: A message that crashes the consumer.
    -   *Symptom*: Consumer restarts loop.
    -   *Fix*: Dead Letter Queue (DLQ).
2.  **The "Data Skew"**: One partition has 90% of data.
    -   *Symptom*: One task is 100% CPU, others idle. Backpressure.
    -   *Fix*: Re-key (Salt) or local aggregation.
3.  **The "Death Spiral"**: System slows down -> Retries increase -> Load increases -> System slows down more.
    -   *Fix*: Circuit Breakers, Exponential Backoff.

### Architectural Reasoning
**Fail Fast vs Fail Safe**
-   **Fail Fast**: Crash immediately on error. Good for data integrity.
-   **Fail Safe**: Drop the bad record and continue. Good for availability.
-   **Stream Processing**: Usually prefers Fail Fast (to prevent data loss), but with DLQ for bad data.

### Key Components
-   **Logs**: `server.log`, `jobmanager.log`.
-   **Thread Dumps**: `jstack`. Crucial for "stuck" processes.
-   **Heap Dumps**: `jmap`. Crucial for Memory Leaks.
