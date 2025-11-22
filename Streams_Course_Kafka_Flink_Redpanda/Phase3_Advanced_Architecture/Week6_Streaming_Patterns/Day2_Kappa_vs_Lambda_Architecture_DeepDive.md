# Day 2: Kappa Architecture - Deep Dive

## Deep Dive & Internals

### Reprocessing in Kappa
1.  **Parallel Run**: Start `Job_V2` reading from the beginning. `Job_V1` keeps running.
2.  **Catch Up**: `Job_V2` processes history at high throughput.
3.  **Switch**: When `Job_V2` catches up to real-time, switch the downstream application to read from `Job_V2`'s output. Kill `Job_V1`.

### The "Out-of-Order" Problem
When reprocessing history, data arrives at maximum speed.
-   **Watermarks**: Crucial for handling event time correctly during replay.
-   **Throttling**: You might need to throttle the replay to avoid overwhelming the downstream DB.

### Advanced Reasoning
**Is Lambda dead?**
Not entirely. Some complex ML training or graph algorithms are still better suited for batch (finite datasets). But for ETL and analytics, Kappa is the standard.

### Performance Implications
-   **Backfill Speed**: Depends on parallelism. Flink can process historical data much faster than real-time (limited only by CPU/Network).
