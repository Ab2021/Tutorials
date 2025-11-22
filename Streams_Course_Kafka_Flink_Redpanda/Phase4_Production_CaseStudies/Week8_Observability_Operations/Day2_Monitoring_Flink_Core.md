# Day 2: Monitoring Flink

## Core Concepts & Theory

### The Flink Metric System
Flink has a pluggable metric system.
-   **System Metrics**: CPU, Memory, GC, Threads (from JVM).
-   **Flink Metrics**: Checkpointing, Restart/Failover, Network Buffers.
-   **User Metrics**: Counters, Gauges, Histograms defined in your code.

### Critical Flink Metrics
1.  **Availability**
    -   `uptime`: Time since last restart.
    -   `numRestarts`: If increasing, the job is unstable.
    -   `fullRestarts`: JobManager failure.

2.  **Throughput & Latency**
    -   `numRecordsInPerSecond` / `numRecordsOutPerSecond`.
    -   `latency`: End-to-end latency (requires Latency Markers, expensive!).

3.  **Backpressure**
    -   `outPoolUsage`: If 100%, the downstream task is slow.
    -   `isBackPressured`: Boolean indicator (newer Flink versions).

4.  **Checkpointing**
    -   `lastCheckpointDuration`: If increasing, state is growing or storage is slow.
    -   `lastCheckpointSize`: Size of the state.

### Architectural Reasoning
**Why not use Latency Markers?**
Flink can inject "Latency Markers" that travel with records.
-   **Pros**: Exact per-operator latency.
-   **Cons**: modifying the stream, adds overhead, skews throughput.
-   **Alternative**: Measure "Business Latency" (Event Time - Processing Time) using a Histogram in the Sink.

### Key Components
-   **MetricReporter**: Interface to send metrics (Prometheus, Datadog, StatsD).
-   **Flink Web UI**: Real-time dashboard (good for debugging, not for alerting).
