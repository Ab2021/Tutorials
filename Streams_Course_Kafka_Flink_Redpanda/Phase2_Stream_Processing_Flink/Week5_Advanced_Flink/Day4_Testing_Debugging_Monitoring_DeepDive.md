# Day 4: Testing - Deep Dive

## Deep Dive & Internals

### Backpressure Monitoring
Flink 1.13+ uses **Task Sampling**.
-   The JobManager periodically samples the stack traces of TaskManagers.
-   If a task is stuck in `requestMemoryBuffer` (waiting for network), it is backpressured.
-   This is zero-overhead compared to the old method (injecting metrics).

### Flame Graphs
Flink Web UI can generate Flame Graphs (CPU profiles) on the fly.
-   Identify hot methods (e.g., Regex parsing, serialization).

### Advanced Reasoning
**Unit Testing Stateful Functions**
You cannot just instantiate `MyProcessFunction` and call `processElement`. You need the `RuntimeContext` (for state access). The `TestHarness` mocks this context, the state backend, and the timer service.

### Performance Implications
-   **Metric Overhead**: Too many metrics (e.g., per-key metrics) can degrade performance. Use `metrics.latency.interval` sparingly.
