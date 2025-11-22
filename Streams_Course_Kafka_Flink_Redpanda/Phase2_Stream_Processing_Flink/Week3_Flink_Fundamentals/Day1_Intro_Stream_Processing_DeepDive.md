# Day 1: Intro to Stream Processing - Deep Dive

## Deep Dive & Internals

### The Flink Runtime
-   **JobGraph**: The optimized logical plan (operators chained together).
-   **ExecutionGraph**: The physical plan (parallel tasks distributed across TaskManagers).
-   **Operator Chaining**: Flink fuses adjacent operators (e.g., Map -> Filter) into a single thread to reduce serialization/deserialization overhead and buffer exchange.

### Memory Management
Flink manages its own memory (off-heap) to avoid JVM GC pauses.
-   **Network Buffers**: For data exchange between TaskManagers.
-   **Managed Memory**: For internal data structures (hash tables, sort buffers) and State Backends (RocksDB).

### Advanced Reasoning
**Why "True Streaming"?**
Spark Streaming (legacy) used micro-batches. This meant latency was bounded by batch duration (seconds). Flink processes event-by-event, achieving sub-millisecond latency. This is critical for fraud detection or high-frequency trading.

### Performance Implications
-   **Backpressure**: Flink uses a credit-based flow control mechanism. If a downstream operator is slow, it stops granting credits to the upstream, naturally slowing down the source without data loss.
