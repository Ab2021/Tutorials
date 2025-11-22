# Day 1: Backpressure - Deep Dive

## Deep Dive & Internals

### Flink Credit-Based Flow Control
-   Each TaskManager has a **Network Buffer Pool**.
-   The receiver sends "credits" to the sender indicating how many buffers it has available.
-   The sender only sends data if it has credits.
-   This prevents a single slow task from overwhelming the network and causing OOMs.

### Identifying the Bottleneck
1.  **Source**: If source is idle but lag is high -> Downstream is slow.
2.  **Sink**: If sink is busy (100% CPU or high I/O wait) -> Sink is the bottleneck.
3.  **Skew**: If only one subtask is backpressured -> Data Skew.

### Advanced Reasoning
**The "Death Spiral"**
If a consumer is slow, it might trigger rebalances (timeouts). Rebalancing stops the world. Lag increases. When it resumes, it has *more* data to catch up, causing more load, leading to another timeout.
-   **Fix**: Increase `session.timeout.ms` and `max.poll.interval.ms`.

### Performance Implications
-   **Buffer Bloat**: Too many buffers increase latency. Flink automatically adjusts buffer size to balance throughput and latency.
