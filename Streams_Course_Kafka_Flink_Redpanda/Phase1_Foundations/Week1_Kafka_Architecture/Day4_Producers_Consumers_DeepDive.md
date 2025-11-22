# Day 4: Producers & Consumers - Deep Dive

## Deep Dive & Internals

### Producer Internals
1.  **`send()`**: Adds record to a buffer (accumulator).
2.  **Sender Thread**: Background thread that drains the buffer.
3.  **Batching**: Group records by partition.
    -   `batch.size`: Max bytes per batch (e.g., 16KB).
    -   `linger.ms`: Max time to wait to fill a batch (e.g., 5ms).
    -   **Trade-off**: High `linger.ms` = High Throughput, Higher Latency.

### Consumer Group Protocol
1.  **JoinGroup**: Consumers send "I want to join" to the **Group Coordinator** (a specific broker).
2.  **SyncGroup**: The Coordinator elects a **Leader Consumer**. The Leader decides the partition assignment and sends it back to the Coordinator.
3.  **Heartbeats**: Consumers send heartbeats. If missed, they are kicked out -> Rebalance.

### Sticky Assignor
-   **Range/RoundRobin**: Reassigns everything from scratch. High churn.
-   **Sticky**: Tries to keep existing assignments. Only moves what's necessary. Reduces rebalance "stop-the-world" time.

### Advanced Reasoning
**Why Client-Side Partitioning?**
The producer decides the partition. This avoids double-hop (Producer -> Broker A -> Broker B). The producer sends directly to the leader of the correct partition.

### Performance Implications
-   **Compression**: Batches are compressed (Snappy, LZ4). Better compression with larger batches.
-   **Fetch Size**: Consumers fetch bytes, not messages. `fetch.min.bytes` allows the broker to wait until it has enough data (efficiency).
