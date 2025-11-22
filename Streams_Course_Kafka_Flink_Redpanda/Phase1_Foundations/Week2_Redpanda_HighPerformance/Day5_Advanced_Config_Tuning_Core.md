# Day 5: Advanced Configuration & Tuning

## Core Concepts & Theory

### Latency vs Throughput
-   **Latency**: Time to deliver one message. (Optimize: `linger.ms=0`, `compression=none`).
-   **Throughput**: Messages per second. (Optimize: `linger.ms=5`, `batch.size=64KB`, `compression=lz4`).
You cannot maximize both simultaneously.

### Network Threads
-   **Kafka**: `num.network.threads`. Handles TCP connections.
-   **Redpanda**: Seastar handles this automatically per core.

### Disk I/O Tuning
-   **Commit Latency**: `log.flush.interval.messages` (fsync).
    -   Kafka defaults to letting OS handle fsync (fast, but risk of data loss on OS crash).
    -   Redpanda defaults to `fsync` on every batch (safe).

### Architectural Reasoning
**The "Zero-Copy" Myth**
Zero-copy is great for plaintext. But if you use TLS (Encryption), the CPU *must* read the data to encrypt it. So Zero-copy is disabled for TLS. Redpanda uses hardware acceleration (AES-NI) to minimize this cost.

### Key Components
-   **linger.ms**: Artificial delay to build batches.
-   **socket.send.buffer.bytes**: TCP buffer size.
-   **compression.type**: Gzip (high CPU), Snappy/LZ4 (fast), Zstd (balanced).
