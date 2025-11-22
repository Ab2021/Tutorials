# Day 5: Tuning - Deep Dive

## Deep Dive & Internals

### Redpanda Tuning (`rpk redpanda tune`)
This command applies OS-level tuning:
-   **Transparent Hugepages (THP)**: Disables it (causes latency spikes).
-   **Swappiness**: Sets to 0.
-   **Clocksource**: Sets to `tsc` (fastest).

### Producer Tuning Checklist
1.  `batch.size`: Increase to 16KB-64KB.
2.  `linger.ms`: Set to 5-10ms.
3.  `compression.type`: `lz4` or `zstd`.
4.  `acks`: `1` for throughput, `all` for safety.

### Consumer Tuning Checklist
1.  `fetch.min.bytes`: Wait for data before returning.
2.  `max.poll.records`: Process more data per poll.
3.  `socket.receive.buffer.bytes`: Increase for high-latency WAN links.

### Performance Implications
-   **Compression**: High compression (Gzip) saves disk/network but burns CPU. If your broker is CPU-bound, switch to LZ4.
