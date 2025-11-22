# Day 5: Tuning - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you optimize for low latency?**
    -   *A*: `linger.ms=0`, `acks=1`, `compression=none`, ensure producers/consumers are close to brokers.

2.  **Q: How do you optimize for high throughput?**
    -   *A*: Batching (`linger.ms > 0`), Compression (`lz4`), Parallelism (more partitions/consumers).

3.  **Q: What is the impact of `fsync`?**
    -   *A*: `fsync` forces data to physical disk. It is slow. Doing it on every message destroys throughput. Kafka relies on replication for safety, not fsync.

### Production Challenges
-   **Challenge**: **High Tail Latency (p99)**.
    -   *Cause*: GC pauses (Kafka), noisy neighbors, or slow disk.
    -   *Fix*: Tune GC, isolate resources, upgrade disks.

-   **Challenge**: **Network Saturation**.
    -   *Scenario*: 10Gbps link is full.
    -   *Fix*: Enable compression (`zstd`), scale out brokers.

### Troubleshooting Scenarios
**Scenario**: Producer throughput is low, but CPU/Disk are idle.
-   *Cause*: `linger.ms` might be too high, or `max.in.flight.requests` is 1 (stop-and-wait).
-   *Fix*: Check client config.
