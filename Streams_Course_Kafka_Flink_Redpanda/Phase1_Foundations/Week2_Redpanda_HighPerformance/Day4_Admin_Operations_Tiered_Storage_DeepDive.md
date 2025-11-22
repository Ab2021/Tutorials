# Day 4: Admin Operations - Deep Dive

## Deep Dive & Internals

### Redpanda Tiered Storage Internals
-   **Manifest**: Redpanda maintains a manifest file in S3 that lists all segments.
-   **Upload**: As soon as a segment is closed (or reaches a size limit), it is queued for upload.
-   **Prefetching**: When a consumer reads from S3, Redpanda speculatively fetches the *next* segment to hide latency.

### ACLs (Access Control Lists)
-   **Principal**: User (User:Alice).
-   **Resource**: Topic, Group, Cluster.
-   **Operation**: Read, Write, Describe, Create.
-   **Pattern**: Literal (exact name) or Prefixed.

### Advanced Reasoning
**Why is Rebalancing hard?**
It involves copying gigabytes of data. If you do it too fast, you saturate the NIC. If you do it too slow, the cluster remains unbalanced. Redpanda uses a specialized **Partition Balancer** that continuously optimizes placement based on disk/CPU usage.

### Performance Implications
-   **Tiered Storage Latency**: Reading from S3 has high First-Byte Latency (50-100ms). Throughput is high. Not suitable for real-time latency-sensitive apps, but great for catch-up or replay.
