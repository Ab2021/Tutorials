# Day 5: Capacity Planning - Deep Dive

## Deep Dive & Internals

### Network Bandwidth Planning
Network is often the hidden bottleneck.
-   **Ingress**: Producer traffic.
-   **Replication**: Ingress * (ReplicationFactor - 1).
-   **Egress**: Consumer traffic. (Ingress * NumberOfConsumers).
-   **Total**: Ingress + Replication + Egress.
**Example**: 100MB In, 3x Rep, 2 Consumers.
-   Total = 100 + 200 + 200 = 500 MB/sec.
-   Ensure NIC (Network Interface Card) supports this (e.g., 10Gbps = 1.25 GB/sec).

### CPU Sizing: Compression
Compression (Zstd/LZ4) saves disk/network but burns CPU.
-   **Producer**: Compresses batch.
-   **Broker**: No CPU cost (Zero Copy) *unless* message format conversion is needed.
-   **Consumer**: Decompresses.
**Warning**: If Broker and Client versions mismatch, Broker must down-convert, burning massive CPU.

### Partition Sizing
-   **Too Few**: Limited parallelism.
-   **Too Many**: High unavailability during leader election (Controller overload). High memory overhead.
-   **Rule of Thumb**: < 4000 partitions per broker. < 200,000 per cluster.

### Advanced Reasoning
**OS Page Cache**
Kafka relies heavily on Linux Page Cache.
-   **RAM Sizing**: You don't need massive Heap (4-6GB is enough). You need massive **Free RAM** for Page Cache.
-   If Consumers are fast, they read from Page Cache (RAM) -> Zero Disk Read.
-   If Consumers lag, they read from Disk -> Slow.

### Performance Implications
-   **RAID**: RAID 10 is good for performance/redundancy. JBOD (Just a Bunch of Disks) is supported by Kafka (software redundancy) and preferred for cost.
