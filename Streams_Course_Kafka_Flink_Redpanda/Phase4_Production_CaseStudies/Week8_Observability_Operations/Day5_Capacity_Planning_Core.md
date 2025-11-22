# Day 5: Capacity Planning & Sizing

## Core Concepts & Theory

### The Sizing Equation
Capacity Planning is math, not magic.
**Inputs**:
1.  **Throughput**: 100 MB/sec (Peak).
2.  **Retention**: 7 Days.
3.  **Replication**: 3x.

**Outputs**:
1.  **Storage**: `100MB * 86400s * 7days * 3replicas = 181 TB`.
2.  **Bandwidth**: `100MB (In) + 300MB (Replication) + 100MB (Consumer) = 500 MB/sec`.
3.  **CPU**: Compression cost + Serialization cost.

### Headroom
Always provision for **Peak** traffic + **Headroom** (e.g., 30%).
-   **Why?**: To handle catch-up bursts. If you size exactly for peak, you can never catch up after an outage.

### Architectural Reasoning
**Disk I/O: The Bottleneck**
Kafka is usually Disk I/O bound or Network bound.
-   **IOPS**: Not important for Kafka (Sequential I/O).
-   **Throughput**: Very important. NVMe SSDs are preferred.
-   **HDD**: Can be used for "Cold" storage (Tiered Storage), but risky for active segments.

### Key Components
-   **Broker Count**: `Max(StorageReq / DiskPerBroker, BandwidthReq / NetworkPerBroker)`.
-   **Partition Count**: `Throughput / MaxThroughputPerPartition`.
