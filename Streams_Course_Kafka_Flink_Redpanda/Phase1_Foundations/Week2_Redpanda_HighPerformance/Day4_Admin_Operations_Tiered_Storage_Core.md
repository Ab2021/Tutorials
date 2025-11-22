# Day 4: Admin Operations & Tiered Storage

## Core Concepts & Theory

### Tiered Storage (Shadow Indexing)
Tiered Storage allows you to offload older log segments to object storage (S3, GCS).
-   **Hot Set**: Recent data stays on local NVMe SSD (fast).
-   **Cold Set**: Old data moves to S3 (cheap, infinite).
-   **Transparent**: Consumers don't know the difference. They just request offset 0, and the broker fetches it from S3.

### Partition Rebalancing
Moving partitions between brokers to balance load.
-   **Data Movement**: Heavy operation. Uses network bandwidth.
-   **Throttling**: Crucial to avoid starving production traffic during rebalance.

### Architectural Reasoning
**Why Tiered Storage?**
It decouples **Compute** (Brokers/CPU) from **Storage** (Disk).
-   Without Tiered Storage: To store more data, you need more brokers (expensive).
-   With Tiered Storage: You just pay for S3. You can have a small cluster storing PBs of data.

### Key Components
-   **Remote Write**: Uploading segments to S3.
-   **Remote Read**: Fetching segments from S3.
-   **Cache**: Local disk cache for recently accessed remote segments.
