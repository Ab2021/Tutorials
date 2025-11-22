# Day 4: Tiered Storage

## Core Concepts & Theory

### The Problem
Storing PBs of data on local NVMe SSDs is expensive and limits scalability (adding brokers just for disk space).

### The Solution: Tiered Storage
Offload old data to Object Storage (S3/GCS).
-   **Hot Data**: Local Disk (Fast).
-   **Cold Data**: S3 (Cheap, Infinite).
-   **Transparent**: Consumers don't know the difference. The broker fetches from S3 automatically if needed.

### Architectural Reasoning
**Decoupling Compute and Storage**
-   **Stateless Brokers**: If a broker fails, it doesn't need to recover TBs of data. It just connects to S3.
-   **Elasticity**: Scale brokers up/down for *throughput* (CPU/Network), not for *storage*.

### Key Components
-   **Local Cache**: To speed up reads of recently accessed cold data.
-   **Upload Policy**: When to move data to S3 (e.g., after 1 hour).
