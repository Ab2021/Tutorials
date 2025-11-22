# Day 3: Multi-DC - Deep Dive

## Deep Dive & Internals

### Active-Active Conflicts
User updates Profile in DC1 (Name=A) and DC2 (Name=B) at the same time.
-   **Last Write Wins (LWW)**: Based on timestamp.
-   **CRDTs**: Conflict-free Replicated Data Types (merge logic).
-   **Sticky Routing**: Route User X always to DC1 to avoid conflicts.

### MirrorMaker 2 Architecture
-   **Source Connector**: Reads from DC1.
-   **Sink Connector**: Writes to DC2.
-   **Checkpoint Connector**: Translates consumer group offsets.
-   **Heartbeat Connector**: Monitoring.

### Advanced Reasoning
**Stretch Cluster vs Replication**
-   **Stretch Cluster**: One Kafka cluster spanning 3 AZs (Availability Zones). Synchronous replication. Zero RPO. High latency.
-   **Async Replication**: Two separate clusters. Asynchronous. Non-zero RPO. Low latency.

### Performance Implications
-   **Bandwidth**: Cross-region traffic is expensive ($$$). Compress data (Zstd) before replicating.
