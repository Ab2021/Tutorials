# Day 3: Multi-DC Replication

## Core Concepts & Theory

### Disaster Recovery (DR)
What if the entire Data Center (AWS Region) goes down?
-   **Active-Passive**: Write to DC1. Replicate to DC2. If DC1 dies, switch to DC2.
-   **Active-Active**: Write to DC1 and DC2. Replicate bi-directionally. (Complex conflict resolution).

### Replication Tools
-   **MirrorMaker 2 (MM2)**: Connect-based. Replicates topics, ACLs, configs.
-   **Confluent Replicator**: Commercial tool.
-   **Redpanda Remote Read Replica**: Native replication.

### Architectural Reasoning
**Offset Translation**
Offsets are not identical across clusters (DC1 offset 100 != DC2 offset 100).
-   MM2 emits checkpoints for the *downstream* cluster so consumers can resume seamlessly.
-   **Timestamp preservation**: Crucial for time-based lookups.

### Key Components
-   **Heartbeats**: To detect cluster health.
-   **Cluster Linking**: Protocol for replication.
