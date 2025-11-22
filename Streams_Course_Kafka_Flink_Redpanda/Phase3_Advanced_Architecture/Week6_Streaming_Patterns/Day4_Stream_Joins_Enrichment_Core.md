# Day 4: Stream Joins & Enrichment

## Core Concepts & Theory

### Join Types
1.  **Stream-Stream Join**: Windowed Join or Interval Join. (e.g., AdClick JOIN AdImpression).
2.  **Stream-Table Join (Enrichment)**:
    -   **Lookup Join**: Query external DB for every record. (Slow).
    -   **Temporal Join**: Join with a versioned table in Flink state. (Fast).

### Enrichment Patterns
-   **Async I/O**: Use `AsyncDataStream` to query a remote service (User Profile Service) without blocking.
-   **Broadcast State**: Broadcast the "Dimension Table" (e.g., Currency Rates) to all nodes. Local lookup.

### Architectural Reasoning
**Latency vs Freshness**
-   **Broadcast**: Zero latency, but data might be stale (eventual consistency).
-   **Async I/O**: High latency (network RTT), but data is fresh.
-   **Temporal Join**: Best of both. High throughput, consistent point-in-time semantics.

### Key Components
-   `AsyncFunction`
-   `BroadcastProcessFunction`
-   `TemporalTableFunction`
