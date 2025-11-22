# Day 1: Flink SQL - Deep Dive

## Deep Dive & Internals

### Changelog Stream Types
1.  **Append-Only**: Only INSERTs. (e.g., Log files).
2.  **Retract**: INSERT and DELETE. (e.g., `COUNT` decreases when a record leaves a window).
3.  **Upsert**: INSERT and UPDATE. (e.g., Database CDC).

### Joins in SQL
-   **Regular Join**: Keeps ALL history of both sides in state. Infinite state growth!
-   **Interval Join**: `A.time BETWEEN B.time - 1h AND B.time`. State is cleaned up after 1h.
-   **Temporal Table Join**: Join with a "versioned table" (e.g., Currency Rates at time T).

### Advanced Reasoning
**Mini-Batch Aggregation**
By default, Flink SQL updates the result for *every* input row. This causes high I/O.
-   **Mini-Batch**: Buffer inputs for 500ms, compute aggregate in memory, send 1 update. Drastically reduces state access.

### Performance Implications
-   **State Size**: `GROUP BY` on high-cardinality keys (User ID) without windows will grow state forever. Use `Idle State Retention` config to clean up old keys.
