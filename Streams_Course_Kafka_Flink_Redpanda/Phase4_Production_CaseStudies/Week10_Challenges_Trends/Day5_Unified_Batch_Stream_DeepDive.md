# Day 5: Unified Batch/Stream - Deep Dive

## Deep Dive & Internals

### Batch Execution Mode in Flink
When `RuntimeMode.BATCH` is enabled:
1.  **Blocking Shuffle**: Tasks write all output to disk, then next stage reads. (Recoverable).
2.  **Sort-Merge Join**: Sort both inputs, then merge. Faster than Hash Join for huge data.
3.  **No Watermarks**: Not needed. We have all data.

### The "Streaming First" Mindset
Design for Streaming. Run as Batch for backfill.
-   **Windowing**: Works in both.
-   **State**: In Batch, state is just "Spill to Disk". In Stream, it's RocksDB.

### Iceberg / Hudi / Delta
The "Lakehouse" enables Kappa Architecture on S3.
-   **Stream**: Flink writes to Iceberg.
-   **Batch**: Spark/Trino reads from Iceberg.
-   **Upserts**: These formats support `UPDATE/DELETE` on S3.

### Performance Implications
-   **Latency**: Streaming engine overhead is higher than pure Batch engine (Spark) for massive historical loads.
-   **Optimization**: Flink's Batch scheduler is getting better, but Spark is still king of pure Batch ETL.
