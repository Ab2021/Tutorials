# Day 2: IoT Telemetry - Deep Dive

## Deep Dive & Internals

### Handling Out-of-Order Data
IoT devices often lose connectivity and upload buffered data later.
-   **Event Time**: MUST use device timestamp, not ingestion timestamp.
-   **Watermarks**: Allow for significant lateness (e.g., 1 hour).
-   **Allowed Lateness**: Update previous windows if data arrives late.

### Geofencing Implementation
**Task**: Alert if a car enters a restricted zone.
-   **Naive**: Check every point against every polygon. (O(N*M) - Slow).
-   **Optimized**:
    -   **Spatial Indexing**: Use S2 Geometry or H3 (Uber) to map Lat/Lon to a Cell ID.
    -   **KeyBy**: Cell ID.
    -   **Broadcast**: Broadcast active Geofences to all tasks.

### Compression
IoT data is repetitive.
-   **Delta Encoding**: Store difference from previous value.
-   **Gorilla Compression**: Specialized float compression (used in Prometheus/InfluxDB).
-   **Kafka**: Use Zstd. It works great on JSON/Avro IoT data.

### Performance Implications
-   **Write Load**: 1M writes/sec is heavy for any DB.
-   **Solution**: Flink aggregates (1-min avg) reduce volume by 60x before writing to DB.
