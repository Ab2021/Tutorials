# Day 2: IoT Telemetry - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you handle 1 million unique keys (Cars) in Flink?**
    -   *A*: Flink scales horizontally. 1M keys is fine. State is sharded. Ensure `maxParallelism` is high enough (e.g., 4096) to allow future scaling.

2.  **Q: How do you detect if a device has stopped sending data?**
    -   *A*: **ProcessFunction** with a Timer. Reset timer on every event. If timer fires, emit "Device Offline" alert.

3.  **Q: Why use a Time Series Database (TSDB) instead of Postgres?**
    -   *A*: TSDBs (Influx, Timescale) are optimized for high write throughput and time-range queries. They compress data much better.

### Production Challenges
-   **Challenge**: **Thundering Herd**.
    -   *Scenario*: Network outage recovers. 1M devices reconnect and upload buffered data simultaneously.
    -   *Fix*: Kafka handles the spike. Flink might lag. Enable Backpressure. Ensure Autoscaling is not too aggressive (don't scale up for a temporary spike).

-   **Challenge**: **Clock Drift**.
    -   *Scenario*: Device clock is wrong (Year 1970).
    -   *Fix*: Filter invalid timestamps at ingestion. Use NTP on devices.

### Troubleshooting Scenarios
**Scenario**: Kafka topic retention is too short for the outage duration.
-   *Cause*: Devices were offline for 7 days. Kafka retention is 3 days. Data lost.
-   *Fix*: Tiered Storage (Infinite Retention).
