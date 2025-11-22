# Day 1: Monitoring - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you distinguish between a slow consumer and a stuck consumer?**
    -   *A*: **Slow Consumer**: Offsets are advancing, but Lag is increasing (Production > Consumption). **Stuck Consumer**: Offsets are NOT advancing at all.

2.  **Q: Why is `NetworkProcessorAvgIdlePercent` important?**
    -   *A*: It measures how much time network threads are idle. If it's 0%, the broker is network-bound (cannot accept new connections/requests), even if CPU is low.

3.  **Q: What is the impact of scraping JMX metrics too frequently?**
    -   *A*: JMX is a blocking operation in some JVM versions. High scrape frequency can cause GC pauses or slow down the broker.

### Production Challenges
-   **Challenge**: **Metric Explosion in Prometheus**.
    -   *Scenario*: Dev team creates 10,000 dynamic topics. Prometheus OOMs.
    -   *Fix*: Whitelist specific metrics in JMX Exporter. Drop partition-level metrics for non-critical topics.

-   **Challenge**: **False Positives on Lag**.
    -   *Scenario*: Batch job runs every hour. Lag spikes to 1M, then drops. Alert wakes you up.
    -   *Fix*: Alert on *Lag Duration* (Lag > 1M for > 10 mins) or *Derivative* (Lag is increasing fast).

### Troubleshooting Scenarios
**Scenario**: Broker is up, but `OfflinePartitions` > 0.
-   *Cause*: The disk containing those partitions might be corrupted or read-only (I/O error).
-   *Fix*: Check `server.log` for I/O errors. Replace disk.
