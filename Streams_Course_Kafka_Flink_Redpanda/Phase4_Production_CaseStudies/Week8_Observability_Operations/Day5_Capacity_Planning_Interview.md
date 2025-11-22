# Day 5: Capacity Planning - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How many partitions should I create for a topic?**
    -   *A*: `Max(TargetThroughput / ProducerThroughput, TargetThroughput / ConsumerThroughput)`. Also consider future growth.

2.  **Q: Why is "Zero Copy" important for capacity?**
    -   *A*: It allows the broker to send data from Disk to Network without copying it into User Space (RAM). Reduces CPU usage and Context Switches.

3.  **Q: How do you handle a sudden 10x traffic spike?**
    -   *A*: Kafka buffers it on disk. Latency might increase, but data is safe. Consumers will lag. To fix: Scale consumers or throttle producers.

### Production Challenges
-   **Challenge**: **Disk Full Outage**.
    -   *Scenario*: Disk fills up. Broker crashes.
    -   *Fix*: Set `log.retention.bytes`. Use Tiered Storage. Monitor disk usage.

-   **Challenge**: **Noisy Neighbor**.
    -   *Scenario*: One topic consumes all network bandwidth.
    -   *Fix*: Use **Quotas** (Network bandwidth quotas) to limit throughput per client/user.

### Troubleshooting Scenarios
**Scenario**: High I/O Wait time on Linux.
-   *Cause*: Disk saturation.
-   *Fix*: Check if consumers are reading old data (Page Cache miss). Add more disks or brokers.
