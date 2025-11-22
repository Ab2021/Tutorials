# Day 1: The Log Abstraction - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between a Queue and a Log?**
    -   *A*: A Queue (like RabbitMQ) is transient; messages are deleted once consumed. A Log (like Kafka) is durable; messages are persisted and can be replayed.

2.  **Q: Why is sequential I/O faster than random I/O?**
    -   *A*: Sequential I/O avoids disk seek time (moving the head) and rotational latency. It allows the OS to perform aggressive read-ahead (prefetching).

3.  **Q: Explain the concept of "Log Compaction".**
    -   *A*: Instead of deleting old logs by time, Kafka keeps the *latest* value for each key. This effectively turns the log into a database table snapshot.

### Production Challenges
-   **Challenge**: **Disk Full**.
    -   *Scenario*: Producers write faster than retention policy deletes old segments.
    -   *Fix*: Monitor disk usage. Use tiered storage (S3) to offload old data.

-   **Challenge**: **Slow Consumers**.
    -   *Scenario*: A consumer falls behind and data is deleted before it can be read.
    -   *Fix*: Increase retention time or fix the consumer performance.

### Troubleshooting Scenarios
**Scenario**: You see high I/O wait times on your broker.
-   **Check**: Are you doing random reads? (Consumers reading very old data that is not in page cache).
-   **Check**: Is the disk saturated? (Use `iostat`).
