# Day 4: Stream Joins - Deep Dive

## Deep Dive & Internals

### Temporal Table Join
`SELECT * FROM Orders o JOIN Rates FOR SYSTEM_TIME AS OF o.ts r ON o.currency = r.currency`
-   Flink keeps the history of `Rates` in state.
-   When an Order arrives with `ts=10:00`, Flink looks up the rate *at 10:00*, even if the current time is 10:05.
-   **Prerequisite**: The Rates stream must be a changelog.

### Async I/O Caching
To reduce load on the external DB:
-   **Cache**: Use Guava/Caffeine cache inside the AsyncFunction.
-   **TTL**: Expire cache entries to balance freshness.

### Advanced Reasoning
**Handling Late Data in Joins**
If an Order arrives late (`ts=09:55`), and we only keep Rates for 1 hour, and current time is 11:00, the join might fail or produce incorrect results if state was cleaned up.
-   **Watermarks** drive state cleanup.

### Performance Implications
-   **Lookup Join**: The bottleneck is usually the external DB. Use a high-performance KV store (Redis/Cassandra) and batch requests.
