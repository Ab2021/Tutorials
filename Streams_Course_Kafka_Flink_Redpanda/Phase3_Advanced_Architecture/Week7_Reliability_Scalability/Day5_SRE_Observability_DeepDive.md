# Day 5: SRE - Deep Dive

## Deep Dive & Internals

### Distributed Tracing
Tracing a message from Producer -> Kafka -> Flink -> Sink.
-   **Context Propagation**: Injecting `traceparent` header into the Kafka message.
-   **Span**: Each hop creates a span.
-   **Sampling**: Tracing every message is expensive. Sample 0.1%.

### Kafka Lag Monitoring
-   **Burrow**: LinkedIn's tool. Checks if consumer is committing offsets but falling behind.
-   **Flink Metrics**: `records-lag-max`.

### Advanced Reasoning
**Alert Fatigue**
Don't alert on "CPU > 80%". Alert on "SLO Breach" (User is impacted).
-   **Burn Rate**: How fast are we consuming the Error Budget? Alert if we will run out of budget in 4 hours.

### Performance Implications
-   **Metric Cardinality**: Avoid metrics with high cardinality tags (e.g., `user_id`, `transaction_id`). It will kill Prometheus.
