# Day 1: Monitoring Kafka - Deep Dive

## Deep Dive & Internals

### JMX vs Native Metrics
-   **Kafka (JVM)**: Uses JMX (Java Management Extensions). Heavy. Requires `jmx_exporter` sidecar/agent to convert to Prometheus format.
    -   *Overhead*: JMX scraping can be CPU intensive. Don't scrape too often (< 15s).
-   **Redpanda (C++)**: Native Prometheus endpoint (`:9644/metrics`). Zero overhead.

### Histogram Pitfalls
Kafka's JMX histograms (e.g., `99thPercentile`) are calculated *inside* the broker using a decaying reservoir.
-   **Pros**: Pre-calculated.
-   **Cons**: Can be misleading if traffic is bursty.
-   **Best Practice**: Export raw histograms (buckets) to Prometheus and calculate percentiles there (`histogram_quantile`).

### Consumer Lag Monitoring
Lag is the difference between `LogEndOffset` (Broker) and `CurrentOffset` (Consumer).
-   **Internal**: Consumers send offset commits to `__consumer_offsets`.
-   **External Monitoring (Burrow)**: Look at the *rate* of commits vs *rate* of production.
    -   If `ProductionRate > ConsumptionRate`, Lag grows.
    -   If `Lag` is stable but high, it's just latency.
    -   If `Lag` is growing, it's an incident.

### Advanced Reasoning
**The "Stale Metric" Trap**
If a broker crashes, the Prometheus exporter might stop sending data.
-   If you alert on `UnderReplicatedPartitions > 0`, and the metric *disappears*, the alert resolves!
-   **Fix**: Always alert on `up == 0` (Target down) or `absent(metric)`.

### Performance Implications
-   **Cardinality**: Kafka creates metrics *per partition*.
    -   100 topics * 50 partitions = 5,000 metrics.
    -   Multiply by 10 metric types = 50,000 time series.
    -   **Fix**: Use JMX Exporter regex to aggregate metrics (remove `partition` label) unless debugging.
