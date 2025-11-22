# Day 1: Monitoring Kafka & Redpanda - Deep Dive

## Deep Dive & Internals

### JMX Architecture
Kafka exposes metrics via JMX (Java Management Extensions).
- **MBeans**: Managed Beans representing metrics
- **ObjectName**: Hierarchical naming (e.g., `kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec`)
- **JMX Exporter**: Translates JMX to Prometheus format

### Redpanda Metrics Pipeline
Redpanda uses Seastar framework metrics:
- **Prometheus Format**: Native support, no JMX needed
- **Per-Core Metrics**: Redpanda is thread-per-core, metrics are per-shard
- **Admin API**: `/metrics` endpoint, `/public_metrics` for external monitoring

### Advanced Reasoning
**Metric Cardinality Problem**
Kafka has thousands of metrics. Exporting all of them to Prometheus causes:
- High memory usage in Prometheus
- Slow queries
**Solution**: Use JMX Exporter rules to filter and aggregate metrics.

### Performance Implications
- **Metric Collection Overhead**: Minimal (<1% CPU) if done correctly
- **Scrape Interval**: 15-30 seconds is standard. Faster = more load.
