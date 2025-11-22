# Flink Monitoring

## Core Concepts

### Key Flink Metrics
- numRecordsIn/Out: Throughput
- checkpointDuration: State snapshot time
- buffers.outPoolUsage: Backpressure indicator
- numRestarts: Job stability

### Metric Types
- Counter: Cumulative (records processed)
- Gauge: Current value (buffer usage)
- Histogram: Distribution (latency percentiles)

### Monitoring Stack
- Prometheus Reporter
- Grafana Dashboards
- Flink Web UI
