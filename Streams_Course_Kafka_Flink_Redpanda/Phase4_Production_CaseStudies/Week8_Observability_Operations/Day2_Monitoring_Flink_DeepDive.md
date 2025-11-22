# Flink Monitoring - Deep Dive

## Internals

### Metric Reporters
Flink supports multiple reporters simultaneously:
- JMX (local debugging)
- Prometheus (production)
- Datadog, InfluxDB, etc.

### Metric Scopes
- Job-level: Aggregated across all tasks
- Task-level: Per parallel instance
- Operator-level: Per transformation

### Backpressure Detection
Flink 1.13+ uses task sampling instead of metrics for zero overhead.
