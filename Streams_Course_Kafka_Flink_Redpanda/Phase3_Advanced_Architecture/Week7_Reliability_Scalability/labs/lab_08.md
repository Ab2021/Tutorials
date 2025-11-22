# Lab 08: Flink Metrics & Reporters

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Configure Flink metrics reporters
- Export to Prometheus
- Monitor job health

## Problem Statement
Configure a Flink job to export metrics to Prometheus. Monitor key metrics like `numRecordsIn`, `numRecordsOut`, and `checkpointDuration`.

## Starter Code
```yaml
# flink-conf.yaml
metrics.reporters: prom
metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
metrics.reporter.prom.port: 9250-9260
```

## Hints
<details>
<summary>Hint 1</summary>
Add the Prometheus reporter JAR to Flink's lib directory.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**flink-conf.yaml:**
```yaml
metrics.reporters: prom
metrics.reporter.prom.factory.class: org.apache.flink.metrics.prometheus.PrometheusReporterFactory
metrics.reporter.prom.port: 9250-9260
```

**Download Prometheus Reporter:**
```bash
wget https://repo.maven.apache.org/maven2/org/apache/flink/flink-metrics-prometheus/1.18.0/flink-metrics-prometheus-1.18.0.jar
cp flink-metrics-prometheus-1.18.0.jar $FLINK_HOME/lib/
```

**Prometheus Config:**
```yaml
scrape_configs:
  - job_name: 'flink'
    static_configs:
      - targets: ['localhost:9250', 'localhost:9251']
```

**Key Metrics to Monitor:**
```promql
# Records processed
flink_taskmanager_job_task_operator_numRecordsIn

# Checkpoint duration
flink_jobmanager_job_lastCheckpointDuration

# Backpressure
flink_taskmanager_job_task_buffers_outPoolUsage
```

**Grafana Dashboard:**
Import dashboard ID: 10369 (Flink Dashboard)
</details>
