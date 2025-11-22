# Lab 07: Prometheus Metrics Export

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Export Kafka metrics to Prometheus
- Configure JMX Exporter
- Create Grafana dashboards

## Problem Statement
Configure Kafka to export JMX metrics to Prometheus. Set up a Prometheus server to scrape these metrics and visualize them in Grafana.

## Starter Code
```yaml
# docker-compose.yml
services:
  kafka:
    environment:
      KAFKA_JMX_PORT: 9999
      KAFKA_JMX_HOSTNAME: localhost
```

## Hints
<details>
<summary>Hint 1</summary>
Use the JMX Exporter Java agent to expose metrics on an HTTP endpoint.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**docker-compose.yml:**
```yaml
version: '3'
services:
  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - \"9092:9092\"
      - \"7071:7071\"  # JMX Exporter
    environment:
      KAFKA_OPTS: \"-javaagent:/usr/share/jmx_exporter/jmx_prometheus_javaagent.jar=7071:/usr/share/jmx_exporter/kafka-broker.yml\"
    volumes:
      - ./jmx_exporter:/usr/share/jmx_exporter
  
  prometheus:
    image: prom/prometheus
    ports:
      - \"9090:9090\"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - \"3000:3000\"
```

**prometheus.yml:**
```yaml
scrape_configs:
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:7071']
```

**kafka-broker.yml (JMX rules):**
```yaml
lowercaseOutputName: true
rules:
  - pattern: kafka.server<type=(.+), name=(.+)><>Value
    name: kafka_server_$1_$2
```

**Verify:**
```bash
# Check metrics endpoint
curl http://localhost:7071/metrics

# Access Prometheus
open http://localhost:9090

# Query: kafka_server_BrokerTopicMetrics_MessagesInPerSec
```
</details>
