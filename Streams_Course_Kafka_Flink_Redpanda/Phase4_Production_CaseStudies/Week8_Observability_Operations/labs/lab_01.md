# Lab 01: Kafka JMX Metrics Export

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
- Export Kafka JMX metrics
- Configure JMX Exporter
- View metrics in Prometheus

## Problem Statement
Configure Kafka to export JMX metrics using the JMX Prometheus Exporter. Verify metrics are accessible at the HTTP endpoint.

## Starter Code
```yaml
# docker-compose.yml
services:
  kafka:
    environment:
      KAFKA_OPTS: "-javaagent:/path/to/jmx_exporter.jar=7071:/path/to/config.yml"
```

## Hints
<details>
<summary>Hint 1</summary>
Download JMX Exporter JAR from Maven Central.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```yaml
# docker-compose.yml
services:
  kafka:
    environment:
      KAFKA_OPTS: "-javaagent:/usr/share/jmx_exporter/jmx_prometheus_javaagent.jar=7071:/usr/share/jmx_exporter/kafka-broker.yml"
    volumes:
      - ./jmx_exporter:/usr/share/jmx_exporter
```

```bash
# Verify metrics
curl http://localhost:7071/metrics | grep kafka_server

# Expected output: kafka_server_BrokerTopicMetrics_MessagesInPerSec
```
</details>
