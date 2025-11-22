# Day 1: Monitoring Kafka & Redpanda

## Core Concepts & Theory

### Key Metrics to Monitor
**Broker Metrics:**
- `MessagesInPerSec`: Incoming message rate
- `BytesInPerSec` / `BytesOutPerSec`: Network throughput
- `UnderReplicatedPartitions`: Partitions not fully replicated (critical!)
- `OfflinePartitionsCount`: Partitions with no leader
- `ActiveControllerCount`: Should be 1 (0 or >1 indicates problem)

**Topic Metrics:**
- `BytesInPerSec` per topic
- `MessagesInPerSec` per topic
- `TotalFetchRequestsPerSec`
- `TotalProduceRequestsPerSec`

**Consumer Metrics:**
- `records-lag-max`: Maximum lag across all partitions
- `records-consumed-rate`: Consumption rate
- `fetch-latency-avg`: Time to fetch data

### Architectural Reasoning
**Why Monitor UnderReplicatedPartitions?**
This is the most critical metric. It indicates:
- Broker is down or slow
- Network issues
- Disk I/O saturation
If this stays > 0 for extended periods, you risk data loss.

### Redpanda-Specific Metrics
- `vectorized_reactor_utilization`: CPU utilization per core
- `vectorized_storage_log_compacted_segment`: Compaction activity
- `vectorized_kafka_rpc_active_connections`: Active client connections

### Key Components
- **JMX Exporter**: Exposes Kafka metrics as Prometheus format
- **Redpanda Admin API**: REST API at `:9644/metrics`
- **Burrow**: LinkedIn's consumer lag monitoring tool
