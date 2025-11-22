# Lab 14: Performance Benchmarking

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Benchmark Kafka throughput
- Measure latency percentiles
- Optimize configuration

## Problem Statement
Use `kafka-producer-perf-test` and `kafka-consumer-perf-test` to benchmark your Kafka cluster. Measure throughput (MB/sec) and latency (P50, P99, P999). Tune configurations to improve performance.

## Starter Code
```bash
kafka-producer-perf-test \
  --topic perf-test \
  --num-records 1000000 \
  --record-size 1024 \
  --throughput -1 \
  --producer-props bootstrap.servers=localhost:9092
```

## Hints
<details>
<summary>Hint 1</summary>
Increase `batch.size` and `linger.ms` for higher throughput.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Producer Benchmark:**
```bash
kafka-producer-perf-test \
  --topic perf-test \
  --num-records 10000000 \
  --record-size 1024 \
  --throughput -1 \
  --producer-props \
    bootstrap.servers=localhost:9092 \
    acks=1 \
    batch.size=32768 \
    linger.ms=10 \
    compression.type=lz4
```

**Sample Output:**
```
500000 records sent, 99800.4 records/sec (97.46 MB/sec)
1000000 records sent, 100000.0 records/sec (97.66 MB/sec)
...
10000000 records sent in 100.5 sec
Throughput: 99502.49 records/sec (97.17 MB/sec)
Latency: avg=5.2ms, p50=3ms, p99=25ms, p999=45ms
```

**Consumer Benchmark:**
```bash
kafka-consumer-perf-test \
  --bootstrap-server localhost:9092 \
  --topic perf-test \
  --messages 10000000 \
  --threads 1
```

**Sample Output:**
```
10000000 records consumed in 95.3 sec
Throughput: 104932.0 records/sec (102.47 MB/sec)
```

**Tuning Tips:**
```properties
# Producer
batch.size=65536
linger.ms=20
compression.type=zstd
buffer.memory=67108864

# Broker
num.network.threads=8
num.io.threads=16
socket.send.buffer.bytes=1048576
socket.receive.buffer.bytes=1048576

# Consumer
fetch.min.bytes=1048576
fetch.max.wait.ms=500
```

**Comparison:**
| Config | Throughput | P99 Latency |
|--------|------------|-------------|
| Default | 50 MB/sec | 50ms |
| Tuned | 100 MB/sec | 25ms |
</details>
