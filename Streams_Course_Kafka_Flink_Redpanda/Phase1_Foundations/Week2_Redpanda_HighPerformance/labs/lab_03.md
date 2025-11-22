# Lab 03: Redpanda vs Kafka Benchmarking

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Measure throughput using `rpk`'s built-in benchmark tool.
-   Compare performance (if you have a Kafka cluster running).

## Problem Statement
Run a producer benchmark against your Redpanda node.
-   Message Size: 1KB
-   Count: 100,000 messages
-   Compression: LZ4

## Starter Code
```bash
rpk topic produce --help # Look for benchmark flags? 
# Actually, rpk has a specific plugin or you use kafka-producer-perf-test
```
*Correction*: `rpk` has a simplified interface, but often we use the standard `kafka-producer-perf-test` tool which comes with Redpanda too.

## Hints
<details>
<summary>Hint 1</summary>
Try `rpk redpanda tune` (if on Linux) to see tuning options. For benchmarking, use `kafka-producer-perf-test.sh` inside the container.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
docker exec -it redpanda /bin/bash

# Inside container
kafka-producer-perf-test.sh   --topic bench-test   --num-records 100000   --record-size 1024   --throughput -1   --producer-props bootstrap.servers=localhost:9092 compression.type=lz4
```
</details>
