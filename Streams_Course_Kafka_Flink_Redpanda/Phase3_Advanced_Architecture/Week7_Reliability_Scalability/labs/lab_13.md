# Lab 13: Chaos Engineering

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
- Implement chaos experiments
- Test system resilience
- Validate recovery procedures

## Problem Statement
Use Chaos Mesh or manual methods to inject failures: kill a Kafka broker, introduce network latency, and fill disk space. Verify that the system recovers gracefully and no data is lost.

## Starter Code
```bash
# Kill broker
docker kill kafka-1

# TODO: Verify system continues operating
```

## Hints
<details>
<summary>Hint 1</summary>
Monitor under-replicated partitions during failures.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Experiment 1: Broker Failure**
```bash
# Baseline
kafka-topics --bootstrap-server localhost:9092 --describe --topic test-topic

# Kill one broker (assuming 3 brokers)
docker kill kafka-1

# Verify: System should continue
kafka-console-producer --bootstrap-server localhost:9092 --topic test-topic
# Should still work (connects to kafka-2 or kafka-3)

# Check under-replicated partitions
kafka-topics --bootstrap-server localhost:9092 --describe --under-replicated-partitions

# Restart broker
docker start kafka-1

# Verify: Partitions re-sync
# Under-replicated count should return to 0
```

**Experiment 2: Network Partition**
```bash
# Install tc (traffic control)
# Introduce 500ms latency
docker exec kafka-1 tc qdisc add dev eth0 root netem delay 500ms

# Measure impact on producer latency
# Expected: Increased latency but no failures

# Remove latency
docker exec kafka-1 tc qdisc del dev eth0 root
```

**Experiment 3: Disk Full**
```bash
# Fill disk (use with caution!)
docker exec kafka-1 dd if=/dev/zero of=/tmp/bigfile bs=1M count=10000

# Monitor broker logs
docker logs kafka-1

# Expected: Broker stops accepting writes
# Other brokers take over

# Cleanup
docker exec kafka-1 rm /tmp/bigfile
```

**Verification Checklist:**
- ✅ No message loss during broker failure
- ✅ Producer/Consumer automatic failover
- ✅ Partitions re-replicate after recovery
- ✅ Latency degrades gracefully under network issues
- ✅ Alerts fire for under-replicated partitions
</details>
