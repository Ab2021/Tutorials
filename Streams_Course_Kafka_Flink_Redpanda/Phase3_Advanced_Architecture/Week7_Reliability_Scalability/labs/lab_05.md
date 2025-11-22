# Lab 05: MirrorMaker 2 Setup

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
- Configure MirrorMaker 2
- Replicate topics across clusters
- Understand offset translation

## Problem Statement
Set up MirrorMaker 2 to replicate a topic from `source-cluster` to `target-cluster`. Verify that messages are replicated and consumer offsets are translated correctly.

## Starter Code
```properties
# mm2.properties
clusters = source, target
source.bootstrap.servers = localhost:9092
target.bootstrap.servers = localhost:9093

# Replication flows
source->target.enabled = true
source->target.topics = test-topic

# TODO: Add checkpoint and heartbeat connectors
```

## Hints
<details>
<summary>Hint 1</summary>
MirrorMaker 2 runs as a Kafka Connect cluster.
</details>

<details>
<summary>Hint 2</summary>
Replicated topics are prefixed with the source cluster name (e.g., `source.test-topic`).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**mm2.properties:**
```properties
# Cluster definitions
clusters = source, target
source.bootstrap.servers = localhost:9092
target.bootstrap.servers = localhost:9093

# Replication flow
source->target.enabled = true
source->target.topics = .*

# Source connector
source->target.emit.checkpoints.enabled = true
source->target.emit.checkpoints.interval.seconds = 60

# Heartbeat
source->target.emit.heartbeats.enabled = true
source->target.heartbeats.topic.replication.factor = 1

# Checkpoint
source->target.checkpoints.topic.replication.factor = 1

# Sync topic configs
sync.topic.configs.enabled = true
```

**Start MirrorMaker 2:**
```bash
connect-mirror-maker mm2.properties
```

**Verify Replication:**
```bash
# List topics on target cluster
kafka-topics --bootstrap-server localhost:9093 --list

# Expected: source.test-topic

# Consume from replicated topic
kafka-console-consumer --bootstrap-server localhost:9093 \
  --topic source.test-topic --from-beginning
```

**Offset Translation:**
```bash
# Consumer group offsets are automatically translated
# Check translated offsets
kafka-consumer-groups --bootstrap-server localhost:9093 \
  --describe --group my-group
```
</details>
