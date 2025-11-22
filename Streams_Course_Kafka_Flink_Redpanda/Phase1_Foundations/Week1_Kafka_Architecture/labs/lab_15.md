# Lab 15: Multi-Broker Setup (Manual)

## Difficulty
🔴 Hard

## Estimated Time
90 mins

## Learning Objectives
-   Understand `server.properties`.
-   Run Kafka without Docker (simulating bare metal).

## Problem Statement
Download the Kafka binary (tgz). Create 3 copies of `config/server.properties`:
-   Broker 0: Port 9092, LogDir /tmp/kafka-logs-0
-   Broker 1: Port 9093, LogDir /tmp/kafka-logs-1
-   Broker 2: Port 9094, LogDir /tmp/kafka-logs-2
Start Zookeeper and all 3 brokers manually in separate terminals. Create a replicated topic and verify it works.

## Starter Code
```properties
# server-0.properties
broker.id=0
listeners=PLAINTEXT://:9092
log.dirs=/tmp/kafka-logs-0
zookeeper.connect=localhost:2181
```

## Hints
<details>
<summary>Hint 1</summary>
Make sure `log.dirs` are unique for each broker, otherwise they will lock the same files.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Config Files
**server-1.properties**
```properties
broker.id=1
listeners=PLAINTEXT://:9093
log.dirs=/tmp/kafka-logs-1
zookeeper.connect=localhost:2181
```
(Repeat for others with unique IDs and ports).

### Step 2: Start Zookeeper
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

### Step 3: Start Brokers
```bash
bin/kafka-server-start.sh config/server-0.properties &
bin/kafka-server-start.sh config/server-1.properties &
bin/kafka-server-start.sh config/server-2.properties &
```

### Step 4: Verify
```bash
bin/kafka-topics.sh --create --topic manual-test --partitions 3 --replication-factor 3 --bootstrap-server localhost:9092
```
</details>
