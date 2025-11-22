# Lab 12: Disaster Recovery Testing

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
- Test failover scenarios
- Verify data integrity after recovery
- Measure RTO and RPO

## Problem Statement
Simulate a datacenter failure by stopping the source Kafka cluster. Verify that MirrorMaker 2 has replicated all data to the target cluster. Failover consumers to the target cluster and measure recovery time.

## Starter Code
```bash
# Stop source cluster
docker-compose stop kafka-source

# TODO: Verify replication lag is zero
# TODO: Failover consumers
```

## Hints
<details>
<summary>Hint 1</summary>
Check MirrorMaker 2 checkpoint topic for offset translation.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Pre-Failure Setup:**
```bash
# Produce to source
kafka-console-producer --bootstrap-server localhost:9092 \
  --topic test-topic

# Verify replication
kafka-console-consumer --bootstrap-server localhost:9093 \
  --topic source.test-topic --from-beginning
```

**Simulate Failure:**
```bash
# Record current offset
SOURCE_OFFSET=$(kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group my-group | grep test-topic | awk '{print $3}')

echo \"Source offset before failure: $SOURCE_OFFSET\"

# Stop source cluster
docker-compose stop kafka-source

# Wait for detection (heartbeat timeout)
sleep 30
```

**Failover:**
```bash
# Check replicated offset on target
TARGET_OFFSET=$(kafka-console-consumer --bootstrap-server localhost:9093 \
  --topic source.test-topic --from-beginning --max-messages 1000 | wc -l)

echo \"Target offset: $TARGET_OFFSET\"

# Calculate RPO (data loss)
RPO=$((SOURCE_OFFSET - TARGET_OFFSET))
echo \"RPO: $RPO messages lost\"

# Reconfigure consumers to target cluster
# Update bootstrap.servers to localhost:9093
# Update topic name to source.test-topic

# Measure RTO (time to recover)
START_TIME=$(date +%s)
# ... consumer starts processing ...
END_TIME=$(date +%s)
RTO=$((END_TIME - START_TIME))
echo \"RTO: $RTO seconds\"
```

**Verification:**
```bash
# Verify no data loss (RPO = 0)
if [ $RPO -eq 0 ]; then
  echo \"✅ Zero data loss\"
else
  echo \"❌ Lost $RPO messages\"
fi

# Verify RTO meets SLA (e.g., < 5 minutes)
if [ $RTO -lt 300 ]; then
  echo \"✅ RTO within SLA\"
else
  echo \"❌ RTO exceeded SLA\"
fi
```
</details>
