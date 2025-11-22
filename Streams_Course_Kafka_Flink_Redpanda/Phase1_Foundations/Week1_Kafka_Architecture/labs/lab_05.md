# Lab 05: ISR & Min.Insync.Replicas

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Understand durability guarantees.
-   Simulate broker failures.
-   Observe the effect of `acks=all` and `min.insync.replicas`.

## Problem Statement
1.  Create a topic `critical-data` with replication factor 3 and `min.insync.replicas=2`.
2.  Start a producer with `acks='all'`.
3.  Stop 2 out of 3 brokers (leaving only 1 alive).
4.  Try to produce a message.
5.  **Expected Result**: The producer should fail with `NOT_ENOUGH_REPLICAS`.

## Starter Code
```python
# Producer Configuration
conf = {
    'bootstrap.servers': 'localhost:9092',
    'acks': 'all',
    'retries': 3
}
```

## Hints
<details>
<summary>Hint 1</summary>
To stop a broker in Docker Compose: `docker-compose stop kafka3`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Create Topic
```bash
docker exec kafka1 kafka-topics --create --topic critical-data --partitions 1 --replication-factor 3 --bootstrap-server localhost:9092 --config min.insync.replicas=2
```

### Step 2: Python Producer Script
```python
from confluent_kafka import Producer, KafkaError

def produce_critical():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'acks': 'all',  # Wait for all ISRs
        'retries': 0    # Fail fast for this demo
    }
    p = Producer(conf)

    def delivery_report(err, msg):
        if err is not None:
            print(f'Delivery failed: {err}')
        else:
            print(f'Delivered to {msg.topic()}')

    p.produce('critical-data', b'important payload', callback=delivery_report)
    p.flush()

if __name__ == "__main__":
    produce_critical()
```

### Step 3: Test
1.  Run script -> Success.
2.  `docker-compose stop kafka3` -> Run script -> Success (2 replicas alive >= min 2).
3.  `docker-compose stop kafka2` -> Run script -> **Failure** (1 replica alive < min 2).
    -   Error: `KafkaError{code=NOT_ENOUGH_REPLICAS,val=19,str="Broker: Not enough in-sync replicas"}`
</details>
