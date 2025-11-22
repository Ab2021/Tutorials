# Lab 04: Consumer Groups

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand Consumer Groups and parallel consumption.
-   Observe rebalancing behavior.
-   Implement a consumer loop.

## Problem Statement
1.  Write a consumer script that joins a group named `click-processors`.
2.  Run **two instances** of this script in separate terminals.
3.  Produce messages to `user-clicks` (which has 3 partitions).
4.  Observe how partitions are assigned (e.g., Consumer A gets 0,1; Consumer B gets 2).
5.  Kill one consumer and observe the rebalance.

## Starter Code
```python
from confluent_kafka import Consumer

def consume_loop(group_id, topics):
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }
    c = Consumer(conf)
    c.subscribe(topics)
    
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None: continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            print(f"Received message: {msg.value().decode('utf-8')}")
    finally:
        c.close()
```

## Hints
<details>
<summary>Hint 1</summary>
Use `c.subscribe(topics, on_assign=print_assignment, on_revoke=print_revocation)` to visualize when rebalances happen.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Consumer, KafkaError

def print_assignment(consumer, partitions):
    print('Assignment:', partitions)

def print_revocation(consumer, partitions):
    print('Revocation:', partitions)

def consume_loop(group_id, topics):
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }
    
    c = Consumer(conf)
    # Subscribe with callbacks to see rebalancing
    c.subscribe(topics, on_assign=print_assignment, on_revoke=print_revocation)

    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break

            print(f"Received message: {msg.value().decode('utf-8')} from partition {msg.partition()}")

    except KeyboardInterrupt:
        pass
    finally:
        c.close()

if __name__ == "__main__":
    consume_loop("click-processors", ["user-clicks"])
```
</details>
