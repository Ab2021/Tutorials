# Lab 07: Custom Partitioner

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Understand how producers decide which partition to send to.
-   Implement a custom partitioning strategy.

## Problem Statement
You have a topic `vip-users` with 4 partitions.
-   Partition 0: Reserved for VIP users (IDs starting with 'vip_').
-   Partitions 1-3: For regular users.
Implement a custom partitioner in Python to enforce this logic.

## Starter Code
```python
def my_partitioner(key, all_partitions, available_partitions):
    # key is bytes
    # return partition_id (int)
    pass

p = Producer({
    'bootstrap.servers': 'localhost:9092',
    'partitioner': my_partitioner
})
```

## Hints
<details>
<summary>Hint 1</summary>
The `key` argument in the partitioner callback is bytes. Decode it to string to check the prefix.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Producer
import random

def vip_partitioner(key, all_partitions, available_partitions):
    if key is None:
        return random.choice([1, 2, 3])
    
    key_str = key.decode('utf-8')
    
    if key_str.startswith('vip_'):
        return 0  # VIPs go to partition 0
    
    # Regular users go to 1, 2, or 3
    # Simple hash to distribute
    return 1 + (hash(key_str) % 3)

def run():
    p = Producer({
        'bootstrap.servers': 'localhost:9092',
        'partitioner': vip_partitioner
    })
    
    p.produce('vip-users', key=b'vip_alice', value=b'high_priority')
    p.produce('vip-users', key=b'bob', value=b'low_priority')
    p.flush()

if __name__ == "__main__":
    run()
```
</details>
