# Lab 08: Idempotent Producer

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand "Exactly-Once" semantics at the producer level.
-   Configure idempotence.

## Problem Statement
Configure a producer to be idempotent. Simulate a network error (or just verify configuration) to ensure that retries do not cause duplicates.
Note: Simulating actual network duplicates is hard without a proxy (like Toxiproxy), so we will focus on the configuration and verification.

## Starter Code
```python
conf = {
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,
    # What else is required for idempotence?
}
```

## Hints
<details>
<summary>Hint 1</summary>
When `enable.idempotence` is True, `acks` must be `all`. `retries` must be > 0. `max.in.flight.requests.per.connection` must be <= 5.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Producer

def run_idempotent():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'enable.idempotence': True,
        'acks': 'all',
        'retries': 5,
        'max.in.flight.requests.per.connection': 5
    }
    
    p = Producer(conf)
    
    # Sending messages works exactly the same
    p.produce('secure-topic', key=b'id1', value=b'value1')
    p.flush()
    
    print("Message sent idempotently.")

if __name__ == "__main__":
    run_idempotent()
```
</details>
