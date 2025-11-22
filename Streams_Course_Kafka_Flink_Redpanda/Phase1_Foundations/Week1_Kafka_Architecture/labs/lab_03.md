# Lab 03: Basic Producer

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Implement a basic Kafka Producer.
-   Understand the asynchronous nature of `produce()`.
-   Use delivery reports (callbacks) to confirm message receipt.

## Problem Statement
Create a producer that sends 100 JSON events to `user-clicks`. Each event should look like `{"user_id": i, "action": "click"}`. You must register a callback to print whether the delivery was successful or failed.

## Starter Code
```python
from confluent_kafka import Producer
import json

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def produce_events(topic):
    p = Producer({'bootstrap.servers': 'localhost:9092'})
    # Loop and produce...
    # Don't forget to flush!

if __name__ == "__main__":
    produce_events("user-clicks")
```

## Hints
<details>
<summary>Hint 1</summary>
`p.produce()` is async. It just adds to a buffer. If you exit the script immediately, messages won't be sent. You need to call `p.flush()` at the end.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Producer
import json
import time

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

def produce_events(topic):
    conf = {'bootstrap.servers': 'localhost:9092'}
    p = Producer(conf)

    for i in range(100):
        event = {"user_id": i, "action": "click", "timestamp": time.time()}
        # Serialize to JSON string, then bytes
        value = json.dumps(event).encode('utf-8')
        # Use user_id as the key to ensure ordering per user
        key = str(i).encode('utf-8')
        
        p.produce(topic, key=key, value=value, callback=delivery_report)
        
        # Trigger callbacks periodically to free up memory
        p.poll(0)

    # Wait for any outstanding messages to be delivered and delivery reports to be received.
    p.flush()

if __name__ == "__main__":
    produce_events("user-clicks")
```
</details>
