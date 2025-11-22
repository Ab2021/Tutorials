# Lab 01: Consumer Lag Monitoring

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
- Monitor Kafka consumer lag
- Understand lag metrics
- Identify slow consumers

## Problem Statement
Create a Kafka consumer that processes messages slowly (sleep 100ms per message). Monitor the consumer lag using `kafka-consumer-groups` command and identify when lag exceeds 1000 messages.

## Starter Code
```python
from confluent_kafka import Consumer, KafkaError
import time

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'slow-consumer-group',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['test-topic'])

# TODO: Add slow processing and lag monitoring
```

## Hints
<details>
<summary>Hint 1</summary>
Use `time.sleep(0.1)` to simulate slow processing.
</details>

<details>
<summary>Hint 2</summary>
Run `kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group slow-consumer-group` to check lag.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Consumer, KafkaError
import time

def consume_slowly():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'slow-consumer-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True
    })
    
    consumer.subscribe(['test-topic'])
    
    try:
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
            if msg.error():
                print(f"Error: {msg.error()}")
                continue
            
            # Simulate slow processing
            time.sleep(0.1)
            print(f"Processed: {msg.value().decode('utf-8')}")
            
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == '__main__':
    consume_slowly()
```

**Monitoring Commands:**
```bash
# Check consumer lag
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group slow-consumer-group

# Expected output shows LAG column increasing
```
</details>
