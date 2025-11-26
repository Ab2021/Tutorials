# Lab: Day 48 - Kafka Streams

## Goal
Produce and Consume messages with Kafka.

## Prerequisites
- Docker (Kafka).
- `pip install kafka-python`

## Step 1: Start Kafka
```yaml
# docker-compose.yml
version: '2'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - 9092:9092
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```
`docker-compose up -d`

## Step 2: Producer (`producer.py`)

```python
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

for i in range(10):
    data = {'number': i}
    # Send to topic 'test-topic'
    producer.send('test-topic', value=data)
    print(f"Sent: {data}")
    time.sleep(1)

producer.flush()
```

## Step 3: Consumer (`consumer.py`)

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening...")
for message in consumer:
    print(f"Received: {message.value} from Partition: {message.partition}")
```

## Step 4: Run It
1.  Run Consumer.
2.  Run Producer.

## Challenge: Partitions
1.  Create a topic with 3 partitions (using Kafka CLI inside container).
2.  Run 2 Consumers (same `group_id`).
3.  Run Producer.
4.  Observe how messages are split between the 2 consumers.
