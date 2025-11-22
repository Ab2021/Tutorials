import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase1_Foundations\Week1_Kafka_Architecture\labs"

labs_content = {
    "lab_01.md": """# Lab 01: Start a Kafka Cluster

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Understand how to set up a local Kafka environment using Docker Compose.
-   Verify cluster health using the Kafka CLI.
-   Understand the dependency on Zookeeper (or KRaft).

## Problem Statement
Set up a 3-node Kafka cluster with a single Zookeeper node using Docker Compose. Once running, use the CLI to list the active brokers.

## Starter Code
```yaml
version: '2'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka1:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      # Complete the configuration...
```

## Hints
<details>
<summary>Hint 1: Advertised Listeners</summary>
You need to configure `KAFKA_ADVERTISED_LISTENERS` so that clients (and other brokers) know how to reach this broker. For Docker, use `PLAINTEXT://kafka1:29092` for internal communication and `PLAINTEXT_HOST://localhost:9092` for external.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### `docker-compose.yml`
```yaml
version: '2'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - 22181:2181

  kafka1:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka1:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  kafka2:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9093:9093"
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka2:29092,PLAINTEXT_HOST://localhost:9093
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  kafka3:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9094:9094"
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka3:29092,PLAINTEXT_HOST://localhost:9094
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

### Verification
Run `docker-compose up -d`.
Then check the logs or use `zookeeper-shell` to list brokers:
```bash
docker exec -it <container_id> zookeeper-shell zookeeper:2181 ls /brokers/ids
# Output should be: [1, 2, 3]
```
</details>
""",
    "lab_02.md": """# Lab 02: Topic Management

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use the `AdminClient` API in Python.
-   Understand partitions and replication factors.
-   Handle `TopicAlreadyExists` exceptions.

## Problem Statement
Write a Python script using `confluent-kafka` to create a topic named `user-clicks` with **3 partitions** and a **replication factor of 2**. If the topic already exists, print a message instead of crashing.

## Starter Code
```python
from confluent_kafka.admin import AdminClient, NewTopic

def create_topic(conf, topic_name):
    admin_client = AdminClient(conf)
    # Your code here...

if __name__ == "__main__":
    conf = {'bootstrap.servers': 'localhost:9092'}
    create_topic(conf, "user-clicks")
```

## Hints
<details>
<summary>Hint 1</summary>
Use `admin_client.create_topics()`. It returns a dictionary of futures. You need to wait on the future to see if it succeeded.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaException

def create_topic(conf, topic_name):
    admin_client = AdminClient(conf)
    
    # Define the topic: Name, Partitions, Replication Factor
    new_topic = NewTopic(topic_name, num_partitions=3, replication_factor=2)
    
    # Create the topic
    fs = admin_client.create_topics([new_topic])
    
    # Wait for the result
    for topic, f in fs.items():
        try:
            f.result()  # The result itself is None
            print(f"Topic {topic} created")
        except Exception as e:
            if e.args[0].code() == KafkaException.TOPIC_ALREADY_EXISTS:
                 print(f"Topic {topic} already exists")
            else:
                print(f"Failed to create topic {topic}: {e}")

if __name__ == "__main__":
    conf = {'bootstrap.servers': 'localhost:9092'}
    create_topic(conf, "user-clicks")
```
</details>
""",
    "lab_03.md": """# Lab 03: Basic Producer

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
""",
    "lab_04.md": """# Lab 04: Consumer Groups

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
""",
    "lab_05.md": """# Lab 05: ISR & Min.Insync.Replicas

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
""",
    "lab_06.md": """# Lab 06: Log Compaction

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand how log compaction works.
-   Configure a topic for compaction.
-   Verify that old values for the same key are deleted.

## Problem Statement
1.  Create a topic `user-states` with `cleanup.policy=compact` and `segment.ms=100` (to trigger compaction quickly).
2.  Produce: `Key=A, Val=1`, then `Key=A, Val=2`, then `Key=A, Val=3`.
3.  Consume from the beginning.
4.  **Expected Result**: Eventually, you should only see `Key=A, Val=3`. (Note: You might see Val=2 momentarily before the cleaner thread runs).

## Starter Code
```python
# Topic Config
config = {
    'cleanup.policy': 'compact',
    'segment.ms': '100',
    'min.cleanable.dirty.ratio': '0.01' # Aggressive cleaning
}
```

## Hints
<details>
<summary>Hint 1</summary>
Log compaction is not instant. The "active segment" is never compacted. You need to roll the segment (hence low `segment.ms`) to make it eligible for compaction.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Create Compacted Topic
```python
from confluent_kafka.admin import AdminClient, NewTopic

admin = AdminClient({'bootstrap.servers': 'localhost:9092'})
topic = NewTopic("user-states", num_partitions=1, replication_factor=1,
                 config={
                     'cleanup.policy': 'compact',
                     'segment.ms': '100',
                     'min.cleanable.dirty.ratio': '0.01'
                 })
admin.create_topics([topic])
```

### Step 2: Produce Updates
```python
p = Producer({'bootstrap.servers': 'localhost:9092'})
key = b"user1"
p.produce("user-states", key=key, value=b"login")
p.produce("user-states", key=key, value=b"update_profile")
p.produce("user-states", key=key, value=b"logout") # This should be the final state
p.flush()
```

### Step 3: Consume
Wait a few seconds for the cleaner to run.
```python
c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'state-reader',
    'auto.offset.reset': 'earliest'
})
c.subscribe(["user-states"])

# You should eventually see only "logout" for user1, 
# though you might see all 3 if you consume too fast before compaction.
```
</details>
""",
    "lab_07.md": """# Lab 07: Custom Partitioner

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
""",
    "lab_08.md": """# Lab 08: Idempotent Producer

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
""",
    "lab_09.md": """# Lab 09: Consumer Offsets

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand how offsets are committed.
-   Manually commit offsets.
-   Seek to a specific offset (replay).

## Problem Statement
1.  Produce 10 messages to a topic.
2.  Consume them and print them.
3.  Use `consumer.seek()` to rewind to offset 5.
4.  Consume again. You should see messages 5-9 again.

## Starter Code
```python
from confluent_kafka import TopicPartition

# ... inside consumer loop ...
# Rewind logic
tp = TopicPartition(topic, partition, 5)
consumer.assign([tp])
consumer.seek(tp)
```

## Hints
<details>
<summary>Hint 1</summary>
You cannot `seek()` on a subscribed consumer directly if you are using dynamic partition assignment (`subscribe()`). You usually need to do it inside the `on_assign` callback or use `assign()` (manual assignment) instead of `subscribe()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Consumer, TopicPartition

def run_seek():
    c = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'rewind-group',
        'auto.offset.reset': 'earliest'
    })
    
    # Manual assignment is easier for seek demos
    tp = TopicPartition('user-clicks', 0)
    c.assign([tp])
    
    # Read first 10
    for _ in range(10):
        msg = c.poll(1.0)
        if msg: print(f"Read offset {msg.offset()}")
        
    print("Rewinding to offset 5...")
    tp.offset = 5
    c.seek(tp)
    
    # Read again
    while True:
        msg = c.poll(1.0)
        if msg is None: break
        print(f"Re-read offset {msg.offset()}")
        
    c.close()

if __name__ == "__main__":
    run_seek()
```
</details>
""",
    "lab_10.md": """# Lab 10: Kafka Connect Basics

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
-   Understand the role of Kafka Connect.
-   Configure a standalone connector (FileStreamSource).

## Problem Statement
Use the `connect-standalone` mode (available in the Confluent container) to read lines from a text file and publish them to a Kafka topic.
1.  Create a file `data.txt` with some lines.
2.  Configure `source.properties`.
3.  Run the connector.

## Starter Code
```properties
# source.properties
name=local-file-source
connector.class=FileStreamSource
tasks.max=1
file=/tmp/data.txt
topic=connect-test
```

## Hints
<details>
<summary>Hint 1</summary>
You need to mount the file into the container or create it inside the container.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Create Data File
```bash
docker exec -it kafka1 bash -c "echo 'hello world' > /tmp/data.txt"
```

### Step 2: Create Config File
Create `connect-file-source.properties` inside the container:
```properties
name=local-file-source
connector.class=org.apache.kafka.connect.file.FileStreamSourceConnector
tasks.max=1
file=/tmp/data.txt
topic=connect-test
```

### Step 3: Run Connect (Standalone)
```bash
# This command assumes you are inside the container and have the config
connect-standalone /etc/kafka/connect-standalone.properties connect-file-source.properties
```

### Step 4: Verify
Consume from `connect-test`:
```bash
kafka-console-consumer --bootstrap-server localhost:9092 --topic connect-test --from-beginning
```
</details>
""",
    "lab_11.md": """# Lab 11: Kafka Streams DSL (Faust)

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Understand stream processing concepts (map, filter, group_by).
-   Use a Python streaming library (Faust) to mimic Kafka Streams.

## Problem Statement
Implement a "Word Count" application using Faust.
1.  Read from `sentences` topic.
2.  Split lines into words.
3.  Count occurrences of each word.
4.  Print the counts to stdout (or a topic).

## Starter Code
```python
import faust

app = faust.App('word-count', broker='kafka://localhost:9092')
topic = app.topic('sentences', value_type=str)

@app.agent(topic)
async def count_words(sentences):
    async for sentence in sentences:
        # Logic here...
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
Faust tables (`app.Table`) are used for stateful aggregations like counting.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import faust

app = faust.App('word-count', broker='kafka://localhost:9092')
topic = app.topic('sentences', value_type=str)

# Table to store counts. Default value is 0.
word_counts = app.Table('word_counts', default=int)

@app.agent(topic)
async def count_words(sentences):
    async for sentence in sentences:
        for word in sentence.split():
            word_counts[word] += 1
            print(f"Word: {word}, Count: {word_counts[word]}")

if __name__ == '__main__':
    app.main()
```
Run with: `python worker.py worker -l info`
</details>
""",
    "lab_12.md": """# Lab 12: Schema Registry Setup

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Understand why schemas are important (contracts).
-   Use Avro serialization.
-   Interact with Confluent Schema Registry.

## Problem Statement
1.  Start Schema Registry (add to Docker Compose).
2.  Write a producer that sends **Avro** encoded data (User: name, age).
3.  The schema should be automatically registered.

## Starter Code
```yaml
# docker-compose snippet
schema-registry:
  image: confluentinc/cp-schema-registry:latest
  environment:
    SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka1:29092
    SCHEMA_REGISTRY_HOST_NAME: schema-registry
```

## Hints
<details>
<summary>Hint 1</summary>
Use `confluent_kafka.schema_registry.avro.AvroProducer` (legacy) or `SerializingProducer` with `AvroSerializer`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Python Producer with Avro
```python
from confluent_kafka import SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

schema_str = \"\"\"
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"}
    ]
}
\"\"\"

def run():
    schema_registry_conf = {'url': 'http://localhost:8081'}
    schema_registry_client = SchemaRegistryClient(schema_registry_conf)

    avro_serializer = AvroSerializer(schema_registry_client, schema_str)

    producer_conf = {
        'bootstrap.servers': 'localhost:9092',
        'key.serializer': None,
        'value.serializer': avro_serializer
    }

    p = SerializingProducer(producer_conf)

    p.produce(topic='users-avro', value={"name": "Alice", "age": 30})
    p.flush()

if __name__ == "__main__":
    run()
```
</details>
""",
    "lab_13.md": """# Lab 13: ACLs & Security

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Understand Kafka ACLs (Access Control Lists).
-   Restrict access to a topic.

## Problem Statement
*Note: This requires a Kafka cluster configured with an Authorizer (e.g., `SimpleAclAuthorizer`). For this lab, we will assume the environment is set up or we will use the CLI to simulate the commands.*

1.  Create a user `alice`.
2.  Deny `alice` from reading topic `secret`.
3.  Verify that `alice` cannot consume.

## Starter Code
```bash
kafka-acls --bootstrap-server localhost:9092 --add --allow-principal User:bob --operation Read --topic secret
```

## Hints
<details>
<summary>Hint 1</summary>
By default, if no ACLs exist, access might be allowed (depending on `allow.everyone.if.no.acl.found`).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Command to Add ACL
```bash
# Allow Bob to read/write
kafka-acls --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:bob \
  --operation Read \
  --operation Write \
  --topic secret

# Deny Alice (if implicit allow is on, or just don't add her)
kafka-acls --bootstrap-server localhost:9092 \
  --add \
  --deny-principal User:alice \
  --operation All \
  --topic secret
```
</details>
""",
    "lab_14.md": """# Lab 14: Monitoring with JMX

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Enable JMX in Kafka.
-   Connect using JConsole or VisualVM.
-   Identify key metrics (MessagesInPerSec, BytesOutPerSec).

## Problem Statement
1.  Configure `KAFKA_JMX_OPTS` in Docker Compose.
2.  Expose the JMX port.
3.  Connect via JConsole on your host machine.
4.  Find the `MessagesInPerSec` MBean.

## Starter Code
```yaml
environment:
  KAFKA_JMX_PORT: 9101
  KAFKA_JMX_HOSTNAME: localhost
```

## Hints
<details>
<summary>Hint 1</summary>
You might need to set `-Dcom.sun.management.jmxremote.rmi.port=9101` as well.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Docker Compose Update
```yaml
    environment:
      KAFKA_JMX_PORT: 9101
      KAFKA_JMX_HOSTNAME: localhost
    ports:
      - "9101:9101"
```

### Connection
1.  Open `jconsole` (part of JDK).
2.  Connect to `localhost:9101`.
3.  Navigate to `MBeans` -> `kafka.server` -> `BrokerTopicMetrics` -> `MessagesInPerSec`.
</details>
""",
    "lab_15.md": """# Lab 15: Multi-Broker Setup (Manual)

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
"""
}

print("🚀 Upgrading Week 1 Labs with Comprehensive Solutions...")

for filename, content in labs_content.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Upgraded {filename}")

print("✅ Week 1 Labs Upgrade Complete!")
