# Lab 02: Topic Management

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
