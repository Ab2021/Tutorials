# Lab 12: Schema Registry Setup

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

schema_str = """
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"}
    ]
}
"""

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
