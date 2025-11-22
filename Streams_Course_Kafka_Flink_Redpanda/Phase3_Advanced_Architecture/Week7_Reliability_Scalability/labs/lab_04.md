# Lab 04: Avro Schema Evolution

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Practice schema evolution
- Understand backward compatibility
- Handle schema changes safely

## Problem Statement
1. Register a schema `User(name, age)`
2. Produce messages using this schema
3. Evolve the schema to add an optional field `email`
4. Verify old consumers can still read new messages

## Starter Code
```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

value_schema_str = """
{
   "type": "record",
   "name": "User",
   "fields": [
       {"name": "name", "type": "string"},
       {"name": "age", "type": "int"}
   ]
}
"""

# TODO: Evolve schema and test compatibility
```

## Hints
<details>
<summary>Hint 1</summary>
Add new fields with default values to maintain backward compatibility.
</details>

<details>
<summary>Hint 2</summary>
Use `"default": null` for optional fields.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**V1 Schema:**
```python
value_schema_v1 = """
{
   "type": "record",
   "name": "User",
   "fields": [
       {"name": "name", "type": "string"},
       {"name": "age", "type": "int"}
   ]
}
"""
```

**V2 Schema (Backward Compatible):**
```python
value_schema_v2 = """
{
   "type": "record",
   "name": "User",
   "fields": [
       {"name": "name", "type": "string"},
       {"name": "age", "type": "int"},
       {"name": "email", "type": ["null", "string"], "default": null}
   ]
}
"""
```

**Producer with V2:**
```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

avroProducer = AvroProducer({
    'bootstrap.servers': 'localhost:9092',
    'schema.registry.url': 'http://localhost:8081'
}, default_value_schema=avro.loads(value_schema_v2))

# Produce with new field
avroProducer.produce(topic='users', value={
    "name": "Alice", 
    "age": 30, 
    "email": "alice@example.com"
})
avroProducer.flush()
```

**Consumer with V1 (still works!):**
```python
from confluent_kafka.avro import AvroConsumer

avroConsumer = AvroConsumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'v1-consumer',
    'schema.registry.url': 'http://localhost:8081'
})

avroConsumer.subscribe(['users'])

while True:
    msg = avroConsumer.poll(1)
    if msg:
        print(msg.value())  # {'name': 'Alice', 'age': 30}
        # email field is ignored by V1 consumer
```

**Verification:**
```bash
# Check compatibility
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data '{"schema":"..."}' \
  http://localhost:8081/compatibility/subjects/users-value/versions/latest
```
</details>
