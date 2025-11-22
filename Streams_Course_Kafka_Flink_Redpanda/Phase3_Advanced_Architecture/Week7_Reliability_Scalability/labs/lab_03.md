# Lab 03: Schema Registry Setup

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
- Deploy Confluent Schema Registry
- Register Avro schemas
- Understand schema versioning

## Problem Statement
Set up Schema Registry using Docker Compose and register your first Avro schema for a `User` record with fields: `name` (string) and `age` (int).

## Starter Code
```yaml
version: '3'
services:
  schema-registry:
    image: confluentinc/cp-schema-registry:latest
    ports:
      - "8081:8081"
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: 'kafka:9092'
```

## Hints
<details>
<summary>Hint 1</summary>
Use the Schema Registry REST API at `http://localhost:8081`.
</details>

<details>
<summary>Hint 2</summary>
Register schemas using POST to `/subjects/{subject}/versions`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**docker-compose.yml:**
```yaml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
  
  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
  
  schema-registry:
    image: confluentinc/cp-schema-registry:latest
    depends_on:
      - kafka
    ports:
      - "8081:8081"
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: 'kafka:9092'
```

**Register Schema:**
```bash
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data '{
    "schema": "{\"type\":\"record\",\"name\":\"User\",\"fields\":[{\"name\":\"name\",\"type\":\"string\"},{\"name\":\"age\",\"type\":\"int\"}]}"
  }' \
  http://localhost:8081/subjects/users-value/versions
```

**Verify:**
```bash
# List all subjects
curl http://localhost:8081/subjects

# Get schema by ID
curl http://localhost:8081/schemas/ids/1
```
</details>
