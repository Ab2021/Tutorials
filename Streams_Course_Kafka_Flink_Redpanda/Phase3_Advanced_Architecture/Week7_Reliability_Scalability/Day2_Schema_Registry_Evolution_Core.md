# Day 2: Schema Registry & Evolution

## Core Concepts & Theory

### Why Schema Registry?
In a decoupled system, producers and consumers need a contract.
-   **Producer**: Serializes data using Schema ID 1.
-   **Registry**: Stores Schema 1 = `User(name, age)`.
-   **Consumer**: Downloads Schema 1 to deserialize.

### Evolution Rules
1.  **Backward Compatibility**: New schema can read old data. (Add optional field).
2.  **Forward Compatibility**: Old schema can read new data. (Add optional field, ignore unknown).
3.  **Full Compatibility**: Both ways.

### Architectural Reasoning
**The "Central Nervous System"**
The Schema Registry is the single source of truth for data governance. It prevents "Garbage In, Garbage Out". If a producer tries to send bad data, the serializer fails *before* sending to Kafka.

### Key Components
-   **Subject**: The scope of the schema (usually `TopicName-value`).
-   **ID**: Global unique ID for a schema version.
-   **Avro/Protobuf**: Preferred serialization formats.
