# Day 3: Schema Registry & Serialization

## Core Concepts & Theory

### The Contract
In a decoupled system, Producers and Consumers need a **Contract** to understand the data format.
-   **JSON**: Flexible but verbose. No schema enforcement. "Schemaless".
-   **Avro/Protobuf**: Binary, compact, strongly typed. Requires a schema.

### The Schema Registry
A central repository that stores schemas.
1.  **Producer**: Checks Registry. "Is this schema ID 1?" -> Sends `[ID=1][Payload]`.
2.  **Consumer**: Reads `[ID=1]`. Asks Registry "What is schema 1?" -> Deserializes payload.

### Compatibility Modes
-   **Backward**: New schema can read old data. (Delete field).
-   **Forward**: Old schema can read new data. (Add optional field).
-   **Full**: Both ways.

### Architectural Reasoning
**Why Schema Registry?**
It prevents "Poison Pills". If a producer changes the data format (e.g., renames "id" to "user_id") without a registry, downstream consumers will crash. The Registry rejects incompatible changes at the *producer* level.

### Key Components
-   **Subject**: Scope for a schema (usually `topic-value`).
-   **Schema ID**: Global unique ID.
-   **Avro/Protobuf/JSON Schema**: Supported formats.
