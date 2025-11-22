# Day 3: Schema Registry - Deep Dive

## Deep Dive & Internals

### Caching
-   **Client-Side**: Producers and Consumers cache Schema IDs. They don't hit the Registry for every message.
-   **Server-Side**: The Registry is backed by a Kafka topic (`_schemas`). It is a stateful application built on top of Kafka.

### Redpanda's Built-in Registry
Redpanda embeds the Schema Registry *inside* the broker binary.
-   **Port 8081**: Exposes the standard Confluent Schema Registry API.
-   **No Sidecar**: You don't need a separate `schema-registry` container.

### Advanced Reasoning
**Why Binary Formats (Avro/Proto)?**
-   **Size**: JSON `{"id": 123456789}` is 16 bytes. Avro might be 4 bytes (varint). At 1TB/day, this saves 50% storage and bandwidth.
-   **Parsing Speed**: Binary parsing is orders of magnitude faster than JSON parsing.

### Performance Implications
-   **First Request Latency**: The first message is slow (fetching schema). Subsequent messages are fast (cached).
