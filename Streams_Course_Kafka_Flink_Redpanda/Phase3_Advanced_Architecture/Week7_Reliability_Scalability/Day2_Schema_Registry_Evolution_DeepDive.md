# Day 2: Schema Registry - Deep Dive

## Deep Dive & Internals

### Serialization Process
1.  **Producer**: `record = {name: "A"}`.
2.  **Look up**: Hash the schema. Check local cache. If missing, POST to Registry. Get ID=5.
3.  **Serialize**: `[MagicByte][ID=5][BinaryData]`.
4.  **Send**: To Kafka.

### Deserialization Process
1.  **Consumer**: Read bytes.
2.  **Extract ID**: Read first 5 bytes. ID=5.
3.  **Look up**: Check local cache. If missing, GET /schemas/ids/5.
4.  **Deserialize**: Use Schema 5 to read the binary data.

### Advanced Reasoning
**Transitive Compatibility**
-   **Check**: Is V3 compatible with V2?
-   **Transitive Check**: Is V3 compatible with V2 AND V1?
-   **Why?**: If you have data from 1 year ago (V1) in the topic, the consumer must be able to read it even if the current version is V3.

### Performance Implications
-   **Caching**: The Registry is not in the hot path. Producers/Consumers cache IDs. If Registry goes down, the app continues working (as long as it doesn't encounter a *new* schema).
