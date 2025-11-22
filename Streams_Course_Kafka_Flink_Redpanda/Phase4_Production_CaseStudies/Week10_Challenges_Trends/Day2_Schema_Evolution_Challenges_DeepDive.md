# Day 2: Schema Evolution - Deep Dive

## Deep Dive & Internals

### Flink State Evolution
How to add a field to a `ValueState<User>`?
1.  **Avro**: If `User` is an Avro generated class, Flink supports schema evolution out of the box (using Avro Serializer).
2.  **POJO**: Flink supports adding fields *if* they are new POJO fields.
3.  **Kryo**: **NO**. Kryo is not compatible. If you change the class, state is lost.

### Schema Registry in Production
-   **Subject Naming**: `topic-value`.
-   **Validation**: Producer checks registry before sending.
-   **Caching**: Producer/Consumer cache the schema ID (4 bytes). The payload only contains the ID + Data.

### Handling "Breaking" Changes
Sometimes you MUST break compatibility (e.g., rename field).
-   **Dual Topics**:
    1.  Create `topic_v2`.
    2.  Producer writes to both (or switches).
    3.  New Consumer reads `topic_v2`.
    4.  Old Consumer reads `topic_v1`.
-   **Translation Layer**: A Flink job that reads `v1`, maps to `v2`, writes to `topic_v2`.

### Performance Implications
-   **Deserialization Overhead**: JSON is slow. Avro is fast. Protobuf is fastest.
-   **Payload Size**: JSON is verbose. Avro/Proto are compact (no field names in payload).
