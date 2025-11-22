# Day 2: Schema Evolution Challenges

## Core Concepts & Theory

### The Problem
Streaming jobs run for years. Data formats change.
-   **Producer**: Adds a field `email`.
-   **Consumer**: Expects `email` to be missing.
-   **State**: Flink state contains old objects.

### Compatibility Modes
1.  **Backward**: New schema can read Old data. (Consumer upgrade first).
2.  **Forward**: Old schema can read New data. (Producer upgrade first).
3.  **Full**: Both ways.

### Strategies
1.  **Schema Registry**: Central authority. Rejects incompatible schemas.
2.  **Protobuf/Avro**: Binary formats with built-in evolution rules (tags/defaults).
3.  **JSON**: Flexible but dangerous (no type safety).

### Architectural Reasoning
**Why is State Evolution Hard?**
In a stateless app, you just restart with new code. In Flink, the **State** (Checkpoints) is serialized binary data.
-   If you change the class definition (`User`), Flink cannot deserialize the old checkpoint.
-   **Solution**: Use POJO serialization or Avro for State. Avoid Java Serialization.
