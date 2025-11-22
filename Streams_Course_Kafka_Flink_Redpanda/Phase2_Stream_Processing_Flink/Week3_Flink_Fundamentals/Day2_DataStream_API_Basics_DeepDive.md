# Day 2: DataStream API - Deep Dive

## Deep Dive & Internals

### Serialization
Flink needs to serialize data to send it over the network.
-   **POJOs**: Flink analyzes your class. If it follows POJO rules (public fields, empty constructor), Flink uses a highly efficient custom serializer.
-   **Kryo**: Fallback for generic types. Slower.
-   **Tuples**: Extremely efficient.

### Type Erasure
Java erases generic types at runtime (`List<String>` becomes `List`). Flink uses **TypeInformation** to capture type info at compile time so it can serialize correctly at runtime.

### Advanced Reasoning
**Rich Functions**
Standard functions (`MapFunction`) are just lambdas. **RichFunctions** (`RichMapFunction`) give you access to the lifecycle:
-   `open()`: Called once during initialization (connect to DB here).
-   `close()`: Called at shutdown.
-   `getRuntimeContext()`: Access to State, Metrics, and Broadcast variables.

### Performance Implications
-   **Object Reuse**: Flink can be configured to reuse mutable objects (`enableObjectReuse()`) to reduce GC pressure. Dangerous if you hold references to objects.
