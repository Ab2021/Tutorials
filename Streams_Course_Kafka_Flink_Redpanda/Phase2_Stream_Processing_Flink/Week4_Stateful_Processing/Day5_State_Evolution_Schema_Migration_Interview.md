# Day 5: State Evolution - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you add a field to a POJO in State without losing data?**
    -   *A*: Ensure the POJO follows Flink rules (public fields). Flink's POJO serializer supports adding fields (they will be null for old records).

2.  **Q: What is the State Processor API?**
    -   *A*: A library to read, modify, and write Savepoints offline.

3.  **Q: Can you change the `keyBy` field and restore state?**
    -   *A*: No. Changing the key changes the partitioning. State is bound to the key. You must use State Processor API to re-key the state.

### Production Challenges
-   **Challenge**: **Incompatible Serializer**.
    -   *Scenario*: `java.io.InvalidClassException`.
    -   *Fix*: Provide a custom `TypeSerializerSnapshot` or use State Processor API.

### Troubleshooting Scenarios
**Scenario**: Job fails to restore after adding a field.
-   *Cause*: You were using `Kryo` (generic) serializer instead of POJO. Kryo does not support evolution.
-   *Fix*: Always ensure your classes are recognized as POJOs or Avro from Day 1.
