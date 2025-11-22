# Day 3: Keyed State - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens to Keyed State when you change parallelism?**
    -   *A*: Flink redistributes the **Key Groups** among the new number of tasks. State is preserved.

2.  **Q: Why is `MapState` better than `ValueState<HashMap>`?**
    -   *A*: `MapState` allows partial updates/reads. `ValueState` requires full serialization/deserialization of the whole map.

3.  **Q: How do you handle State Schema Evolution?**
    -   *A*: Use Avro/Protobuf or Flink's POJO serializer which supports adding fields. Or use the State Processor API to migrate state offline.

### Production Challenges
-   **Challenge**: **State Explosion**.
    -   *Scenario*: Job runs fine for a week, then OOMs.
    -   *Cause*: Unique keys are infinite, and no TTL is set.
    -   *Fix*: Enable TTL.

### Troubleshooting Scenarios
**Scenario**: `InvalidProgramException: Serializer mismatch`.
-   *Cause*: You changed the class structure of the state object but tried to restore from an old checkpoint.
-   *Fix*: Revert code or use State Processor API to migrate.
