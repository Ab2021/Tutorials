# Day 2: DataStream API - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between `map` and `flatMap`?**
    -   *A*: `map` produces exactly one output for every input. `flatMap` can produce 0, 1, or many.

2.  **Q: Why do we need `keyBy`?**
    -   *A*: To partition the state. Aggregations (sum, min, max) and Windows usually require a keyed stream so that the state is scoped to the key.

3.  **Q: What happens if you use a non-serializable object in a stream?**
    -   *A*: The job fails at submission time (or runtime) with `NotSerializableException`.

### Production Challenges
-   **Challenge**: **Serialization Bottleneck**.
    -   *Scenario*: CPU is high, but logic is simple.
    -   *Cause*: Using complex objects that fall back to Kryo serialization.
    -   *Fix*: Use POJOs or Tuples. Register custom serializers.

### Troubleshooting Scenarios
**Scenario**: `NullPointerException` in `open()`.
-   *Cause*: Trying to access runtime context before it's initialized (rare) or external resource failure.
