# Day 2: Schema Evolution - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if a producer sends a message with a schema not in the registry?**
    -   *A*: The send fails (if validation is on). Or it registers a new version (if auto-register is on). In Prod, **Auto-Register should be OFF**.

2.  **Q: How do you migrate Flink state to a new schema?**
    -   *A*: **State Processor API**. Read the old Savepoint (offline), map the data to the new class, write a new Savepoint. Start job from new Savepoint.

3.  **Q: JSON vs Avro for Streaming?**
    -   *A*: **Avro**. Strong typing, schema evolution, smaller payload. JSON is only for debugging or external APIs.

### Production Challenges
-   **Challenge**: **The "Unknown Field" Crash**.
    -   *Scenario*: Upstream adds a field. Downstream JSON parser crashes on unknown field.
    -   *Fix*: Configure parser to `IGNORE_UNKNOWN_PROPERTIES`.

-   **Challenge**: **Registry Downtime**.
    -   *Scenario*: Schema Registry is down.
    -   *Fix*: Clients cache schemas. They can survive if they've seen the schema before. New schemas will fail. High Availability (HA) for Registry is critical.
