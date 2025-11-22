# Day 3: Schema Registry - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if the Schema Registry goes down?**
    -   *A*: Existing producers/consumers continue working (cached schemas). New producers/consumers (or new schemas) will fail.

2.  **Q: Explain Backward Compatibility.**
    -   *A*: Consumers with the *new* schema can read data written with the *old* schema. This allows you to upgrade consumers *first*.

3.  **Q: How does the consumer know which schema to use?**
    -   *A*: The first 5 bytes of the message payload contain the "Magic Byte" and the 4-byte Schema ID.

### Production Challenges
-   **Challenge**: **Incompatible Schema Change**.
    -   *Scenario*: Developer renames a required field. Registry rejects the registration.
    -   *Fix*: Add an alias (if supported) or create a new version with a default value.

### Troubleshooting Scenarios
**Scenario**: `SerializationException: Unknown magic byte`.
-   *Cause*: You are trying to consume data using an Avro deserializer, but the data was produced as raw JSON or String.
