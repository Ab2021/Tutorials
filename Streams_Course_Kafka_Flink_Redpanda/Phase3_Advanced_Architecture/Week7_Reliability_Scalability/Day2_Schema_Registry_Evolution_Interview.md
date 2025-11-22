# Day 2: Schema Registry - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What happens if the Schema Registry is down?**
    -   *A*: Producers/Consumers use their local cache. They fail only if they encounter a schema ID they haven't seen before.

2.  **Q: How do you delete a field safely?**
    -   *A*: In Forward Compatibility: Consumers must stop using the field first. Then Producer stops sending it. Then remove from schema.

3.  **Q: Why Avro over JSON?**
    -   *A*: Compact (no field names in payload), fast, and has strict schema evolution rules enforced by the Registry.

### Production Challenges
-   **Challenge**: **Incompatible Schema Change**.
    -   *Scenario*: Dev changes `age` from `int` to `string`. Registry rejects registration. Producer fails.
    -   *Fix*: Follow evolution rules. Add `age_str` as a new field.

### Troubleshooting Scenarios
**Scenario**: `SerializationException: Error retrieving Avro schema`.
-   *Cause*: Network issue to Registry, or the ID in the message does not exist (Registry was wiped?).
