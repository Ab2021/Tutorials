# Day 1: Flink SQL - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between the Table API and SQL?**
    -   *A*: They are equivalent. Table API is embedded in Java/Python (compile-time checks). SQL is text-based (runtime checks). Both use the same optimizer.

2.  **Q: How do you handle infinite state in Flink SQL?**
    -   *A*: Use Window Aggregations, Interval Joins, or configure `table.exec.state.ttl` to expire old state.

3.  **Q: What is a Temporal Table Join?**
    -   *A*: Joining a stream with a slowly changing dimension table (e.g., Orders JOIN Products) at a specific point in time.

### Production Challenges
-   **Challenge**: **Retraction Storm**.
    -   *Scenario*: A multi-level aggregation (Count -> Max). An update upstream causes a Retract(-1) and Accumulate(+1) downstream. This doubles the load.
    -   *Fix*: Use Mini-Batch aggregation or optimize the query plan.

### Troubleshooting Scenarios
**Scenario**: `TableException: Table is not an append-only table. Use the toChangelogStream()`.
-   *Cause*: You tried to convert a dynamic table with updates/deletes into a simple DataStream.
-   *Fix*: Use `to_changelog_stream` to handle the update flags.
