# Day 2: CEP - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between `next()` and `followedBy()`?**
    -   *A*: `next` requires strict sequence (no events in between). `followedBy` allows non-matching events in between.

2.  **Q: How does CEP handle state?**
    -   *A*: It stores partial matches in the State Backend (RocksDB).

3.  **Q: Why is `.within()` important?**
    -   *A*: To prune state. Without it, partial matches accumulate forever, leading to OOM.

### Production Challenges
-   **Challenge**: **High Latency with Complex Patterns**.
    -   *Scenario*: Pattern `A followedBy B`. Stream has 1M "A"s and no "B".
    -   *Cause*: Flink checks every "A" against every incoming event.
    -   *Fix*: Use stricter conditions or shorter windows.

### Troubleshooting Scenarios
**Scenario**: Pattern not matching.
-   *Cause*: Time characteristic. Are you using Event Time? Are watermarks progressing? CEP relies heavily on correct time.
