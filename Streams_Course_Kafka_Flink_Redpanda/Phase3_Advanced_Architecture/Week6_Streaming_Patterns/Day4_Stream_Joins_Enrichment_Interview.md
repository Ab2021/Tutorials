# Day 4: Stream Joins - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between a Window Join and an Interval Join?**
    -   *A*: Window Join joins elements in the same fixed window. Interval Join joins elements within a relative time range (e.g., A.ts between B.ts - 5min and B.ts + 5min).

2.  **Q: How do you optimize a stream enrichment with a large static dataset?**
    -   *A*: If it fits in memory, use Broadcast State. If not, use Async I/O with caching or a Temporal Join if the dataset changes over time.

3.  **Q: What is "Skew" in joins?**
    -   *A*: One key (e.g., "Null" or a popular Item) has massive data. All goes to one node.
    -   *Fix*: Salt the key (add random suffix) to distribute load.

### Production Challenges
-   **Challenge**: **External Service Failure**.
    -   *Scenario*: Async I/O calls to User Service start timing out.
    -   *Fix*: Circuit Breaker pattern, exponential backoff, and fallback (default values).

### Troubleshooting Scenarios
**Scenario**: Join producing no results.
-   *Cause*: Timezones! One stream is UTC, other is EST. Timestamps don't align.
