# Day 3: Clickstream - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you calculate "Active Users" (DAU) in real-time?**
    -   *A*: Use a **HyperLogLog** (HLL) sketch. It estimates cardinality with low memory (KB) and high accuracy. Exact count requires storing every UserID (GBs of state).

2.  **Q: How do you handle late-arriving events in Session Windows?**
    -   *A*: Flink can merge the late event into an existing session (and extend it). If the session was already emitted, it can emit an "Update" (Retraction).

3.  **Q: What is the "Funnel Analysis" problem?**
    -   *A*: "View -> Click -> Buy". Order matters. Use Flink CEP or SQL `MATCH_RECOGNIZE`.

### Production Challenges
-   **Challenge**: **High Cardinality Dimensions**.
    -   *Scenario*: Grouping by URL. If you have 1M unique URLs (query params), the result set explodes.
    -   *Fix*: Normalize URLs (strip query params) or use Top-N patterns.

-   **Challenge**: **Ad Blockers**.
    -   *Scenario*: 20% of events are missing.
    -   *Fix*: Server-side tracking (Proxy) instead of Client-side.

### Troubleshooting Scenarios
**Scenario**: Session Window never closes.
-   *Cause*: Watermark is stuck. No new data arriving to push the watermark forward.
-   *Fix*: Use `withIdleness` in watermark strategy.
