# Day 4: Testing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you test a function that depends on processing time?**
    -   *A*: Use `TestHarness` to manually advance processing time in a deterministic way.

2.  **Q: How do you detect backpressure?**
    -   *A*: Check the Flink Web UI (Backpressure tab) or monitor `outPoolUsage` metrics.

3.  **Q: What is a Flame Graph?**
    -   *A*: A visualization of stack traces to identify CPU hotspots.

### Production Challenges
-   **Challenge**: **Silent Failure**.
    -   *Scenario*: Job is running but producing no data.
    -   *Cause*: Watermark stalled, or logic error filtering everything.
    -   *Fix*: Monitor `numRecordsOut` and `lastCheckpointDuration`.

### Troubleshooting Scenarios
**Scenario**: Checkpoint size is growing linearly.
-   *Cause*: State leak.
-   *Fix*: Analyze checkpoint metrics. Use State Processor API to inspect the checkpoint content.
