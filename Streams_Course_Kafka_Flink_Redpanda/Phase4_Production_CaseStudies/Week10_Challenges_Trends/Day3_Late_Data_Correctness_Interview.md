# Day 3: Late Data - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Watermark and Event Time?**
    -   *A*: Event Time is an attribute of the *record*. Watermark is a *control signal* in the stream that measures the progress of Event Time.

2.  **Q: Can Watermarks go backwards?**
    -   *A*: No. Watermarks are monotonically increasing. If a source sends a lower watermark, it is ignored.

3.  **Q: How do you handle data from a device that was offline for 3 days?**
    -   *A*: If your window is 1 hour, this data is "Late".
    -   Option A: Drop it.
    -   Option B: Side Output -> Batch Process -> Merge with real-time results.

### Production Challenges
-   **Challenge**: **Stuck Watermark**.
    -   *Scenario*: One Kafka partition is empty. Job stops producing output.
    -   *Fix*: Enable Idleness detection.

-   **Challenge**: **Future Timestamps**.
    -   *Scenario*: Buggy device sends Year 3000. Watermark jumps to 3000. All windows close instantly. Real data is now "Late".
    -   *Fix*: Filter "Future" timestamps at ingestion.

### Troubleshooting Scenarios
**Scenario**: Windows are closing too early (missing data).
-   *Cause*: Watermark delay is too short (e.g., 1s) for the network jitter.
-   *Fix*: Increase bounded out-of-orderness delay.
