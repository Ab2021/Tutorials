# Day 4: Windowing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the difference between Tumbling and Sliding windows?**
    -   *A*: Tumbling windows do not overlap. Sliding windows overlap.

2.  **Q: How does a Session Window work?**
    -   *A*: It groups elements by key and merges windows that are within a "gap" of each other. It has no fixed duration.

3.  **Q: When would you use `ProcessWindowFunction`?**
    -   *A*: When you need the window start/end timestamps or need to perform a calculation that requires all elements (like median).

### Production Challenges
-   **Challenge**: **State Explosion with Sliding Windows**.
    -   *Scenario*: Window(1hr, slide 1s).
    -   *Fix*: Don't do this. Use a larger slide or a different pattern.

### Troubleshooting Scenarios
**Scenario**: Window results are incorrect (too low).
-   *Cause*: Late data is being dropped.
-   *Fix*: Check `sideOutputLateData` to see if data is arriving after the watermark.
