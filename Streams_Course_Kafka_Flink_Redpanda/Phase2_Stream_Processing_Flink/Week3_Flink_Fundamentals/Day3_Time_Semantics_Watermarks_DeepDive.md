# Day 3: Time Semantics - Deep Dive

## Deep Dive & Internals

### Watermark Propagation
Watermarks flow through the DAG.
-   **One-to-One**: Simple propagation.
-   **Many-to-One (Union/KeyBy)**: An operator receives watermarks from multiple upstream channels. It takes the **minimum** of all incoming watermarks.
    -   *Implication*: One slow upstream partition holds back the event time for the entire job.

### Idle Sources
If a Kafka partition has no data, it sends no watermarks. The downstream operator's watermark (min of all inputs) stalls.
-   **Fix**: `withIdleness(Duration)`. Marks a source as idle so it is ignored in the min calculation.

### Advanced Reasoning
**Late Data Handling**
What happens when an event arrives *after* the watermark has passed?
1.  **Default**: Dropped.
2.  **Allowed Lateness**: `allowedLateness(Time)`. Keep the window state around for a bit longer.
3.  **Side Output**: `sideOutputLateData(tag)`. Divert to a separate stream for manual handling.

### Performance Implications
-   **Watermark Interval**: `setAutoWatermarkInterval(200ms)`. Too frequent = CPU overhead. Too infrequent = jerky progress and higher latency for window results.
