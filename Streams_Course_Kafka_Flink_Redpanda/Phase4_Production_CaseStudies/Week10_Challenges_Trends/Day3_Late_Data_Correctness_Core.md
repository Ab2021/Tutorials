# Day 3: Late Data & Correctness

## Core Concepts & Theory

### The Reality of Time
-   **Event Time**: When it happened (Phone clock).
-   **Processing Time**: When the server saw it (Server clock).
-   **Skew**: The difference. Caused by network partitions, airplane mode, crashes.

### Watermarks
A Watermark `W(T)` means: "I assert that no event with timestamp < T will arrive anymore."
-   It flows with the stream.
-   It triggers windows.
-   It trades **Latency** vs **Completeness**.

### Handling Lateness
1.  **Allowed Lateness**: Keep window state open for X minutes. If late data arrives, update the result.
2.  **Side Output**: If data is *too* late (after allowed lateness), send to a "Late" stream (DLQ) for manual fix.

### Architectural Reasoning
**Correctness vs Latency**
-   **Strict Correctness**: Wait for 100% of data. Latency = Infinity.
-   **Low Latency**: Emit result immediately. Accuracy = Low.
-   **Streaming Solution**: Emit early result (speculative), then emit updates (retractions) as data arrives.
