# Day 3: Time Semantics & Watermarks

## Core Concepts & Theory

### The Three Times
1.  **Event Time**: The time the event actually occurred (timestamp in the device). This is what matters for correctness.
2.  **Processing Time**: The time the event arrived at the Flink machine.
3.  **Ingestion Time**: The time the event entered the Flink source.

### The Problem of Out-of-Order Data
In distributed systems, events often arrive out of order. An event from 10:00 might arrive after an event from 10:05 due to network lag.
If you calculate "How many clicks at 10:00?", you can't just wait for the clock to hit 10:01. You might miss the late data.

### Watermarks
A **Watermark(t)** is a control message that flows through the stream and asserts: "No more events with timestamp < t will arrive."
-   It allows the system to measure progress in Event Time.
-   When a window operator receives Watermark(t), it knows it can safely close any window ending before t.

### Architectural Reasoning
**Correctness vs. Latency**
Watermarks allow you to trade off latency for correctness.
-   **Aggressive Watermarks**: Assume little lag. Low latency, but data might be dropped as "late".
-   **Conservative Watermarks**: Assume huge lag. High correctness, but high latency (waiting for late data).

### Key Components
-   **TimestampAssigner**: Extracts the timestamp from the event.
-   **WatermarkGenerator**: Emits watermarks (Periodic or Punctuated).
