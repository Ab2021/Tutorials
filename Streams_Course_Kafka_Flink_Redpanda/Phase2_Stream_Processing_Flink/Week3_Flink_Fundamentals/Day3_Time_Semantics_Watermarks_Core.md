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


### Advanced Theory: Watermark Strategies & Idleness
**1. Periodic vs Punctuated Watermarks**
-   **Periodic**: The system calls `getCurrentWatermark()` every 200ms. Good for regular streams.
-   **Punctuated**: The system emits a watermark *per event* (or based on event content). Good for sparse streams.

**2. The "Idle Source" Problem**
If a Kafka partition goes silent (no data), it stops sending watermarks.
-   **Impact**: The global watermark (Min of all partitions) stalls. Windows never close.
-   **Solution**: `WatermarkStrategy.withIdleness(Duration.ofMinutes(1))`. If a partition is idle, mark it as such so it doesn't hold back the global watermark.

**3. Handling "Future" Data**
If a device sends a timestamp from the year 3000:
-   The watermark jumps to 3000.
-   All windows from now until 3000 close instantly.
-   Real data arriving later is considered "Late" and dropped.
-   **Fix**: Filter out future timestamps in the Source.
