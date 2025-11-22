# Day 1: The Log Abstraction & Event Streaming

## Core Concepts & Theory

### The "Log" Abstraction
In the context of streaming systems, a **Log** is not just a text file for error messages. It is an **append-only, totally ordered sequence of records ordered by time**.
- **Append-Only**: You can only add to the end. Old data is immutable.
- **Ordered**: Events have a strict order ($t_1 < t_2 < t_3$).
- **Durable**: The log is persisted to disk.

This simple abstraction is the heart of Kafka and Redpanda. It unifies database commit logs and distributed messaging.

### Batch vs. Stream Processing
- **Batch**: Processing a bounded dataset at rest. High latency, high throughput. "Tell me what happened yesterday."
- **Stream**: Processing an unbounded dataset in motion. Low latency. "Tell me what is happening right now."
- **The Dual Nature**: A table is a snapshot of a stream at a point in time. A stream is the history of changes to a table.

### Architectural Reasoning
**Why the Log?**
1.  **Simplicity**: Append-only operations are O(1) and extremely fast on spinning disks and SSDs (sequential I/O).
2.  **Buffering**: Decouples producers from consumers. Producers don't block if consumers are slow.
3.  **Replayability**: Because the log is durable, consumers can rewind and re-read data. This enables fault tolerance and new use cases (like training a new ML model on old data).

### Key Components
-   **Event**: A key-value pair with a timestamp (e.g., `UserLogin {id: 123, time: 10:00}`).
-   **Stream**: An unbounded sequence of events.
-   **Producer**: The application creating events.
-   **Consumer**: The application reading events.
