# Day 1: Event Sourcing - Deep Dive

## Deep Dive & Internals

### Snapshotting
Replaying 1 million events to get the current balance is slow.
-   **Snapshot**: Periodically save the current state (e.g., every 1000 events).
-   **Recovery**: Load latest snapshot + replay subsequent events.
-   **Flink State**: Flink's checkpoints act as automatic snapshots for the stream processing part.

### Event Schema Evolution
-   **Upcasting**: Converting old event formats to new ones on-the-fly during replay.
-   **Weak Schema**: Storing events as JSON blobs (flexible but risky).
-   **Strong Schema**: Avro/Protobuf (safe but requires migration logic).

### Advanced Reasoning
**Consistency in CQRS**
CQRS implies **Eventual Consistency**. The Read Model lags behind the Write Model.
-   **Read-Your-Own-Writes**: A UI pattern where the client waits for the event to be indexed before reloading the page, or optimistically updates the UI.

### Performance Implications
-   **Write Throughput**: Extremely high (append-only).
-   **Read Latency**: Depends on the speed of the Projection engine (Flink).
