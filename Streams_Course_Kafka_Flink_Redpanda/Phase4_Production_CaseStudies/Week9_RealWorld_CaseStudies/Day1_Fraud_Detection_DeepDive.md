# Day 1: Fraud Detection - Deep Dive

## Deep Dive & Internals

### The "Impossible Travel" Pattern
**Rule**: Two transactions from the same card in different locations with speed > 500 mph.
**Implementation**:
-   **KeyBy**: `card_id`.
-   **State**: Store `last_location` and `last_timestamp`.
-   **Process**:
    1.  On new event, calculate distance and time diff.
    2.  Calculate speed.
    3.  If speed > threshold, emit Alert.
    4.  Update state.

### Handling Late Data
Fraud detection is time-sensitive.
-   **Watermarks**: Crucial. If data arrives 1 hour late, it might trigger a false positive (or negative).
-   **Strategy**: Drop extremely late data? Or process it and send a "Correction" event? Usually, for blocking, we only care about low latency.

### Feature Store Integration
Models need historical features (e.g., "Is this amount > 5x the user's average?").
-   **Online Feature Store (Redis/Cassandra)**: Flink queries this via Async I/O.
-   **Latency**: Async I/O adds latency.
-   **Optimization**: Pre-load hot profiles into Flink Managed Memory (State).

### Performance Implications
-   **State Size**: Storing profile for 100M users is huge. Use **RocksDB** backend.
-   **Throughput**: CEP is CPU intensive. Scale out.
