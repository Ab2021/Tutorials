# Day 2: Checkpointing & Savepoints

## Core Concepts & Theory

### Checkpointing (Fault Tolerance)
Automatic, periodic snapshots of the application state.
-   **Purpose**: Recovery from failure.
-   **Mechanism**: Chandy-Lamport algorithm (Barrier alignment).
-   **Consistency**: Guarantees **Exactly-Once** state consistency.

### Savepoints (Operations)
Manual, user-triggered snapshots.
-   **Purpose**: Updates, A/B testing, Rescaling, Migration.
-   **Format**: Canonical format, portable across versions.
-   **Self-Contained**: Includes all necessary metadata.

### Architectural Reasoning
**Barrier Alignment**
Barriers flow with the stream. When an operator receives barriers from all inputs, it snapshots its state.
-   **Exactly-Once**: Wait for all barriers (alignment). No data processed from fast streams while waiting.
-   **At-Least-Once**: Don't wait. Process data as it comes. (Faster, but duplicates possible on recovery).

### Key Components
-   `checkpoint.interval`: How often? (e.g., 1 min).
-   `min.pause.between.checkpoints`: Prevent "Checkpoint Storm".
-   `state.checkpoints.dir`: S3 path.
