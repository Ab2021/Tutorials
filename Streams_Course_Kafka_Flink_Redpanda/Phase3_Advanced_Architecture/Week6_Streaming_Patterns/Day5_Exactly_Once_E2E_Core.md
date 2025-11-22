# Day 5: Exactly-Once End-to-End

## Core Concepts & Theory

### The Guarantee
"Exactly-Once" usually means **Exactly-Once State Processing**.
"End-to-End Exactly-Once" means **Source + Processing + Sink** guarantees no duplicates in the external system.

### Requirements
1.  **Source**: Must be replayable (Kafka).
2.  **Processing**: Deterministic state (Flink Checkpointing).
3.  **Sink**: Must be Transactional (Two-Phase Commit) or Idempotent.

### Two-Phase Commit (2PC)
-   **Phase 1 (Pre-Commit)**: Flink writes data to the sink (e.g., Kafka "pending" transaction). Happens continuously.
-   **Phase 2 (Commit)**: When Flink completes a checkpoint, it tells the sink to "Commit" the transaction.

### Architectural Reasoning
**Idempotency vs Transactions**
-   **Idempotent Sink**: `PUT(key, val)`. If you retry, it just overwrites. Simple, fast. (Redis, Cassandra, Elastic).
-   **Transactional Sink**: `BEGIN... INSERT... COMMIT`. Needed for append-only systems (Kafka, Files) or multi-row updates (RDBMS).

### Key Components
-   `TwoPhaseCommitSinkFunction`
-   `KafkaSink` (EOS mode)
