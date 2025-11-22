# Day 2: Checkpointing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: Explain the Chandy-Lamport algorithm in Flink.**
    -   *A*: It uses "Barriers" injected into the stream. Operators snapshot state when they receive barriers. It allows consistent snapshots without stopping the world.

2.  **Q: What is the difference between a Checkpoint and a Savepoint?**
    -   *A*: Checkpoints are automatic for recovery. Savepoints are manual for operations (upgrades/rescaling).

3.  **Q: Why would you use Unaligned Checkpoints?**
    -   *A*: To allow checkpoints to succeed even when the network is saturated (backpressure), at the cost of larger storage.

### Production Challenges
-   **Challenge**: **Checkpoints timing out**.
    -   *Cause*: State is too big, network to S3 is slow, or backpressure is delaying barriers.
    -   *Fix*: Incremental checkpoints, Unaligned checkpoints, or optimize state.

-   **Challenge**: **State Processor API**.
    -   *Scenario*: You need to fix a bug in the state (e.g., remove bad keys) inside a Savepoint.
    -   *Fix*: Use the State Processor API to read/write Savepoints offline.

### Troubleshooting Scenarios
**Scenario**: Job stuck in "In Progress" checkpoint loop.
-   *Cause*: One operator is stuck (infinite loop or deadlock) and not processing the barrier.
