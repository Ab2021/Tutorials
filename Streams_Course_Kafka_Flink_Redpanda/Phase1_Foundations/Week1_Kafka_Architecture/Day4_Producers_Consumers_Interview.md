# Day 4: Producers & Consumers - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: How do you ensure strict ordering of messages?**
    -   *A*: Ensure all messages go to the **same partition** (use the same Key). Set `max.in.flight.requests.per.connection=1` (or 5 if idempotence is on).

2.  **Q: What triggers a Consumer Rebalance?**
    -   *A*: A consumer joining/leaving, a consumer crashing (missed heartbeat), or a topic partition count changing.

3.  **Q: What is the difference between `at-least-once` and `exactly-once`?**
    -   *A*: `at-least-once`: Retries on failure, duplicates possible. `exactly-once`: Transactional guarantees, no duplicates.

### Production Challenges
-   **Challenge**: **Rebalance Storm**.
    -   *Scenario*: Consumers keep joining and leaving in a loop.
    -   *Cause*: Processing takes too long (`max.poll.interval.ms` exceeded). The broker thinks the consumer is dead.
    -   *Fix*: Increase `max.poll.interval.ms` or optimize processing logic.

-   **Challenge**: **Poison Pill**.
    -   *Scenario*: A message crashes the consumer. Consumer restarts, reads same message, crashes again.
    -   *Fix*: Dead Letter Queue (DLQ) + Error handling.

### Troubleshooting Scenarios
**Scenario**: Consumer Lag is increasing.
-   **Check**: Is the processing logic too slow?
-   **Check**: Do you need more consumers (and partitions)?
-   **Check**: Is there a rebalance loop?
