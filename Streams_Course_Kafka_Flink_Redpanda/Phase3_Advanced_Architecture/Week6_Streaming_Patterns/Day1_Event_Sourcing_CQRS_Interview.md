# Day 1: Event Sourcing - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the main disadvantage of Event Sourcing?**
    -   *A*: Complexity. Handling eventual consistency, schema evolution, and GDPR (deleting data from an immutable log) is hard.

2.  **Q: How do you handle GDPR "Right to be Forgotten" in Kafka?**
    -   *A*: Crypto-shredding (encrypting user data with a key, and deleting the key) or using short retention with a compacted topic for the "current state" only.

3.  **Q: What is the difference between a Command and an Event?**
    -   *A*: Command = Intent ("CreateOrder"). Can be rejected. Event = Fact ("OrderCreated"). Cannot be rejected, has already happened.

### Production Challenges
-   **Challenge**: **Replay takes too long**.
    -   *Scenario*: Rebuilding a view takes 2 days.
    -   *Fix*: Parallelize the replay (partitioning) or use snapshots.

### Troubleshooting Scenarios
**Scenario**: Read Model is out of sync.
-   *Cause*: The projection job failed or is lagging.
-   *Fix*: Monitor consumer lag. Implement an "anti-entropy" mechanism to compare Write/Read models periodically.
