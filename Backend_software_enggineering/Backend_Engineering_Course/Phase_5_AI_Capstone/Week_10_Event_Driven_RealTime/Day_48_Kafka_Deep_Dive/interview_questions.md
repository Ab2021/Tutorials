# Day 48: Interview Questions & Answers

## Conceptual Questions

### Q1: How does Kafka guarantee message ordering?
**Answer:**
*   **Scope**: Ordering is guaranteed **per partition**, NOT per topic.
*   **Mechanism**: If you send Message A then Message B to Partition 0, Consumer will read A then B.
*   **Routing**: Use a Key (e.g., `user_id`). All messages for `user_id=123` hash to the same partition.

### Q2: What is a "Consumer Rebalance"?
**Answer:**
*   **Event**: A consumer joins or leaves a group.
*   **Action**: Kafka stops all consumers, re-assigns partitions, and resumes.
*   **Impact**: "Stop the World" pause. (Newer protocols like Cooperative Sticky Assignors minimize this).

### Q3: Explain "Log Compaction".
**Answer:**
*   **Standard Retention**: Delete logs older than 7 days.
*   **Compaction**: Keep the *latest value* for each Key.
*   **Use Case**: Storing the current state of a user profile. If User A updated profile 5 times, we only need the last update.

---

## Scenario-Based Questions

### Q4: You have 10 partitions and 15 consumers in a group. What happens?
**Answer:**
*   **Idle Consumers**: 10 consumers will read 1 partition each. 5 consumers will sit idle.
*   **Fix**: Increase partitions to 15 (hard to do dynamically) or reduce consumers.

### Q5: How do you achieve "Exactly-Once Semantics" in Kafka?
**Answer:**
*   **Problem**: Producer sends msg, network fails, Producer retries. Msg stored twice.
*   **Solution**: **Idempotent Producer** (`enable.idempotence=true`). Kafka assigns a PID (Producer ID) and Sequence Number. It de-duplicates retries automatically.
*   **Transactional API**: For "Read-Process-Write" cycles.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use Kafka for a simple job queue. Good idea?
**Answer:**
*   **Maybe not**.
*   **Complexity**: Kafka requires Zookeeper (or KRaft), schema registry, etc.
*   **Features**: Kafka doesn't support "Individual Message Acks" or "DLQ per message" easily like RabbitMQ/SQS.
*   **Verdict**: Use RabbitMQ/SQS for simple job queues. Use Kafka for data streaming/pipelines.
