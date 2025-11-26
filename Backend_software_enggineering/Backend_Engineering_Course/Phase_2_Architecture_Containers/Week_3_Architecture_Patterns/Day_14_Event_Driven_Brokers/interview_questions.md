# Day 14: Interview Questions & Answers

## Conceptual Questions

### Q1: Explain "Consumer Groups" in Kafka.
**Answer:**
*   **Concept**: A way to parallelize consumption.
*   **Mechanism**: If you have a Topic with 4 Partitions and a Consumer Group with 2 Consumers, each Consumer reads from 2 Partitions.
*   **Scaling**: If you add 2 more Consumers (Total 4), each reads from 1 Partition.
*   **Limit**: You cannot have more Consumers than Partitions (extras will sit idle).

### Q2: What is a Dead Letter Queue (DLQ)?
**Answer:**
*   **Problem**: A consumer fails to process a message (e.g., JSON parse error). If it retries forever, it blocks the queue ("Poison Pill").
*   **Solution**: After N retries, move the message to a separate queue (DLQ).
*   **Action**: Engineers monitor the DLQ, fix the bug/data, and manually re-process or discard the messages.

### Q3: How do you guarantee message ordering in Kafka?
**Answer:**
*   **Scope**: Kafka only guarantees ordering **within a partition**, not across the whole topic.
*   **Strategy**: Use a **Partition Key** (e.g., `user_id`).
*   **Result**: All events for `user_123` will go to Partition 1. Since Partition 1 is read sequentially, the events for that user are ordered.

---

## Scenario-Based Questions

### Q4: You need to migrate a legacy system to EDA. The legacy system cannot publish events. What do you do?
**Answer:**
*   **Pattern**: **CDC (Change Data Capture)**.
*   **Tool**: Debezium.
*   **Mechanism**: Debezium connects to the legacy DB's transaction log (WAL in Postgres, Binlog in MySQL). Whenever a row is inserted/updated, Debezium detects it and publishes an event to Kafka.
*   **Benefit**: No code changes needed in the legacy app.

### Q5: Your consumer processed a message, updated the DB, but crashed before sending the ACK to the broker. What happens?
**Answer:**
*   **Result**: The broker thinks the message failed. It re-delivers the message to another consumer (or the same one after restart).
*   **Impact**: The DB update runs twice.
*   **Fix**: **Idempotency**. The consumer must check "Have I processed this Message ID?" before updating the DB.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use Kafka for a "Request-Response" flow (waiting for a reply). Is this good?
**Answer:**
**Generally No.**
*   **Reasoning**: Kafka is asynchronous. Using it for sync RPC (Request -> Wait -> Reply) introduces high latency and complexity (Correlation IDs, temporary reply queues).
*   **Alternative**: Use HTTP/gRPC for direct queries. Use Kafka for "Fire and Forget" notifications or background processing.
