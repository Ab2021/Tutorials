# Day 47: Interview Questions & Answers

## Conceptual Questions

### Q1: RabbitMQ vs Kafka. When to use which?
**Answer:**
*   **RabbitMQ**:
    *   **Complex Routing**: Need Fanout/Topic routing.
    *   **Transient Data**: Process and delete.
    *   **Push Model**: Broker pushes to consumer.
*   **Kafka**:
    *   **High Throughput**: Millions of events/sec.
    *   **Event Sourcing**: Replay old events (Log storage).
    *   **Pull Model**: Consumer pulls from broker.

### Q2: What is "Prefetch Count" in RabbitMQ?
**Answer:**
*   **Problem**: RabbitMQ pushes messages as fast as possible. If Consumer A is slow and Consumer B is fast, A gets overloaded while B sits idle.
*   **Solution**: Set `prefetch=1`.
*   **Effect**: RabbitMQ only sends 1 message to a consumer at a time. It waits for `ack` before sending the next. Ensures **Fair Dispatch**.

### Q3: What happens if a Queue is full?
**Answer:**
*   **Default**: RabbitMQ uses RAM. If RAM fills up, it pages to disk (slow).
*   **Limits**: You can set a `max-length`.
*   **Overflow Behavior**: Drop oldest, Reject new, or Dead Letter.

---

## Scenario-Based Questions

### Q4: You need to implement a "Delayed Job" (e.g., Send email in 15 mins). Can RabbitMQ do this?
**Answer:**
*   **Natively**: No (unlike SQS).
*   **Workaround**:
    1.  **TTL + DLQ**: Send message to `WaitQueue` with `TTL=15min`.
    2.  `WaitQueue` has no consumers.
    3.  When TTL expires, RabbitMQ moves it to the `DLQ` (Dead Letter Queue).
    4.  Consumers listen to `DLQ`.

### Q5: Messages are getting lost during a broker restart. Why?
**Answer:**
*   **Cause**:
    1.  Queue was not declared as `durable`.
    2.  Message was not published as `persistent` (delivery_mode=2).
*   **Fix**: Enable both. (Performance penalty: Disk I/O).

---

## Behavioral / Role-Specific Questions

### Q6: A developer uses RabbitMQ as a database to store user sessions. Is this okay?
**Answer:**
*   **No**.
*   **Anti-Pattern**: Queues are for *moving* data, not *storing* it.
*   **Risk**: If the queue grows too large, RabbitMQ performance degrades significantly.
*   **Advice**: Use Redis for sessions.
