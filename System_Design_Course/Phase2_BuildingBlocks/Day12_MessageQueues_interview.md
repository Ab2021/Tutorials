# Day 12 Interview Prep: Message Queues

## Q1: Kafka vs RabbitMQ?
**Answer:**
*   **RabbitMQ:** Traditional Queue. Good for complex routing (Exchanges), task queues, and where latency < 1ms is needed. Deletes message after consumption.
*   **Kafka:** Streaming Platform. Good for high throughput, replayability (retention), and event sourcing. Persists messages on disk.

## Q2: How does Kafka guarantee ordering?
**Answer:**
*   Kafka guarantees ordering **only within a partition**.
*   It does NOT guarantee ordering across the whole topic.
*   To ensure order for a specific entity (e.g., User actions), use the UserID as the Partition Key. All events for UserID=123 will go to Partition 5 and be ordered.

## Q3: What happens if a Consumer fails?
**Answer:**
*   **RabbitMQ:** Message is returned to queue (NACK) and redelivered to another consumer.
*   **Kafka:** Consumer Group Rebalancing. The partitions owned by the dead consumer are reassigned to other living consumers.

## Q4: How to handle failed messages (Dead Letter Queue)?
**Answer:**
*   If a message crashes the consumer (poison pill), don't block the queue.
*   Move the message to a separate "Dead Letter Queue" (DLQ) for manual inspection.
*   Alert the team.
