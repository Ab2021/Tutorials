# Day 14: Event-Driven Architecture (EDA) & Message Brokers

## 1. Stop Calling Me!

In synchronous systems (REST/gRPC), services call each other.
*   **Problem**: Service A needs B. B needs C. If C is slow, A is slow. If C is down, A is down.
*   **Solution**: Invert the dependency. Service A yells "I did something!" (Event). Service B and C listen and react.

### 1.1 Events vs Commands
*   **Command**: "Create Order". (Intent. Expects a result. Can fail).
*   **Event**: "Order Created". (Fact. Already happened. Immutable).

### 1.2 Pub/Sub Pattern
*   **Publisher**: Emits events to a Topic. Doesn't know who listens.
*   **Subscriber**: Listens to a Topic. Doesn't know who publishes.
*   **Broker**: The middleman (Kafka, RabbitMQ) that manages the message flow.

---

## 2. The Brokers: Kafka vs RabbitMQ

### 2.1 RabbitMQ (The Smart Broker)
*   **Model**: Queue-based.
*   **Mechanism**: Producer sends to Exchange -> Exchange routes to Queues -> Consumer reads.
*   **Consumption**: Destructive. Once a message is ack'd, it's gone from the queue.
*   **Use Case**: Complex routing, task queues (Celery), simple pub/sub.

### 2.2 Apache Kafka (The Dumb Broker / Smart Consumer)
*   **Model**: Log-based.
*   **Mechanism**: Producer appends to a Log (Topic). Consumer reads from an Offset.
*   **Consumption**: Non-destructive. Messages stay for X days. Multiple consumers can read the same message at different speeds.
*   **Use Case**: High throughput, Event Sourcing, Data Streaming, Analytics.

---

## 3. Reliability Patterns

### 3.1 At-Least-Once Delivery
*   **Guarantee**: The message will be delivered *at least* once. It might be delivered twice (e.g., if consumer crashes before ack).
*   **Implication**: Your consumers must be **Idempotent**.
    *   *Bad*: `UPDATE account SET balance = balance - 10` (Run twice = -20).
    *   *Good*: `UPDATE account SET balance = balance - 10 WHERE transaction_id NOT IN (processed_ids)`.

### 3.2 The Transactional Outbox Pattern
*   **Problem**: You save to DB and publish to Kafka. What if DB commits but Kafka fails? (Inconsistency).
*   **Solution**:
    1.  Start DB Transaction.
    2.  Insert into `Users` table.
    3.  Insert into `Outbox` table (in the same DB).
    4.  Commit DB Transaction. (Atomic).
    5.  **Relay Process**: A background worker reads `Outbox` and publishes to Kafka. If it fails, it retries.

---

## 4. Designing Events

### 4.1 Fat vs Thin Events
*   **Thin Event**: `{"order_id": 123}`.
    *   *Pros*: Small.
    *   *Cons*: Consumer must call API to get details (Coupling).
*   **Fat Event**: `{"order_id": 123, "items": [...], "total": 50}`.
    *   *Pros*: Consumer has all data. No API call needed.
    *   *Cons*: Large payload. Schema evolution is harder.
*   **Verdict**: Prefer **Fat Events** (Event-Carried State Transfer) for true decoupling.

---

## 5. Summary

Today we cut the cord.
*   **EDA**: Decouples services in time and space.
*   **Brokers**: Kafka for streams, RabbitMQ for queues.
*   **Idempotency**: Essential for handling retries.

**Tomorrow (Day 15)**: We will take EDA to the extreme with **Event Sourcing** and **CQRS**. We won't just emit events; we will *store* them as our source of truth.
