# Day 46: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between Orchestration and Choreography?
**Answer:**
*   **Orchestration (Conductor)**: A central service (e.g., `OrderService`) tells everyone what to do. "Inventory, reserve items. Payment, charge card." (Tight control, single point of failure).
*   **Choreography (Dancers)**: Services react to events. `OrderService` emits `OrderPlaced`. `Inventory` hears it and reserves items. `Payment` hears it and charges. (Loose coupling, harder to track workflow).

### Q2: What does "At-Least-Once Delivery" mean?
**Answer:**
*   **Guarantee**: The broker ensures the message is delivered *at least once*.
*   **Implication**: It might be delivered *twice* (e.g., if consumer crashes before ack).
*   **Requirement**: Consumers must be **Idempotent** (handling the same message twice shouldn't break things).

### Q3: How do you handle "Dead Letter Queues" (DLQ)?
**Answer:**
*   **Scenario**: A message is malformed or causes a crash. Consumer retries 3 times and fails.
*   **DLQ**: The broker moves the bad message to a separate queue (DLQ) so it doesn't block other messages.
*   **Action**: Engineers inspect the DLQ, fix the bug, and re-process the messages.

---

## Scenario-Based Questions

### Q4: You migrated from REST to EDA, but now users complain "I don't see my order in the history immediately". Why?
**Answer:**
*   **Cause**: **Eventual Consistency**. The `OrderHistoryService` hasn't processed the event yet.
*   **Fix**:
    *   **UI**: Show a "Processing..." state.
    *   **Read-Your-Writes**: Query the `OrderService` (source of truth) for the immediate confirmation, then switch to `HistoryService` later.

### Q5: How do you ensure order of events? (e.g., `OrderCreated` must be processed before `OrderShipped`)
**Answer:**
*   **FIFO Queue**: Use a queue that guarantees order (e.g., RabbitMQ Queue, Kafka Partition).
*   **Partitioning**: Ensure all events for `Order #123` go to the *same* partition/consumer.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use EDA for *everything*, including fetching user profiles. Good idea?
**Answer:**
*   **No**.
*   **Rule**: Use Async for *Actions* (Side effects). Use Sync (REST/gRPC) for *Queries* (Data retrieval).
*   **Reason**: Waiting for an event to return a user profile is slow and complex (Request-Reply pattern over queues is an anti-pattern).
