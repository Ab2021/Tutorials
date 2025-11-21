# Day 22 Interview Prep: Event-Driven Architecture

## Q1: What is the difference between Message Queue and Event Bus?
**Answer:**
*   **Message Queue (SQS/RabbitMQ):** Command-oriented. "Do this". Expects a consumer to process it. 1-to-1 usually.
*   **Event Bus (Kafka/EventBridge):** Fact-oriented. "This happened". Doesn't care who listens. 1-to-Many.

## Q2: How to handle duplicate events?
**Answer:**
*   **Idempotency:** The consumer must be able to handle the same event twice without side effects.
*   **Mechanism:**
    *   Store `processed_event_ids` in a DB table.
    *   Check if ID exists before processing.
    *   Use `INSERT IGNORE` or atomic transactions.

## Q3: What are the challenges of CQRS?
**Answer:**
*   **Complexity:** You now have two databases to manage.
*   **Consistency:** The Read DB lags behind the Write DB. User might create an item and not see it immediately in the list.
*   **Mitigation:** UI Optimistic updates.

## Q4: When to use Event Sourcing?
**Answer:**
*   When audit trails are legal requirements (Banking, Healthcare).
*   When you need to reconstruct past states.
*   When you have complex business logic that depends on sequence of events.
*   **Don't use:** For simple CRUD apps.
