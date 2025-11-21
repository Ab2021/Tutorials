# Day 22: Event-Driven Architecture (EDA)

## 1. Request-Response vs Event-Driven
### Request-Response (REST/gRPC)
*   **Synchronous:** Caller waits for answer.
*   **Coupling:** Caller must know Callee's address.
*   **Flow:** `OrderService` calls `InventoryService`.

### Event-Driven (Pub/Sub)
*   **Asynchronous:** Fire and forget.
*   **Decoupling:** Producer emits event. Doesn't know who listens.
*   **Flow:** `OrderService` emits `OrderCreated`. `InventoryService` listens. `EmailService` listens.

## 2. Components
*   **Producer:** Generates events.
*   **Broker:** Ingests, stores, and routes events (Kafka, RabbitMQ).
*   **Consumer:** Processes events.

## 3. Patterns
*   **Event Notification:** "Something changed". Payload is small (`{id: 123}`). Consumer calls back to get details.
*   **Event-Carried State Transfer:** Payload contains all data (`{id: 123, name: "Bob", email: "..."}`). Consumer updates local cache. No callback needed.
*   **Event Sourcing:** Store *events*, not current state. State is derived by replaying events.

## 4. Pros & Cons
*   **Pros:** Decoupling, Scalability, Extensibility (Add new consumer without changing producer).
*   **Cons:** Complexity, Debugging (Flow is implicit), Eventual Consistency.
