# Day 46: Event-Driven Architecture (EDA)

## 1. The Problem with REST

*   **Scenario**: User places an order.
*   **Monolith**: `OrderService` calls `InventoryService`, `PaymentService`, `EmailService`.
*   **Problem**:
    *   **Latency**: User waits for *all* services to finish (1s + 1s + 1s = 3s).
    *   **Coupling**: If `EmailService` is down, the whole Order fails.
    *   **Tight Coupling**: `OrderService` needs to know about everyone.

---

## 2. The Solution: Events

*   **Concept**: Fire and Forget.
*   **Flow**:
    1.  User places order.
    2.  `OrderService` saves order to DB.
    3.  `OrderService` publishes an event: `OrderPlaced`.
    4.  `OrderService` returns "Success" to user immediately (100ms).
    5.  **Subscribers** (`Inventory`, `Email`) listen for the event and do their work in the background.

---

## 3. Core Concepts

### 3.1 Producer (Publisher)
The service that creates the event. It doesn't know who is listening.

### 3.2 Consumer (Subscriber)
The service that reacts to the event.

### 3.3 Broker (The Post Office)
The middleware (RabbitMQ, Kafka) that receives events and delivers them to consumers.

---

## 4. Pros & Cons

*   **Pros**:
    *   **Decoupling**: Services evolve independently.
    *   **Scalability**: Scale consumers based on load.
    *   **Resilience**: If Email is down, the event stays in the queue until it comes back up.
*   **Cons**:
    *   **Complexity**: Harder to debug (Distributed Tracing is mandatory).
    *   **Consistency**: Eventual Consistency (User might not see the email immediately).

---

## 5. Summary

Today we cut the wires.
*   **Sync**: Phone call (Both must be on the line).
*   **Async**: Text message (Read it when you can).
*   **EDA**: The backbone of modern microservices.

**Tomorrow (Day 47)**: We will implement this using **RabbitMQ**.
