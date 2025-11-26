# Day 12: Microservice Design & Boundaries

## 1. Where to Draw the Line?

The hardest part of microservices isn't the tech (Kubernetes/Docker); it's the **Boundaries**.
*   **Too Big**: You have a "Distributed Monolith".
*   **Too Small**: You have "Nanoservices" (High latency, impossible to debug).

### 1.1 Decomposition Strategies
1.  **By Business Capability**: Align with what the business *does*.
    *   *Examples*: Order Management, Inventory, Shipping, User Profile.
    *   *Pros*: Easy for business stakeholders to understand.
2.  **By Subdomain (DDD)**: Align with the *complexity* of the problem.
    *   **Core Domain**: The secret sauce (e.g., Recommendation Engine). Keep this pure.
    *   **Supporting Subdomain**: Necessary but not unique (e.g., Catalog).
    *   **Generic Subdomain**: Boring stuff (e.g., Auth, Logging). Buy off-the-shelf or use libraries.

---

## 2. The Golden Rule: Database-per-Service

**Rule**: A microservice's database is **private** to that service. No other service can read/write to it directly.

### 2.1 Why?
*   **Coupling**: If Service A and B share a DB, and A changes a column name, B breaks.
*   **Scaling**: Service A might lock tables that Service B needs.

### 2.2 How to Share Data?
If "Order Service" needs user data:
1.  **API Composition**: Call `GET /users/{id}` from User Service.
2.  **Data Replication (Event-Driven)**:
    *   User Service publishes `UserUpdated` event.
    *   Order Service listens and updates its local `user_replica` table.
    *   *Pros*: Fast reads, decoupled availability.
    *   *Cons*: Eventual consistency.

---

## 3. Communication Patterns

### 3.1 Synchronous (Request/Response)
*   **Protocol**: HTTP/REST or gRPC.
*   **Flow**: Client -> Service A -> Service B.
*   **Pros**: Simple, real-time.
*   **Cons**:
    *   **Cascading Failure**: If B is down, A is down.
    *   **Latency**: A waits for B.

### 3.2 Asynchronous (Event-Driven)
*   **Protocol**: Message Broker (Kafka, RabbitMQ).
*   **Flow**: Service A emits event -> Broker -> Service B consumes.
*   **Pros**:
    *   **Decoupling**: A doesn't know B exists.
    *   **Buffering**: If B is down, messages queue up until it recovers.
*   **Cons**: Complexity (Tracing, Dead Letter Queues).

---

## 4. Designing for Failure

In a distributed system, failure is guaranteed.

### 4.1 Circuit Breaker
If Service B fails 50% of the time, stop calling it.
*   **Open State**: Fail immediately (don't wait for timeout).
*   **Half-Open**: Let 1 request through to test if B is back.
*   **Closed**: Normal operation.

### 4.2 Bulkhead Pattern
Isolate resources.
*   Don't let one slow downstream service exhaust all connection threads.
*   *Analogy*: Ship compartments. If one floods, the ship stays afloat.

---

## 5. Summary

Today we learned how to architect for autonomy.
*   **Boundaries**: Follow the business capabilities.
*   **Data**: Keep databases private.
*   **Communication**: Prefer Async for decoupling, Sync for simple queries.

**Tomorrow (Day 13)**: We will solve the problem of "How does the Frontend talk to 50 microservices?" by introducing the **API Gateway** and **BFF** patterns.
