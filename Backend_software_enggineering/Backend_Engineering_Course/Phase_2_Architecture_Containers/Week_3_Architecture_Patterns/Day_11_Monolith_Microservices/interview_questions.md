# Day 11: Interview Questions & Answers

## Conceptual Questions

### Q1: When should you *not* use Microservices?
**Answer:**
*   **Team Size**: If you are a startup with 3 engineers, the Ops overhead of microservices will kill your velocity.
*   **Domain Clarity**: If you don't understand the domain boundaries yet, you will split it wrong, leading to a Distributed Monolith.
*   **Performance**: If latency is critical (e.g., High Frequency Trading), network hops are too expensive.

### Q2: What is the "Strangler Fig Pattern"?
**Answer:**
*   A migration strategy to move from Monolith to Microservices.
*   Instead of a "Big Bang" rewrite, you wrap the old system with a proxy.
*   You build new features as microservices and route traffic to them.
*   Over time, the monolith "strangled" and replaced.

### Q3: How do Microservices communicate?
**Answer:**
1.  **Synchronous**: HTTP/REST or gRPC. (Simple, but couples availability).
2.  **Asynchronous**: Message Queues (Kafka, RabbitMQ). (Decoupled, complex).

---

## Scenario-Based Questions

### Q4: You split a Monolith into "Order Service" and "User Service". Now "Order Service" needs the user's email to send a receipt. How do you handle this?
**Answer:**
*   **Bad**: Order Service calls User Service API on every order (Coupling).
*   **Bad**: Order Service queries User DB (Shared Database).
*   **Good (Data Replication)**: When a User is updated, publish a `UserUpdated` event. Order Service listens and updates its local `UserReplica` table.
*   **Acceptable (ID Token)**: Pass the email in the JWT token when creating the order.

### Q5: Your microservices are failing because Service A calls B, B calls C, and C is down. What pattern fixes this?
**Answer:**
*   **Circuit Breaker**: Service A should wrap the call to B in a circuit breaker. If B fails repeatedly, the breaker "opens" and A returns a fallback (or error) instantly without waiting for timeout, preventing cascading failure.

---

## Behavioral / Role-Specific Questions

### Q6: The CTO wants to move to Microservices because "Netflix does it". How do you respond?
**Answer:**
*   **Acknowledge**: Netflix is great, but they have 1000+ engineers and solve problems we don't have yet.
*   **Assess**: Do we have the problems Microservices solve? (Deployment bottlenecks, scaling conflicts, team coordination issues?)
*   **Proposal**: Suggest a **Modular Monolith** first. Enforce strict boundaries within the code (no cross-module imports). If that works, extracting a service later is easy. If we can't discipline code, we can't discipline distributed systems.
