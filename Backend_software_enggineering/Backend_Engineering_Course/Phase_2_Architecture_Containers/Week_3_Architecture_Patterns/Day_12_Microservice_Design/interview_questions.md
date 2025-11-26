# Day 12: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the "Saga Pattern"?
**Answer:**
*   **Problem**: How to maintain data consistency across microservices without distributed transactions (2PC).
*   **Solution**: A sequence of local transactions. Each service updates its DB and publishes an event to trigger the next step.
*   **Rollback**: If a step fails, you must execute **Compensating Transactions** to undo previous steps (e.g., "Refund Payment" if "Ship Item" fails).
*   **Types**:
    *   *Choreography*: Services talk to each other via events (Decentralized).
    *   *Orchestration*: A central "Orchestrator" service tells others what to do.

### Q2: Why is Two-Phase Commit (2PC) bad for Microservices?
**Answer:**
*   **Blocking**: The coordinator locks resources on all participants until everyone agrees.
*   **Latency**: The slowest node dictates the speed.
*   **SPOF**: If the coordinator dies, everyone hangs.
*   *Verdict*: It kills availability and scalability. Use Sagas (Eventual Consistency) instead.

### Q3: What is the "Database-per-Service" pattern?
**Answer:**
*   Each microservice has its own private database schema.
*   Other services cannot access it directly (no SQL joins across services).
*   They must use the Service's API.
*   *Benefit*: Loose coupling. You can change your schema without breaking others.

---

## Scenario-Based Questions

### Q4: You have a "User Service" and a "Notification Service". When a user is created, we must send an email. If the email fails, should we roll back the user creation?
**Answer:**
*   **Usually No**.
*   **Reasoning**: User creation is the core business value. Sending an email is a side effect.
*   **Approach**:
    1.  Create User (Commit).
    2.  Publish `UserCreated` event.
    3.  Notification Service consumes event -> Sends Email.
    4.  **Failure Handling**: If email fails, retry (Exponential Backoff). If it fails permanently, log it or alert support. Do not delete the user.

### Q5: You are designing a "Product Service". Should "Inventory" (Stock count) be inside it or a separate service?
**Answer:**
*   **It depends on complexity**.
*   **Simple**: If inventory is just a number (`stock: 10`), keep it in Product Service.
*   **Complex**: If inventory involves multiple warehouses, shipping logic, and reservations, split it into `Inventory Service`.
*   **Heuristic**: Does "Inventory" change for different reasons than "Product Description"? Yes -> Split.

---

## Behavioral / Role-Specific Questions

### Q6: A developer creates a "Common" library shared by all microservices. It contains DTOs, Utils, and DB Models. Is this good?
**Answer:**
**Risky.**
*   **Pros**: DRY (Don't Repeat Yourself).
*   **Cons**: **Coupling**. If you change a DTO in the library, you might force *all* services to upgrade and redeploy.
*   **Better Approach**:
    *   Share *stable* code (Logging, Auth middleware).
    *   Duplicate *volatile* code (DTOs, Domain Logic). "Duplication is better than wrong abstraction/coupling."
