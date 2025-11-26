# Day 11: Monolith vs Microservices

## 1. The Great Debate

The transition from Monolith to Microservices is the most significant architectural shift of the last decade. But it is not a silver bullet.

### 1.1 The Monolith
A single codebase, single build artifact, single process (usually).
*   **Pros**:
    *   **Simplicity**: Easy to develop, test, and deploy (copy 1 file).
    *   **Performance**: Function calls are in-memory (nanoseconds), not over network (milliseconds).
    *   **Transactionality**: ACID transactions across all data are easy.
*   **Cons**:
    *   **Coupling**: A bug in the "Billing" module can crash the "User" module.
    *   **Scaling**: You must scale the *entire* app, even if only one part is heavy.
    *   **Tech Stack Lock-in**: Hard to try Go if the app is 10GB of Java.

### 1.2 Microservices
Small, autonomous services that work together.
*   **Pros**:
    *   **Independent Scaling**: Scale "Image Processing" service x10, keep "User" service x1.
    *   **Fault Isolation**: If "Billing" crashes, users can still browse products.
    *   **Tech Freedom**: Use Python for AI, Go for high-load, Node for IO.
*   **Cons**:
    *   **Complexity**: Distributed systems are hard (Network failure, Latency, Consistency).
    *   **Ops Overhead**: You need K8s, Service Mesh, Observability.
    *   **Data Consistency**: No cross-service ACID transactions (Sagas needed).

---

## 2. The Scale Cube

How do we scale? The "AKF Scale Cube" defines 3 dimensions:
1.  **X-Axis (Horizontal Duplication)**: Run multiple copies of the same monolith behind a load balancer. (Easy, standard).
2.  **Y-Axis (Functional Decomposition)**: Split by function/service (Microservices). e.g., Auth Service, Cart Service.
3.  **Z-Axis (Data Partitioning)**: Sharding. Users A-M go to Server 1, N-Z go to Server 2.

---

## 3. The "Distributed Monolith" (Anti-Pattern)

This is the worst of both worlds.
*   **Symptoms**:
    *   Services share a single database.
    *   Services are "chatty" (Service A calls B, which calls C, which calls A).
    *   You must deploy Service A and B together.
*   **Result**: You have the complexity of microservices with the coupling of a monolith.

---

## 4. Bounded Contexts (Domain-Driven Design)

How do you decide where to cut the monolith?
*   **Bounded Context**: A linguistic boundary within which a specific domain model applies.
*   **Example**:
    *   In **Sales Context**, a "Product" has a price, description, and image.
    *   In **Shipping Context**, a "Product" has weight, dimensions, and HS code.
    *   *Don't* try to make one giant "Product" class. Create two services with their own definitions.

---

## 5. Migration Strategy: Strangler Fig Pattern

Don't rewrite from scratch.
1.  **Identify** a non-critical edge capability (e.g., "Email Notifications").
2.  **Build** a new microservice for it.
3.  **Route** traffic to the new service (via Gateway).
4.  **Remove** old code from monolith.
5.  **Repeat**.

---

## 6. Summary

Today we learned that Microservices are a trade-off: **Complexity for Velocity/Scale**.
*   Start with a **Modular Monolith**.
*   Split only when necessary (Team size > 10, or conflicting scaling needs).
*   Avoid the **Distributed Monolith**.

**Tomorrow (Day 12)**: We will get practical. We will design the boundaries for an E-commerce system and learn how to handle data ownership.
