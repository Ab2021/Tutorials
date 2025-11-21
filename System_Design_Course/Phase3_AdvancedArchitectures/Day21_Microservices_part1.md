# Day 21 Deep Dive: Migration Patterns

## 1. The Strangler Fig Pattern
*   **Metaphor:** A vine that grows around a tree and eventually kills it.
*   **Strategy:**
    1.  Put an **API Gateway** in front of the Monolith.
    2.  Identify one functionality to extract (e.g., "User Profile").
    3.  Build a new Microservice for "User Profile".
    4.  Point the Gateway to the new Service for `/users`.
    5.  Repeat until Monolith is gone.
*   **Benefit:** Low risk. Incremental.

## 2. Database per Service
*   **Anti-Pattern:** Shared Database. (If Service A changes schema, Service B breaks).
*   **Pattern:** Each service has its own DB.
*   **Challenge:** How to join data?
    *   **API Composition:** Gateway calls Service A and Service B, then merges results.
    *   **Data Replication:** Service A publishes events ("UserCreated"). Service B listens and stores a local copy of User data.

## 3. Case Study: Uber's Migration
*   **2011:** Monolith (Python/Postgres).
*   **Problem:** 1000s of engineers. Merge conflicts. Deployment blocked by one bad commit.
*   **Solution:** 4000+ Microservices.
*   **Result:** Extreme agility but high complexity.
*   **2020 (DOMA):** Domain-Oriented Microservice Architecture. Grouping microservices into "Domains" (Gateways) to reduce complexity. "Macroservices".

## 4. Circuit Breaker (Hystrix/Resilience4j)
*   **State: Closed:** Traffic flows normally.
*   **State: Open:** Errors exceeded threshold (e.g., 50%). Block all requests immediately (Fail fast).
*   **State: Half-Open:** After timeout, let one request through. If success, Close. If fail, Open.
