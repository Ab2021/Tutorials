# Day 21 Interview Prep: Microservices

## Q1: Shared DB vs Database per Service?
**Answer:**
*   **Shared DB:** Easy data consistency and joins. Bad coupling. If one service locks a table, others wait.
*   **DB per Service:** Good decoupling. Hard data consistency (Eventual Consistency) and joins (API Composition). This is the standard for Microservices.

## Q2: How to handle distributed transactions?
**Answer:**
*   **2PC (Two-Phase Commit):** Strong consistency but blocking and slow.
*   **Saga Pattern:** Sequence of local transactions. If one fails, run compensating transactions. Eventual consistency. Preferred for Microservices.

## Q3: What is a Service Mesh?
**Answer:**
*   Infrastructure layer for service-to-service communication.
*   **Sidecar Proxy (Envoy):** Handles Retries, Timeouts, Circuit Breaking, mTLS, Tracing.
*   **Control Plane (Istio):** Configures the proxies.
*   **Benefit:** Moves logic (Retries/Auth) out of application code.

## Q4: How to test Microservices?
**Answer:**
*   **Unit Test:** Test internal logic.
*   **Integration Test:** Test interaction with DB/Broker.
*   **Contract Test (Pact):** Ensure Service A sends what Service B expects.
*   **End-to-End (E2E):** Test full flow. Expensive and flaky. Minimize these.
