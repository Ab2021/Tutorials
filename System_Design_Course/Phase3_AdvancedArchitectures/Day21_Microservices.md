# Day 21: Microservices vs Monolith

## 1. The Monolith
*   **Definition:** One big codebase, one build artifact (JAR/Binary), one deployment.
*   **Pros:** Simple to develop/test/deploy initially. No network latency between calls. ACID transactions easy.
*   **Cons:**
    *   **Coupling:** A bug in one module crashes the whole app.
    *   **Scaling:** Must scale the whole app, even if only one part is hot.
    *   **Tech Stack:** Stuck with one language/framework.
    *   **Deployment:** Slow build/deploy times.

## 2. Microservices
*   **Definition:** Small, independent services communicating over network (HTTP/gRPC).
*   **Pros:**
    *   **Decoupling:** Fault isolation.
    *   **Scaling:** Scale only what's needed.
    *   **Tech Freedom:** Use Python for AI, Go for high concurrency.
    *   **Agility:** Independent teams deploy independently.
*   **Cons:**
    *   **Complexity:** Distributed systems are hard (Network failure, Consistency).
    *   **Ops:** Need advanced monitoring, logging, deployment (Kubernetes).

## 3. Patterns
*   **API Gateway:** Single entry point. Handles Auth, Rate Limiting.
*   **Circuit Breaker:** Stop calling a failing service to prevent cascading failure.
*   **Bulkhead:** Isolate resources (Thread pools) so one failure doesn't consume all threads.

## 4. When to migrate?
*   **Rule:** Don't start with Microservices. Start with a Modular Monolith.
*   **Migrate when:**
    *   Team size grows (> 20 devs).
    *   Build times become unbearable (> 20 mins).
    *   Scaling requirements differ drastically between modules.
