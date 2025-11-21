# Day 11 Deep Dive: Client-Side vs Server-Side LB

## 1. Server-Side Load Balancing
*   **Architecture:** Client -> LB (Hardware/Software) -> Server Pool.
*   **Pros:** Simple for client. Discovery is handled by LB.
*   **Cons:** LB is a Single Point of Failure (SPOF) and bottleneck. Extra hop (latency).
*   **Example:** AWS ELB, Nginx.

## 2. Client-Side Load Balancing
*   **Architecture:** Client asks Service Registry (Eureka) for list of IPs. Client picks one and connects directly.
*   **Pros:** No bottleneck. No extra hop. Resilient.
*   **Cons:** Complex client logic. Trusted clients only (Internal Microservices).

## 3. Case Study: Netflix Ribbon
*   **Context:** Netflix moved from Monolith to Microservices. Hardware LBs (F5) were too slow/expensive to scale.
*   **Solution:** Ribbon (Java Library).
*   **Mechanism:**
    1.  **Discovery:** Ribbon talks to Eureka (Registry) to get list of "Recommendation Service" nodes.
    2.  **Caching:** Caches list locally.
    3.  **Strategy:** Uses "Zone Aware Round Robin" (Prefer servers in same AWS Availability Zone to save cost/latency).
    4.  **Resilience:** If a call fails, Ribbon retries on a different node instantly.

## 4. Case Study: Consistent Hashing in LBs (Discord)
*   **Problem:** Discord Voice Channels. Users in a channel must be routed to the *same* voice server.
*   **Solution:** Consistent Hashing Ring.
*   **Benefit:** If a voice server crashes, only its users are moved. Other channels stay put.
