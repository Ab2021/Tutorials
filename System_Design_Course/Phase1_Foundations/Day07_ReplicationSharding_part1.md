# Day 7 Deep Dive: Consistent Hashing

## 1. The Problem with Modulo N
*   $N=4$. Key 10. $10 \% 4 = 2$. (Node 2).
*   Node 3 dies. $N=3$. $10 \% 3 = 1$. (Node 1).
*   **Result:** Massive data movement. Cache flush. DB overload.

## 2. The Ring (Consistent Hashing)
*   Imagine a circle with values $0$ to $2^{32}-1$.
*   Place Servers on the ring: $H(Server1)$, $H(Server2)$...
*   Place Keys on the ring: $H(Key1)$.
*   **Lookup:** Go clockwise from Key until you hit a Server.

## 3. Virtual Nodes (VNodes)
*   **Problem:** If we have few servers, distribution might be uneven (Server A gets 90% of ring).
*   **Solution:** Create "Virtual Nodes".
    *   Server A maps to 100 points on the ring: $A_1, A_2, ... A_{100}$.
    *   Server B maps to 100 points: $B_1, B_2, ... B_{100}$.
*   **Benefit:**
    *   **Uniform Distribution:** Better load balancing.
    *   **Heterogeneity:** Powerful servers can have more VNodes.

## 4. Gossip Protocol
*   **How do nodes know about each other?**
*   **Mechanism:** Each node periodically picks a random node and exchanges state ("I am alive", "Node C is dead").
*   **Epidemic:** Information spreads like a virus ($O(\log N)$).
*   **Used by:** Cassandra, DynamoDB, Riak.
