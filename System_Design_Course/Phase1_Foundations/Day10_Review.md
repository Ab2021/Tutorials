# Day 10: Phase 1 Review

## 1. Summary
We have covered the foundational building blocks of distributed systems.
*   **Scalability:** Vertical vs Horizontal.
*   **Networking:** TCP/UDP, HTTP/2, gRPC.
*   **API:** REST, GraphQL, Idempotency.
*   **DB:** ACID, CAP, B-Trees vs LSM.
*   **Caching:** Strategies, Eviction, Thundering Herd.
*   **Replication:** Leader-Follower, Consistent Hashing.
*   **Consensus:** Paxos, Raft.
*   **Transactions:** 2PC, Sagas.

## 2. The System Design Template (4 Steps)
Use this in every interview.
1.  **Clarify Requirements:** Functional (What it does), Non-Functional (Scale, Latency).
2.  **Back-of-Envelope:** QPS, Storage, Bandwidth.
3.  **High-Level Design:** Draw the boxes (LB, App, DB, Cache).
4.  **Deep Dive:** Scale individual components (Sharding, Replication, Async).

## 3. Next Phase
We will move to **Building Blocks** (Load Balancers, Queues, Search Engines) and start writing more code.
