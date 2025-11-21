# Day 30: Project - Distributed Key-Value Store

## 1. Goal
Build a sharded, replicated Key-Value store (like DynamoDB/Cassandra) in Go/Python.
*   **Requirements:**
    *   **Partitioning:** Consistent Hashing.
    *   **Replication:** Factor of 3.
    *   **Consistency:** Eventual (Hinted Handoff).
    *   **API:** `PUT(key, val)`, `GET(key)`.

## 2. Architecture
*   **Nodes:** 3 Docker containers (`Node A`, `Node B`, `Node C`).
*   **Client:** Smart client that knows the Hash Ring.
*   **Storage:** In-memory map (for simplicity) + WAL (Append Only File).

## 3. Strategy
1.  **Hash Ring:** Implement Consistent Hashing.
2.  **Gossip:** Nodes ping each other to detect failures.
3.  **Write:** Client sends to Coordinator. Coordinator writes to $W$ replicas.
4.  **Read:** Client reads from $R$ replicas.
