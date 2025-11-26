# Day 48: Apache Kafka Deep Dive

## 1. The Distributed Log

Kafka is not a queue. It's a **Distributed Commit Log**.
*   **Queue (RabbitMQ)**: Message is removed after consumption.
*   **Log (Kafka)**: Message stays forever (or until retention expires). Consumers just move their "cursor" (Offset).

---

## 2. Architecture

### 2.1 Topic
A category of messages (e.g., `orders`).

### 2.2 Partition
Topics are split into **Partitions** (P0, P1, P2).
*   **Scaling**: P0 is on Server A, P1 on Server B. Allows parallel processing.
*   **Ordering**: Guaranteed *only within a partition*.

### 2.3 Offset
A unique ID for a message in a partition (0, 1, 2...).

---

## 3. Consumer Groups

The magic of Kafka scaling.
*   **Scenario**: Topic `orders` has 4 partitions.
*   **Group G1**:
    *   Consumer A reads P0, P1.
    *   Consumer B reads P2, P3.
*   **Scale Out**: Add Consumer C.
    *   A: P0, P1.
    *   B: P2.
    *   C: P3.
*   **Rule**: A partition is consumed by *only one* consumer in a group.

---

## 4. Durability

*   **Replication Factor**: 3.
*   **Leader**: Handles R/W for a partition.
*   **Follower**: Passively replicates data.
*   **ISR (In-Sync Replica)**: A follower that is up-to-date.

---

## 5. Summary

Today we scaled to millions.
*   **Partitions**: Parallelism.
*   **Consumer Groups**: Load Balancing.
*   **Offsets**: Replayability.

**Tomorrow (Day 49)**: We move from backend-to-backend to backend-to-frontend. **Real-Time Web** (WebSockets).
