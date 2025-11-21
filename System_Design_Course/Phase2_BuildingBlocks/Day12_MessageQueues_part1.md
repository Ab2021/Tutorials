# Day 12 Deep Dive: Kafka Internals

## 1. The Log Abstraction
*   Kafka is not a queue; it's a **Distributed Commit Log**.
*   **Append Only:** Writes are fast (Sequential I/O).
*   **Immutable:** You can't change history.

## 2. Architecture
*   **Topic:** Logical stream (e.g., "clicks").
*   **Partition:** Physical split of a topic. (Topic A -> P0, P1, P2).
    *   **Scaling:** Partitions can be on different servers.
    *   **Ordering:** Guaranteed *only* within a partition.
*   **Offset:** Unique ID of a message in a partition.
*   **Consumer Group:** A group of consumers working together.
    *   Each partition is consumed by *only one* consumer in the group.

## 3. Case Study: LinkedIn Databus (Why Kafka was built)
*   **Problem:** LinkedIn had spaghetti integration. User Service talked to Search, Graph, Hadoop, etc. $N \times M$ connections.
*   **Solution:** Unified Log.
    *   All databases push changes to Kafka.
    *   Search/Graph/Hadoop consume from Kafka.
*   **Result:** Decoupled architecture. "Central Nervous System".

## 4. Exactly-Once Semantics
*   **At Most Once:** Fire and forget. (Data loss possible).
*   **At Least Once:** Retry until Ack. (Duplicates possible).
*   **Exactly Once:** Hard. Kafka supports this using Idempotent Producers and Transactional Writes (Atomic write to multiple partitions).
