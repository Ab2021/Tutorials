# Day 7 Interview Prep: Replication & Sharding

## Q1: What is the difference between Replication and Sharding?
**Answer:**
*   **Replication:** Copying the *same* data to multiple nodes. (For Availability/Read Scaling).
*   **Sharding:** Splitting *different* data across multiple nodes. (For Write Scaling/Storage Capacity).

## Q2: Explain Consistent Hashing.
**Answer:**
*   A technique to distribute keys across nodes such that adding/removing a node only affects $K/N$ keys.
*   Uses a Ring topology.
*   Keys map to the nearest node in clockwise direction.
*   Uses Virtual Nodes to ensure even distribution.

## Q3: What are the problems with Sharding?
**Answer:**
*   **Resharding:** Moving data when a shard fills up is complex.
*   **Celebrity Problem (Hotspot):** If Justin Bieber is on Shard 1, Shard 1 will melt.
*   **Join Complexity:** Joining tables across shards is expensive/impossible.

## Q4: Master-Slave vs Multi-Master?
**Answer:**
*   **Master-Slave:** Simple. One writer. Good for Read-Heavy. SPOF (if Master dies, need promotion).
*   **Multi-Master:** Complex. Multiple writers. Good for Write-Heavy. Conflict resolution needed.
