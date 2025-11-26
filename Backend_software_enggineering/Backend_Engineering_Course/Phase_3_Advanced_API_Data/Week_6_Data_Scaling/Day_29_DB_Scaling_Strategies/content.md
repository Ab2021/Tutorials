# Day 29: Database Scaling Strategies

## 1. The Ceiling

Your startup is successful. Your DB CPU is at 90%. What now?

### 1.1 Vertical Scaling (Scale Up)
*   **Action**: Buy a bigger server (AWS `db.t3.micro` -> `db.r5.24xlarge`).
*   **Pros**: No code changes. Easy.
*   **Cons**: Expensive. Has a physical limit (128 cores, 4TB RAM). Single Point of Failure (SPOF).

### 1.2 Horizontal Scaling (Scale Out)
*   **Action**: Add more servers.
*   **Pros**: Infinite scale. Cheaper hardware.
*   **Cons**: Complex. Consistency issues.

---

## 2. Read Replicas (Master-Slave)

Most apps are **Read-Heavy** (90% reads, 10% writes).
*   **Architecture**:
    *   **Primary (Master)**: Handles Writes (and some reads). Replicates data to Slaves.
    *   **Replicas (Slaves)**: Handle Reads only.
*   **Replication Lag**: The time delay (ms) between a write on Master and it appearing on Replica.
    *   *Risk*: User updates profile, refreshes page, sees old profile (because read hit a lagging replica).

---

## 3. Sharding (Partitioning)

When **Writes** are the bottleneck (or data size > 10TB).
*   **Concept**: Split data across multiple nodes based on a **Shard Key**.
*   **Example**: Shard by `user_id`.
    *   Users 1-1M -> Node A.
    *   Users 1M-2M -> Node B.
*   **Challenges**:
    *   **Resharding**: What if Node A gets full? You must move data to Node C. (Complex).
    *   **Cross-Shard Joins**: `SELECT * FROM orders` (where orders are on different nodes) is incredibly slow/impossible.
    *   **Hot Keys**: If Justin Bieber (`user_id=1`) is on Node A, Node A will melt.

---

## 4. Consistency Models

*   **Strong Consistency**: Read always returns the latest Write. (SQL).
*   **Eventual Consistency**: Read might return stale data, but "eventually" catches up. (NoSQL, DNS).
*   **Tunable Consistency** (Cassandra):
    *   `Write Quorum`: Write to 2 out of 3 nodes.
    *   `Read Quorum`: Read from 2 out of 3 nodes.
    *   Result: Strong consistency without a single master.

---

## 5. Summary

Today we broke the database.
*   **Replicas**: Scale Reads. Watch out for Lag.
*   **Sharding**: Scale Writes. Avoid if possible (complexity is high).
*   **Consistency**: The price we pay for scale.

**Tomorrow (Day 30)**: We wrap up Phase 3 with **Data Pipelines & Backups**. How to move data around without losing it.
