# Day 29: Interview Questions & Answers

## Conceptual Questions

### Q1: What is "Replication Lag" and how do you handle it in the UI?
**Answer:**
*   **Definition**: The delay between writing to Primary and the data appearing in the Replica.
*   **UI Handling**:
    1.  **Sticky Session**: Route the user to the Primary for a few seconds after a write.
    2.  **Optimistic UI**: Update the UI immediately via JS, assuming the write succeeded.
    3.  **Read-Your-Writes**: The backend ensures that if a user just wrote, their subsequent reads go to Primary (or a synced replica).

### Q2: Explain "Consistent Hashing".
**Answer:**
*   **Problem**: In normal Modulo Hashing (`id % N`), if you add a server (N -> N+1), *almost all* keys are remapped. Massive cache miss / data movement.
*   **Consistent Hashing**: Maps keys and servers to a "Ring".
*   **Benefit**: Adding/Removing a node only affects the keys on its immediate neighbor. Only K/N keys need to move. Used in Cassandra, DynamoDB, Discord.

### Q3: What is a "Hot Shard" (or Hot Partition)?
**Answer:**
*   **Scenario**: You shard by `user_id`. Justin Bieber is on Shard 1. He generates 1000x more traffic/data than others.
*   **Result**: Shard 1 is overloaded while Shard 2-10 are idle.
*   **Fix**:
    *   Add a "salt" to the shard key for hot items.
    *   Isolate hot data to a dedicated node.

---

## Scenario-Based Questions

### Q4: You need to migrate a 10TB Monolithic Postgres database to a Sharded architecture with zero downtime. How?
**Answer:**
1.  **Dual Write**: Update app to write to *both* Old DB and New Sharded DB (or use CDC).
2.  **Backfill**: Run a script to copy historical data to New DB.
3.  **Compare**: Verify data consistency.
4.  **Switch Reads**: Point reads to New DB.
5.  **Switch Writes**: Stop writing to Old DB.

### Q5: When would you choose "Multi-Master" replication?
**Answer:**
*   **Scenario**: You have users in US and EU. You want US users to write to US DB (fast) and EU users to write to EU DB (fast).
*   **Challenge**: **Conflict Resolution**. What if US user sets `x=1` and EU user sets `x=2` at the same time?
*   **Use Case**: Collaborative editing (Google Docs), Global apps with high availability needs. (Complex to manage).

---

## Behavioral / Role-Specific Questions

### Q6: A startup founder says "Let's start with Sharding so we are web-scale from Day 1". Good idea?
**Answer:**
*   **No**.
*   **Premature Optimization**: Sharding adds massive operational complexity (backups, joins, transactions).
*   **Advice**: Start with a Monolith DB. Scale Vertically. Use Read Replicas. Only shard when you hit the limit of the biggest machine AWS sells (which is huge).
