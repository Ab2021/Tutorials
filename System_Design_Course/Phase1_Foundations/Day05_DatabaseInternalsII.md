# Day 5: Database Internals II

## 1. SQL vs NoSQL
### SQL (Relational)
*   **Structure:** Tables, Rows, Columns. Fixed Schema.
*   **Properties:** ACID. Strong Consistency.
*   **Scaling:** Vertical (easy), Horizontal (hard - sharding).
*   **Examples:** MySQL, PostgreSQL, Oracle.

### NoSQL (Non-Relational)
*   **Structure:** Key-Value, Document, Column, Graph. Flexible Schema.
*   **Properties:** BASE (Basically Available, Soft state, Eventual consistency).
*   **Scaling:** Horizontal (easy - built-in sharding).
*   **Examples:** MongoDB, Cassandra, Redis, Neo4j.

## 2. CAP Theorem
In a distributed system, you can only have 2 out of 3:
*   **Consistency (C):** Every read receives the most recent write or an error.
*   **Availability (A):** Every request receives a (non-error) response, without the guarantee that it contains the most recent write.
*   **Partition Tolerance (P):** The system continues to operate despite an arbitrary number of messages being dropped or delayed by the network.

**Reality:** Network Partitions (P) *will* happen. So you must choose between CP or AP.
*   **CP (Consistency + Partition Tolerance):** If partition happens, shut down non-consistent nodes. (MongoDB, HBase).
*   **AP (Availability + Partition Tolerance):** If partition happens, return stale data. (Cassandra, DynamoDB).

## 3. PACELC Theorem
An extension of CAP.
*   **If Partition (P):** Choose A or C.
*   **Else (E) (Normal operation):** Choose Latency (L) or Consistency (C).
*   **Example:** DynamoDB chooses Availability (A) during partitions, and Latency (L) during normal operation (Eventual Consistency).
