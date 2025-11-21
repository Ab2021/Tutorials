# Day 4 Interview Prep: Database Internals

## Q1: B-Tree vs LSM Tree?
**Answer:**
*   **B-Tree:** Read-heavy workloads. Updates in place. Fragmentation possible. (MySQL/Postgres).
*   **LSM Tree:** Write-heavy workloads. Append-only. Compaction needed. (Cassandra/RocksDB).

## Q2: What is a Bloom Filter?
**Answer:**
*   A space-efficient probabilistic data structure to test if an element is in a set.
*   False Positives: Possible.
*   False Negatives: Impossible.
*   Used in Databases to avoid disk lookups for keys that don't exist.

## Q3: Explain ACID.
**Answer:**
*   **Atomicity:** Transaction is all or nothing.
*   **Consistency:** Database moves from one valid state to another.
*   **Isolation:** Concurrent transactions don't affect each other.
*   **Durability:** Committed data survives power loss (WAL).

## Q4: How does a database handle concurrent writes?
**Answer:**
*   **Pessimistic Locking:** Lock the row/table. Others wait. (Safe, Slow).
*   **Optimistic Locking:** Version number. Read version 1. Write if version is still 1. If 2, retry. (Fast, good for low contention).
*   **MVCC (Multi-Version Concurrency Control):** Readers don't block writers. Writers don't block readers. Each transaction sees a snapshot.
