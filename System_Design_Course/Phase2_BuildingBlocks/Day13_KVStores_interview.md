# Day 13 Interview Prep: Key-Value Stores

## Q1: Redis vs Memcached?
**Answer:**
*   **Memcached:** Simple, multithreaded, string-only. Good for simple LRU cache.
*   **Redis:** Complex, single-threaded, rich data structures (Lists, Sets). Persistence. Clustering. Lua scripting. The standard choice today.

## Q2: Why is Redis single-threaded?
**Answer:**
*   It's CPU bound? No, it's usually Network/Memory bound.
*   Single thread avoids Context Switching and Lock contention.
*   It can handle 100k+ QPS on a single core.
*   For scaling, run multiple Redis instances (Sharding).

## Q3: Explain Consistent Hashing in DynamoDB.
**Answer:**
*   Used to distribute data across nodes.
*   Output of hash function is a ring.
*   Nodes are placed on the ring.
*   Data maps to the first node clockwise.
*   **Virtual Nodes:** Each physical node appears multiple times on the ring to ensure even distribution.

## Q4: How does Redis persistence work?
**Answer:**
*   **RDB:** Fork a child process. Child dumps memory to disk. Parent continues serving. Fast but potential data loss.
*   **AOF:** Log every write command to a file. Slower but durable.
