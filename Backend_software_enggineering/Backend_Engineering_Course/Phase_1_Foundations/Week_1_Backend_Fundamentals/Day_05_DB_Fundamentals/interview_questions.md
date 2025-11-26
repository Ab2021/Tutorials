# Day 5: Interview Questions & Answers

## Conceptual Questions

### Q1: Explain the 'I' in ACID (Isolation). Why is it important?
**Answer:**
*   **Isolation** ensures that concurrent transactions execute as if they were running sequentially.
*   **Importance**: Without isolation, you get race conditions.
    *   *Dirty Read*: Reading uncommitted data from another transaction that might roll back.
    *   *Lost Update*: Two users edit the same row, and the last one wins, overwriting the first.
*   **Levels**: Read Uncommitted (Lowest) -> Read Committed -> Repeatable Read -> Serializable (Highest/Slowest).

### Q2: When would you choose a NoSQL database over a Relational database?
**Answer:**
I would choose NoSQL (e.g., MongoDB/DynamoDB) when:
1.  **Schema Flexibility**: The data structure varies wildly (e.g., a product catalog where "Shirt" has size/color but "Laptop" has CPU/RAM).
2.  **Write Throughput**: I need to ingest massive amounts of data (IoT logs) and RDBMS locking overhead is too high.
3.  **Horizontal Scale**: I expect to grow to petabytes where sharding a relational DB is painful, but NoSQL handles it natively.
*   *Counter-point*: If the data is highly relational (Users -> Orders -> Payments), RDBMS is strictly better.

### Q3: What is a Vector Database and why do we need it for AI?
**Answer:**
*   **Definition**: A DB optimized to store and query high-dimensional vectors (arrays of floats).
*   **Why**: Traditional DBs search by *keyword matching* (`WHERE text LIKE '%apple%'`).
*   **AI Context**: LLMs convert text/images into "embeddings" (vectors) where similar concepts are close in mathematical space.
*   **Function**: Vector DBs use algorithms like HNSW (Hierarchical Navigable Small World) to perform "Nearest Neighbor Search" efficiently. "Find me the 10 vectors closest to this query vector."

---

## Scenario-Based Questions

### Q4: You are building a "Leaderboard" for a game with millions of players. It needs to update in real-time. Which DB do you use?
**Answer:**
**Redis (Sorted Sets).**
*   **Why**: Redis is in-memory (nanosecond latency).
*   **Data Structure**: The `ZSET` (Sorted Set) data structure is purpose-built for this.
    *   `ZADD leaderboard 1000 "player1"`
    *   `ZREVRANGE leaderboard 0 9` (Get top 10).
*   **RDBMS Approach**: `SELECT * FROM scores ORDER BY score DESC LIMIT 10` is `O(N log N)` or `O(log N)` with an index, but doing this on every write/read for millions of users is heavy on the disk. Redis does it in `O(log N)` in RAM.

### Q5: Your startup uses Postgres. The CTO wants to migrate to MongoDB because "it's web scale". How do you evaluate this?
**Answer:**
I would ask:
1.  **What is the pain point?** Are we hitting connection limits? Write limits? Or is it just "schema rigidity"?
2.  **Data Integrity**: Do we rely on transactions? (e.g., Billing). Mongo supports transactions now, but they are heavier than in Postgres.
3.  **Complexity**: We lose `JOIN`s. We have to do joins in application code, which is slower and buggy.
4.  **Alternative**: Can we use Postgres `JSONB` columns for the flexible parts?
*   *Conclusion*: Unless we have a specific scale problem that Postgres sharding/read-replicas can't solve, migrating is a premature optimization with high risk.

---

## Behavioral / Role-Specific Questions

### Q6: Describe a time you optimized a slow database query.
**Answer:**
*   **Situation**: An API endpoint `/analytics` was taking 5 seconds to load.
*   **Task**: Reduce latency to < 200ms.
*   **Action**:
    1.  **Explain**: I ran `EXPLAIN ANALYZE` on the SQL query.
    2.  **Finding**: It was doing a `SEQ SCAN` (Sequential Scan) on a table with 10M rows because it was filtering on a non-indexed column `status`.
    3.  **Fix**: I added a B-Tree index on the `status` column.
    4.  **Refinement**: I also noticed we were selecting `*` but only using 2 columns, so I changed it to `SELECT col1, col2` (Covering Index).
*   **Result**: Query time dropped to 50ms.
