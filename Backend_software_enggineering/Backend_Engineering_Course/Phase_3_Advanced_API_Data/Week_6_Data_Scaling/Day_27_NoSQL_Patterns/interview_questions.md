# Day 27: Interview Questions & Answers

## Conceptual Questions

### Q1: How does Redis persist data if it's in-memory?
**Answer:**
1.  **RDB (Snapshot)**: Dumps the whole RAM to disk every X minutes. (Fast restart, potential data loss).
2.  **AOF (Append Only File)**: Logs every write command to a file. (Slower restart, minimal data loss).
3.  **Hybrid**: Use both.

### Q2: What is a "Cache Stampede" (Thundering Herd) and how do you prevent it?
**Answer:**
*   **Problem**: A popular cache key expires. 1000 requests hit the server simultaneously. All 1000 see a "Miss" and try to query the DB at once. DB crashes.
*   **Solutions**:
    1.  **Locking**: The first request locks the key. Others wait.
    2.  **Probabilistic Early Expiration**: If TTL is 60s, start refreshing it at 50s with a 10% chance.

### Q3: When should you use MongoDB over Postgres?
**Answer:**
*   **Use Mongo**:
    *   Unstructured/Polymorphic data (Product Catalog).
    *   High Write Throughput (Sharding is built-in).
    *   Rapid Prototyping (Schema-less).
*   **Use Postgres**:
    *   Relational Data (Users, Orders, Payments).
    *   Strict ACID requirements.
    *   Complex Analytics (Joins).
    *   *Note*: Postgres `JSONB` can do 90% of what Mongo does.

---

## Scenario-Based Questions

### Q4: You need to implement a "Recent Views" feature (Last 50 items viewed by user). Which DB?
**Answer:**
*   **Redis List (or Sorted Set)**.
*   **Command**: `LPUSH views:user:1 item_id` + `LTRIM views:user:1 0 49`.
*   **Why**: Extremely fast. Auto-cleanup (LTRIM). No need to store this transient data in the main SQL DB.

### Q5: Your Redis instance is full. What happens?
**Answer:**
*   Depends on `maxmemory-policy`.
    1.  **noeviction**: Returns error on write. (Bad).
    2.  **allkeys-lru**: Deletes the Least Recently Used key (any key).
    3.  **volatile-lru**: Deletes LRU key *that has an expiry set*. (Safest for caches).

---

## Behavioral / Role-Specific Questions

### Q6: A junior dev wants to use Redis as the *primary* database for User Profiles because "it's fast". Do you agree?
**Answer:**
*   **Caution**.
*   **Risk**: RAM is expensive. If you have 100M users, it might not fit in RAM.
*   **Durability**: Redis is less durable than disk-based DBs.
*   **Querying**: You can't do `SELECT * FROM users WHERE age > 20` efficiently in Redis (without extra indexes).
*   **Verdict**: Use SQL/Mongo as primary. Use Redis as a *Cache* for profiles.
