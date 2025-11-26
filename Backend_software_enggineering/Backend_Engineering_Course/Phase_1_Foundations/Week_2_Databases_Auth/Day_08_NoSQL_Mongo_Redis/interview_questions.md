# Day 8: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between Redis and Memcached?
**Answer:**
*   **Data Structures**: Memcached is strictly Key-Value (Strings). Redis supports Lists, Sets, Hashes, etc.
*   **Persistence**: Redis can save to disk (RDB/AOF). Memcached is purely volatile (data lost on restart).
*   **Replication**: Redis supports Master-Slave replication.
*   *Verdict*: Redis is generally superior for modern use cases, but Memcached is still used for simple multi-threaded caching.

### Q2: Explain "Sharding" in MongoDB.
**Answer:**
*   **Definition**: Distributing data across multiple machines.
*   **Mechanism**: You choose a **Shard Key** (e.g., `user_id`). Mongo divides the data into "Chunks" based on ranges of the shard key and distributes them across shards.
*   **Router (mongos)**: The app connects to a router, which directs the query to the correct shard.
*   **Benefit**: Infinite horizontal scaling of storage and write throughput.

### Q3: What is a "Cache Stampede" (or Thundering Herd) and how do you prevent it?
**Answer:**
*   **Problem**: A popular cache key expires. 1000 requests hit the cache simultaneously, get a MISS, and all 1000 hit the Database at once, crashing it.
*   **Solutions**:
    1.  **Locking**: The first process to get a miss sets a lock. Others wait.
    2.  **Probabilistic Early Expiration**: If TTL is 60s, start refreshing it at 50s with a random probability.
    3.  **Soft TTL**: Return the stale value while fetching the new one in the background.

---

## Scenario-Based Questions

### Q4: You are building a "Recently Viewed Items" feature for an e-commerce site. It needs to show the last 10 items a user visited. Which DB and data structure do you use?
**Answer:**
**Redis List.**
*   **Command**: `LPUSH history:user_1 "item_id"`
*   **Trim**: `LTRIM history:user_1 0 9` (Keep only top 10).
*   **Why**: Extremely fast, built-in support for capping the list size. Doing this in SQL would require `INSERT` + `DELETE old rows`, which is heavy.

### Q5: You have a MongoDB collection `users` with 100M documents. You need to rename a field `fname` to `first_name`. How do you do it?
**Answer:**
*   **Challenge**: Updating 100M docs is slow and locks the DB.
*   **Strategy**:
    1.  **Lazy Migration**: Update application code to read `first_name` OR `fname`. When saving a user, write to `first_name` and remove `fname`.
    2.  **Background Script**: Run a slow script that iterates through the collection and updates batches of users during off-peak hours.
    3.  **Update Many**: `db.users.updateMany({}, { $rename: { "fname": "first_name" } })` - *Warning*: This might cause performance issues on a live production DB.

---

## Behavioral / Role-Specific Questions

### Q6: A junior dev suggests using Redis as the *primary* database for user profiles because "it's fast". Do you agree?
**Answer:**
**Caution required.**
*   **Pros**: Fast.
*   **Cons**:
    *   **RAM Cost**: RAM is expensive. Storing TBs of data in Redis is costly.
    *   **Durability**: Even with AOF, Redis is less durable than Postgres.
    *   **Querying**: You can't do complex queries (`WHERE age > 18 AND city = 'NY'`) efficiently without RediSearch.
*   **Decision**: No. Use Postgres/Mongo as the source of truth. Use Redis as a cache. If we *really* need speed, we can use Redis as a "Write-Behind" cache, but data safety is risky.
