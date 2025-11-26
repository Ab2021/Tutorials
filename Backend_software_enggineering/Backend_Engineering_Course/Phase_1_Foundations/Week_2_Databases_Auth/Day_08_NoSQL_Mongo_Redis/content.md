# Day 8: NoSQL - Document & Key-Value Stores

## 1. The NoSQL Revolution

"NoSQL" (Not Only SQL) emerged to solve problems RDBMS couldn't:
1.  **Flexible Schema**: Rapid iteration without `ALTER TABLE`.
2.  **Horizontal Scale**: Sharding out of the box.
3.  **High Performance**: Specialized data structures (e.g., Redis).

---

## 2. Document Stores: MongoDB

MongoDB stores data in **BSON** (Binary JSON).

### 2.1 Core Concepts
*   **Database**: Container for collections.
*   **Collection**: Analogous to a Table.
*   **Document**: Analogous to a Row. A BSON object.

### 2.2 Modeling: Embed vs. Reference
The biggest mistake SQL devs make in Mongo is treating it like SQL.

*   **Scenario**: A User has multiple Addresses.
*   **SQL Approach**: Two tables (`users`, `addresses`) joined by FK.
*   **Mongo Approach 1 (Embedding)**:
    ```json
    {
      "_id": 1,
      "name": "Alice",
      "addresses": [
        { "street": "123 Main St", "city": "NY" },
        { "street": "456 Side St", "city": "LA" }
      ]
    }
    ```
    *   *Pros*: One read fetches everything. Fast. Atomic updates.
    *   *Cons*: Document size limit (16MB). Duplication if addresses are shared.
*   **Mongo Approach 2 (Referencing)**:
    ```json
    // User
    { "_id": 1, "name": "Alice", "address_ids": [101, 102] }
    // Address
    { "_id": 101, "street": "123 Main St" }
    ```
    *   *Pros*: Normalized.
    *   *Cons*: Requires application-level join (2 queries).

### 2.3 When to use Mongo?
*   Content Management Systems (CMS).
*   Catalogs with varied attributes.
*   IoT data (flexible payload).
*   *Avoid for*: Complex transactions (Banking), highly relational data.

---

## 3. Key-Value Stores: Redis

Redis is an **In-Memory** data structure store. It is blazingly fast (sub-millisecond).

### 3.1 Data Structures
Redis is not just Key-Value strings. It's a "Data Structure Server".

1.  **Strings**: Basic value.
    *   `SET user:1 "Alice"`
    *   *Use Case*: Caching HTML fragments, Session tokens.
2.  **Hashes**: Map within a key.
    *   `HSET user:1 name "Alice" age 30`
    *   *Use Case*: Storing user profiles object.
3.  **Lists**: Linked List.
    *   `LPUSH queue "job1"` -> `RPOP queue`
    *   *Use Case*: Message Queues, Recent Activity feeds.
4.  **Sets**: Unique collection.
    *   `SADD active_users "user1"`
    *   *Use Case*: Unique visitors, Tag clouds.
5.  **Sorted Sets (ZSets)**: Sets with a score.
    *   `ZADD leaderboard 100 "user1"`
    *   *Use Case*: Leaderboards, Priority Queues.

### 3.2 Persistence
Redis is in-memory, but can save to disk:
*   **RDB (Snapshots)**: Saves every X minutes. Fast restart, potential data loss.
*   **AOF (Append Only File)**: Logs every write. Slower, better durability.

---

## 4. Caching Patterns

How to use Redis with your main DB (Postgres).

### 4.1 Cache-Aside (Lazy Loading)
1.  App asks Redis for `user:1`.
2.  **Miss**: App asks Postgres. App saves result to Redis. Returns to user.
3.  **Hit**: App returns Redis data immediately.

### 4.2 Write-Through
1.  App writes to Redis AND Postgres at the same time.
2.  *Pros*: Cache is always fresh.
3.  *Cons*: Slow writes (double write).

### 4.3 TTL (Time To Live)
Always set an expiration!
`SET user:1 "Alice" EX 60` (Expires in 60s).
Prevents the cache from filling up with stale data.

---

## 5. Summary

Today we explored the non-relational world.
*   **MongoDB**: Great for flexible documents. Choose Embedding for speed, Referencing for normalization.
*   **Redis**: The Swiss Army Knife of backends. Use it for Caching, Queues, and Leaderboards.

**Tomorrow (Day 9)**: We enter the AI era. We will learn about **Vector Databases**, Embeddings, and how to search by "meaning" rather than "keyword".
