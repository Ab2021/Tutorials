# Day 27: NoSQL Patterns & Trade-offs

## 1. SQL is not the only tool

We love SQL. But sometimes it's the wrong tool.
*   **SQL**: Structured, Relational, ACID.
*   **NoSQL**: Flexible, Scalable, Specific.

---

## 2. Redis Patterns (The Swiss Army Knife)

Redis is not just a cache. It's a data structure server.

### 2.1 Caching (The Basics)
*   **Pattern**: Look-aside.
*   `GET /user/1` -> Check Redis. Miss? -> Check DB -> Write to Redis.
*   **TTL**: Always set an expiration (`SETEX`).

### 2.2 Rate Limiting (Token Bucket)
*   **Problem**: Stop abuse (100 req/min).
*   **Solution**: `INCR user_ip`. If > 100, block.
*   **Advanced**: Use `Lua Scripts` to make it atomic.

### 2.3 Leaderboards (Sorted Sets)
*   **Problem**: "Top 10 Players". In SQL, `ORDER BY score DESC LIMIT 10` is slow for 1M players.
*   **Solution**: Redis Sorted Set (`ZSET`).
    *   `ZADD leaderboard 1000 "Alice"`
    *   `ZREVRANGE leaderboard 0 9 WITH SCORES` (O(log N)). Instant.

---

## 3. MongoDB Patterns (Document Store)

### 3.1 Polymorphic Data
*   **Scenario**: E-Commerce Products.
    *   *Shirt*: has `size`, `color`.
    *   *Laptop*: has `ram`, `cpu`.
*   **SQL**: Complex (EAV Pattern or JSONB).
*   **Mongo**: Natural.
    ```json
    { "type": "shirt", "size": "M" }
    { "type": "laptop", "cpu": "M1" }
    ```

### 3.2 The Attribute Pattern
*   **Scenario**: Search by arbitrary attributes.
*   **Schema**:
    ```json
    {
      "specs": [
        { "k": "ram", "v": "16GB" },
        { "k": "color", "v": "red" }
      ]
    }
    ```
*   **Index**: Create index on `specs.k` and `specs.v`. Fast search for any attribute.

---

## 4. The Trade-off: CAP Theorem

*   **Consistency**: Everyone sees the same data.
*   **Availability**: The system is always up.
*   **Partition Tolerance**: The system works if the network cuts.
*   **Rule**: You can only pick 2.
    *   **SQL (CP)**: Prioritizes Consistency. If master dies, downtime until failover.
    *   **Cassandra/Dynamo (AP)**: Prioritizes Availability. Always accepts writes, but data might be eventually consistent.

---

## 5. Summary

Today we picked the right tool.
*   **Redis**: For speed, counters, leaderboards.
*   **Mongo**: For flexible schemas.
*   **SQL**: For financial transactions and complex joins.

**Tomorrow (Day 28)**: We explore the new kid on the block: **Vector Databases**. We will build the brain of an AI app.
