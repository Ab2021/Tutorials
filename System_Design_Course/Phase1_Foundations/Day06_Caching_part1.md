# Day 6 Deep Dive: Advanced Caching

## 1. Thundering Herd Problem (Cache Stampede)
*   **Scenario:** A popular key expires.
*   **Event:** 10,000 users request the key simultaneously.
*   **Result:** All 10,000 get a "Cache Miss" and hit the DB. DB crashes.
*   **Solutions:**
    *   **Locking:** Only one process allowed to recompute the key. Others wait.
    *   **Probabilistic Early Expiration:** If TTL is 60s, start recomputing at 50s with some probability.

## 2. Cache Penetration
*   **Scenario:** Users request a key that *does not exist* in DB (e.g., `user_id=-1`).
*   **Result:** Always Cache Miss. Always DB Hit.
*   **Solution:**
    *   **Bloom Filter:** Check if key exists before hitting DB.
    *   **Cache Null:** Cache the "Not Found" result for a short time (TTL 5s).

## 3. Cache Avalanche
*   **Scenario:** Many keys expire at the exact same time.
*   **Result:** Huge spike in DB load.
*   **Solution:** Add "Jitter" (Randomness) to TTL. Instead of 60s, use 60s + random(0-10s).

## 4. Redis vs Memcached
| Feature | Redis | Memcached |
| :--- | :--- | :--- |
| **Data Types** | Strings, Lists, Sets, Hashes, Sorted Sets | Strings only |
| **Persistence** | Yes (RDB, AOF) | No (In-memory only) |
| **Replication** | Master-Slave, Cluster | No |
| **Threading** | Single-threaded | Multi-threaded |
| **Use Case** | Complex cache, Message Broker, Leaderboard | Simple KV cache |
