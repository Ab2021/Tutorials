# Day 13: Key-Value Stores

## 1. What is a KV Store?
A NoSQL database that stores data as a collection of key-value pairs.
*   **Key:** Unique identifier.
*   **Value:** Blob, String, JSON, etc.
*   **Operations:** `PUT(key, val)`, `GET(key)`, `DELETE(key)`.
*   **Performance:** $O(1)$ access.

## 2. Redis (Remote Dictionary Server)
*   **Architecture:** In-memory. Single-threaded event loop.
*   **Persistence:**
    *   **RDB (Snapshot):** Dumps memory to disk every X minutes. (Fast restart, data loss possible).
    *   **AOF (Append Only File):** Logs every write. (Slower restart, durable).
*   **Data Structures:** String, List, Set, Hash, Sorted Set, HyperLogLog, Geo.
*   **Use Cases:** Caching, Session Store, Leaderboards, Pub/Sub.

## 3. DynamoDB (Amazon)
*   **Architecture:** Distributed, Leaderless (Dynamo paper).
*   **Consistency:** Tunable (Eventual vs Strong).
*   **Scaling:** Auto-sharding (Partition Key).
*   **Use Cases:** Shopping Cart, Metadata, High scale apps.

## 4. Etcd / Zookeeper
*   **Focus:** Strong Consistency (CP).
*   **Use Cases:** Configuration management, Service Discovery, Leader Election.
*   **Not for:** High throughput data storage.
