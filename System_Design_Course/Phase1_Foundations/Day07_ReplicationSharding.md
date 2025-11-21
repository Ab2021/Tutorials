# Day 7: Replication & Sharding

## 1. Replication
Copying data to multiple machines.
*   **Why?** High Availability (if one fails, others work), Latency (read from nearest), Scalability (read replicas).
*   **Models:**
    *   **Single Leader:** All writes go to Leader. Leader replicates to Followers. Reads from Followers. (MySQL, Postgres, MongoDB).
    *   **Multi-Leader:** Multiple nodes accept writes. (Datacenters).
    *   **Leaderless:** Any node accepts writes. (Cassandra, Dynamo).

## 2. Sharding (Partitioning)
Splitting data across multiple machines because it doesn't fit on one.
*   **Horizontal Partitioning:** Split by rows. (User 1-1M on DB1, 1M-2M on DB2).
*   **Sharding Key:** The column used to decide which shard. (e.g., `user_id`).

## 3. Sharding Strategies
*   **Range Based:** `user_id` 1-1000 $\to$ Node A.
    *   **Pros:** Range queries are easy.
    *   **Cons:** Hotspots (if all new users are active).
*   **Hash Based:** `hash(user_id) % N`.
    *   **Pros:** Uniform distribution.
    *   **Cons:** Resharding is painful (changing N moves all data).

## 4. Consistent Hashing
*   **Problem:** In Hash Based (`% N`), adding a node changes N, remapping almost ALL keys.
*   **Solution:** Map both Nodes and Keys to a Circle (0-360 degrees).
*   **Algorithm:**
    1.  Hash Node IP to a point on the ring.
    2.  Hash Key to a point.
    3.  Walk clockwise to find the first Node.
*   **Benefit:** Adding a node only affects its immediate neighbor. Only $1/N$ keys move.
