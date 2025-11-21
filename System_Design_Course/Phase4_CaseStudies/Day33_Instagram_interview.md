# Day 33 Interview Prep: Design Instagram

## Q1: Push vs Pull Feed?
**Answer:**
*   **Pull:** Good for small scale or celebrities. High Read Latency.
*   **Push:** Good for read-heavy scale. High Write Latency (Fan-out).
*   **Hybrid:** Best of both.

## Q2: How to handle "Celebrity Problem" in Push Model?
**Answer:**
*   Don't push celebrity posts to 100M followers.
*   Mark user as "VIP".
*   When follower reads feed, pull VIP posts dynamically and merge.

## Q3: How to store "Likes"?
**Answer:**
*   **Volume:** Huge.
*   **Async:** When user clicks Like, send to Kafka.
*   **Aggregation:** Flink/Worker aggregates likes per post.
*   **Storage:** Redis Counter (Fast) + Cassandra (Persistent).
*   **Consistency:** It's okay if the count is off by a few for a few seconds.

## Q4: How to implement "Stories" (24h expiry)?
**Answer:**
*   **TTL:** Use Redis/Cassandra TTL feature.
*   **Cleanup:** Data automatically disappears.
*   **Optimization:** Pre-fetch stories of top 5 friends when app opens.
