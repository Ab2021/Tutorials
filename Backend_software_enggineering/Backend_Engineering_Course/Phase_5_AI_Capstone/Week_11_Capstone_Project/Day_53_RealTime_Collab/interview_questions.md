# Day 53: Interview Questions & Answers

## Conceptual Questions

### Q1: How do you implement "Presence" (User A is viewing this doc)?
**Answer:**
*   **Redis Sets**:
    *   On Connect: `SADD doc:123:users user_A`
    *   On Disconnect: `SREM doc:123:users user_A`
*   **Heartbeat**: Clients send "ping" every 30s. If missed, remove from set (Set TTL on the key).

### Q2: What happens if Redis goes down?
**Answer:**
*   **Impact**: Real-time sync stops working across servers. Users on the *same* server might still sync (if using local memory fallback), but users on different servers won't.
*   **Mitigation**: Redis Sentinel/Cluster for High Availability.

### Q3: Why not use Kafka for the real-time sync?
**Answer:**
*   **Latency**: Kafka is fast, but Redis Pub/Sub is faster (in-memory, fire-and-forget).
*   **Persistence**: We don't need to store every keystroke forever. We just need to broadcast it now. Kafka's durability is overkill here.

---

## Scenario-Based Questions

### Q4: 1 Million users connect at the same time (e.g., Super Bowl event). How do you scale?
**Answer:**
*   **Load Balancer**: Round-robin connections to multiple WebSocket servers.
*   **Ephemeral Ports**: Ensure you don't run out of ports (increase `fs.file-max` and `ulimit`).
*   **Sharding**: If one Redis instance can't handle the Pub/Sub volume, shard channels across multiple Redis instances (e.g., `doc:1` -> Redis A, `doc:2` -> Redis B).

### Q5: A user loses internet connection. When they reconnect, they are missing edits. How to fix?
**Answer:**
*   **Version History**: Each message has a sequence number.
*   **Catch-up**: Client sends "Last received seq: 50". Server sends 51-60 from Redis buffer.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use polling instead of WebSockets because "WebSockets are hard". Response?
**Answer:**
*   **Context**: For a document editor, polling (even every 1s) feels laggy.
*   **UX**: Users expect instant feedback.
*   **Server Load**: Polling kills the CPU with empty requests.
*   **Verdict**: WebSockets (or SSE) are non-negotiable for this use case.
