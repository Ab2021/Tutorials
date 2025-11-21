# Day 20 Interview Prep: Project Defense

## Q1: Why did you use Redis Sorted Sets?
**Answer:**
*   Sorted Sets allow us to implement a **Sliding Window Log** efficiently.
*   We can remove old entries (`ZREMRANGEBYSCORE`) and count current ones (`ZCARD`) in $O(\log N)$ time.
*   This is more accurate than a Fixed Window counter (which has edge spikes).

## Q2: What is the memory complexity?
**Answer:**
*   We store a timestamp (8 bytes) for every request in the window.
*   If a user makes 1000 req/min, we store 1000 entries.
*   **Optimization:** If traffic is huge, switch to **Sliding Window Counter** (Approximation) which only stores 2 numbers (Current Count, Previous Count).

## Q3: How to scale this?
**Answer:**
*   **Redis Cluster:** Shard keys by `user_id`. User A goes to Node 1, User B to Node 2.
*   **Read Replicas:** Not needed for Rate Limiting (it's write-heavy).

## Q4: What if Redis fails?
**Answer:**
*   **Fail-Open:** Allow the request. It's better to let a few spammers in than to block legitimate users (and lose revenue).
