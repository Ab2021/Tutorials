# Day 16: Rate Limiting

## 1. Why Rate Limit?
*   **Prevent Abuse:** DDoS, Brute force attacks.
*   **Cost Control:** Prevent users from consuming too many resources.
*   **Fairness:** Ensure one user doesn't hog the system.

## 2. Algorithms
### Token Bucket
*   **Concept:** Bucket holds $N$ tokens. Refill $R$ tokens/sec.
*   **Request:** Needs 1 token. If bucket empty, drop.
*   **Pros:** Allows bursts (up to $N$). Memory efficient.

### Leaky Bucket
*   **Concept:** Queue with constant output rate.
*   **Request:** Add to queue. If full, drop.
*   **Pros:** Smooths traffic (constant rate).
*   **Cons:** No bursts.

### Fixed Window Counter
*   **Concept:** Counter per minute. Reset at :00.
*   **Cons:** Spike at edges (e.g., 59s and 01s allows 2x limit).

### Sliding Window Log
*   **Concept:** Store timestamp of every request. Count logs in last minute.
*   **Pros:** Accurate.
*   **Cons:** High memory (storing logs).

### Sliding Window Counter
*   **Concept:** Hybrid. Weighted average of Previous Window + Current Window.
*   **Pros:** Accurate and Memory efficient.

## 3. Implementation
*   **Local:** In-memory (Guava Ratelimiter). Fast but not distributed.
*   **Distributed:** Redis (Lua Script). Shared state.
