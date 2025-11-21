# Day 20: Project - Distributed Rate Limiter

## 1. Goal
Build a scalable, distributed Rate Limiter service in Python/Go.
*   **Requirements:**
    *   Limit users to 10 requests/minute.
    *   Distributed (Multiple API servers).
    *   Low Latency (< 10ms).
    *   Accurate (Sliding Window).

## 2. Architecture
*   **API Gateway:** Intercepts requests.
*   **Redis Cluster:** Stores counters.
*   **Lua Script:** Ensures atomicity.

## 3. API Design
*   `POST /check_limit`
    *   **Input:** `{"user_id": "123", "limit": 10, "window": 60}`
    *   **Output:** `{"allowed": true, "remaining": 9}`

## 4. Strategy
1.  **Setup:** Run Redis in Docker.
2.  **Code:** Implement Sliding Window Counter in Lua.
3.  **Test:** Run a load test script (100 threads) to verify correctness.
