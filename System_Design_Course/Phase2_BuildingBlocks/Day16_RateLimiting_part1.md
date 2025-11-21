# Day 16 Deep Dive: Distributed Rate Limiting

## 1. The Challenge
*   **Race Conditions:**
    *   User A sends 2 requests.
    *   Server 1 reads counter (5).
    *   Server 2 reads counter (5).
    *   Both increment to 6.
    *   **Result:** User cheated.
*   **Latency:** Checking Redis for every request adds latency.

## 2. Solution: Redis Lua Script
*   **Atomicity:** Lua script runs atomically in Redis. No race conditions.
*   **Logic:**
    ```lua
    local current = redis.call('INCR', key)
    if current == 1 then
        redis.call('EXPIRE', key, 60)
    end
    if current > limit then
        return 0 -- Deny
    end
    return 1 -- Allow
    ```

## 3. Case Study: Stripe API Gateway
*   **Requirement:** High reliability. If Redis is down, don't block payments.
*   **Strategy:**
    *   **Hard Limit:** Redis-based (Global).
    *   **Soft Limit:** Local memory (Per node).
    *   **Fail-Open:** If Redis fails, allow the request (Business priority: Take the money).
*   **Throttling:** Stripe throttles based on:
    *   Request Type (Read vs Write).
    *   User Tier (Free vs Pro).

## 4. Code: Token Bucket (Python)
```python
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.rate = refill_rate
        self.last_refill = time.time()

    def allow(self):
        now = time.time()
        # Refill
        delta = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```
