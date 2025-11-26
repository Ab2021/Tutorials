# Lab: Day 27 - Redis Rate Limiter

## Goal
Implement a "Fixed Window" Rate Limiter using Redis.

## Prerequisites
- Docker (Redis).
- `pip install redis`

## Step 1: Start Redis
```bash
docker run -d -p 6379:6379 --name my-redis redis
```

## Step 2: The Code (`rate_limiter.py`)

```python
import redis
import time

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def is_allowed(user_id: str, limit: int, window: int) -> bool:
    """
    Returns True if user is allowed, False if blocked.
    limit: Max requests
    window: Time window in seconds
    """
    # Key: rate:user:123
    key = f"rate:{user_id}"
    
    # 1. Increment counter
    current_count = r.incr(key)
    
    # 2. If first request, set expiry
    if current_count == 1:
        r.expire(key, window)
        
    # 3. Check limit
    if current_count > limit:
        return False
    return True

# Simulation
user = "user_123"
print(f"Testing Rate Limit for {user} (5 req / 10 sec)...")

for i in range(1, 10):
    allowed = is_allowed(user, limit=5, window=10)
    status = "✅ Allowed" if allowed else "❌ Blocked"
    print(f"Request {i}: {status}")
    time.sleep(1)
```

## Step 3: Run It
`python rate_limiter.py`

*   **Expected Output**:
    *   Requests 1-5: Allowed.
    *   Requests 6-9: Blocked.

## Challenge: Sliding Window
The "Fixed Window" has a flaw: If I make 5 requests at 09:59 and 5 requests at 10:01, I made 10 requests in 2 seconds (burst), but the limiter allows it.
*   **Task**: Implement a **Sliding Window Log** using Redis Sorted Sets (`ZADD`, `ZREMRANGEBYSCORE`, `ZCARD`).
*   Store timestamps as scores. Count how many timestamps are in the last window.
