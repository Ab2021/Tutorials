# Day 20 Deep Dive: Implementation (Python + Redis)

## 1. The Lua Script
Save this as `limiter.lua`.
```lua
-- Keys: [key]
-- Args: [limit, window_size, current_time]
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- Remove old entries (ZREMRANGEBYSCORE)
local clear_before = now - window
redis.call('ZREMRANGEBYSCORE', key, 0, clear_before)

-- Count current entries
local count = redis.call('ZCARD', key)

if count < limit then
    -- Add new entry (ZADD)
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window) -- Auto cleanup
    return 1 -- Allowed
else
    return 0 -- Denied
end
```

## 2. The Python Client
```python
import redis
import time

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Load script
with open('limiter.lua', 'r') as f:
    lua_script = f.read()
limiter = r.register_script(lua_script)

def is_allowed(user_id, limit, window):
    now = time.time()
    key = f"rate_limit:{user_id}"
    result = limiter(keys=[key], args=[limit, window, now])
    return result == 1

# Test
user = "user_123"
for i in range(12):
    allowed = is_allowed(user, 10, 60)
    print(f"Request {i+1}: {'Allowed' if allowed else 'Denied'}")
```

## 3. Explanation
*   **Sorted Set (ZSET):** Stores timestamps.
*   **ZREMRANGEBYSCORE:** Removes timestamps older than the window (Sliding).
*   **ZCARD:** Counts requests in current window.
*   **Atomicity:** Lua ensures no race conditions between Read and Write.
