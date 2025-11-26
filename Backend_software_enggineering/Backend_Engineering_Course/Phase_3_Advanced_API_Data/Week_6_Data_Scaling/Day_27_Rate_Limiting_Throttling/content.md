# Day 27: Rate Limiting & Throttling - Protecting Your APIs

## Table of Contents
1. [Why Rate Limiting?](#1-why-rate-limiting)
2. [Rate Limiting Algorithms](#2-rate-limiting-algorithms)
3. [Distributed Rate Limiting](#3-distributed-rate-limiting)
4. [Implementation Examples](#4-implementation-examples)
5. [Adaptive Throttling](#5-adaptive-throttling)
6. [API Quotas](#6-api-quotas)
7. [DDoS Protection](#7-ddos-protection)
8. [Production Patterns](#8-production-patterns)
9. [HTTP Headers](#9-http-headers)
10. [Summary](#10-summary)

---

## 1. Why Rate Limiting?

### 1.1 The Problem

**Without rate limiting**:
```
Malicious user: 100,000 requests/second
→ Server overload
→ Legitimate users can't access
→ Downtime
```

### 1.2 Benefits

✅ **Prevent abuse**: Stop malicious users  
✅ **Fair usage**: Ensure equal access  
✅ **Cost control**: Limit expensive API calls  
✅ **Infrastructure protection**: Prevent overload  
✅ **Monetization**: Different limits per tier

---

## 2. Rate Limiting Algorithms

### 2.1 Token Bucket

**Concept**: Bucket holds tokens, each request consumes 1 token.

```python
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.tokens = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def allow_request(self):
        # Refill tokens
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        # Check if token available
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        else:
            return False

# Usage:
bucket = TokenBucket(capacity=10, refill_rate=1)  # 10 tokens, refill 1/sec

if bucket.allow_request():
    process_request()
else:
    return_rate_limit_error()
```

**Visual**:
```
Bucket: [T][T][T][T][T]  (5 tokens)
Request → consume token → [T][T][T][T][ ]  (4 tokens)
1 second later → refill → [T][T][T][T][T]  (5 tokens)
```

**Pros**: Allows bursts (if bucket full)  
**Cons**: Bursts can overload

### 2.2 Leaky Bucket

**Concept**: Requests added to queue, processed at fixed rate.

```python
from collections import deque
import time

class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.capacity = capacity  # Max queue size
        self.leak_rate = leak_rate  # Requests per second
        self.queue = deque()
        self.last_leak = time.time()
    
    def allow_request(self):
        # Leak (process) requests
        now = time.time()
        elapsed = now - self.last_leak
        to_leak = int(elapsed * self.leak_rate)
        
        for _ in range(min(to_leak, len(self.queue))):
            self.queue.popleft()
        
        self.last_leak = now
        
        # Add request to queue
        if len(self.queue) < self.capacity:
            self.queue.append(now)
            return True
        else:
            return False  # Queue full
```

**Pros**: Smooth, predictable rate  
**Cons**: Doesn't allow bursts

### 2.3 Fixed Window Counter

```python
import time

class FixedWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit  # Max requests per window
        self.window_size = window_size  # Seconds
        self.counters = {}  # {window_start: count}
    
    def allow_request(self, user_id):
        now = time.time()
        window_start = int(now // self.window_size) * self.window_size
        
        key = f"{user_id}:{window_start}"
        count = self.counters.get(key, 0)
        
        if count < self.limit:
            self.counters[key] = count + 1
            return True
        else:
            return False

# Usage: 100 requests per minute
limiter = FixedWindowCounter(limit=100, window_size=60)

if limiter.allow_request(user_id):
    process_request()
```

**Problem**: Burst at window boundary
```
Window 1 (0-60s):   99 requests at 59s
Window 2 (60-120s): 99 requests at 60s
→ 198 requests in 2 seconds!
```

### 2.4 Sliding Window Log

```python
import time
from collections import deque

class SlidingWindowLog:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.logs = {}  # {user_id: deque([timestamps])}
    
    def allow_request(self, user_id):
        now = time.time()
        
        if user_id not in self.logs:
            self.logs[user_id] = deque()
        
        log = self.logs[user_id]
        
        # Remove expired entries
        while log and log[0] < now - self.window_size:
            log.popleft()
        
        # Check if under limit
        if len(log) < self.limit:
            log.append(now)
            return True
        else:
            return False
```

**Pros**: Accurate, no burst at boundary  
**Cons**: Memory-intensive (stores all timestamps)

### 2.5 Sliding Window Counter (Redis)

```python
import redis
import time

r = redis.Redis()

def sliding_window_counter(user_id, limit=100, window=60):
    now = int(time.time())
    key = f"rate_limit:{user_id}"
    
    # Redis sorted set: {timestamp: 1}
    # Remove expired entries
    r.zremrangebyscore(key, 0, now - window)
    
    # Count requests in window
    count = r.zcard(key)
    
    if count < limit:
        # Add current request
        r.zadd(key, {now: now})
        r.expire(key, window)
        return True
    else:
        return False

# Usage
if sliding_window_counter(user_id, limit=100, window=60):
    process_request()
else:
    return 429  # Too Many Requests
```

---

## 3. Distributed Rate Limiting

### 3.1 Redis-Based (Centralized)

```python
import redis

r = redis.Redis(host='redis-cluster')

def rate_limit(user_id, limit=100, window=60):
    key = f"rate_limit:{user_id}:{int(time.time() // window)}"
    
    # Increment counter
    count = r.incr(key)
    
    # Set expiry on first request
    if count == 1:
        r.expire(key, window)
    
    return count <= limit

# Usage across multiple app servers:
# Server 1, Server 2, Server 3 all use same Redis
# → Consistent rate limiting
```

### 3.2 Token Bucket in Redis

```python
def token_bucket_redis(user_id, capacity=10, refill_rate=1):
    key = f"bucket:{user_id}"
    now = time.time()
    
    # Lua script (atomic)
    script = """
    local tokens_key = KEYS[1]
    local timestamp_key = KEYS[2]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    
    local last_refill = tonumber(redis.call('get', timestamp_key) or now)
    local tokens = tonumber(redis.call('get', tokens_key) or capacity)
    
    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * refill_rate)
    
    if tokens >= 1 then
        tokens = tokens - 1
        redis.call('set', tokens_key, tokens)
        redis.call('set', timestamp_key, now)
        return 1
    else
        return 0
    end
    """
    
    result = r.eval(script, 2, f"{key}:tokens", f"{key}:timestamp", 
                    capacity, refill_rate, now)
    
    return result == 1
```

---

## 4. Implementation Examples

### 4.1 FastAPI Middleware

```python
from fast fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/users")
@limiter.limit("100/minute")
def get_users(request: Request):
    return {"users": [...]}
```

### 4.2 Flask-Limiter

```python
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

@app.route("/api/users")
@limiter.limit("10/minute")
def get_users():
    return {"users": [...]}
```

### 4.3 NGINX Rate Limiting

```nginx
http {
    limit_req_zone $binary_remote_addr zone=mylimit:10m rate=10r/s;
    
    server {
        location /api/ {
            limit_req zone=mylimit burst=20 nodelay;
            proxy_pass http://backend;
        }
    }
}
```

---

## 5. Adaptive Throttling

### 5.1 Load-Based Throttling

```python
import psutil

def adaptive_rate_limit(user_id, base_limit=100):
    # Get CPU usage
    cpu_usage = psutil.cpu_percent()
    
    # Reduce limit if CPU high
    if cpu_usage > 80:
        adjusted_limit = base_limit * 0.5  # 50% limit
    elif cpu_usage > 60:
        adjusted_limit = base_limit * 0.75  # 75% limit
    else:
        adjusted_limit = base_limit
    
    return rate_limit(user_id, limit=adjusted_limit)
```

### 5.2 Quality of Service (QoS)

```python
def qos_rate_limit(user_id, tier):
    limits = {
        "free": 10,
        "basic": 100,
        "premium": 1000,
        "enterprise": 10000
    }
    
    limit = limits.get(tier, 10)
    return rate_limit(user_id, limit=limit)

# Usage
if qos_rate_limit(user_id, tier="premium"):
    process_request()
```

---

## 6. API Quotas

### 6.1 Daily Quotas

```python
def check_daily_quota(user_id, quota=1000):
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"quota:{user_id}:{today}"
    
    count = r.incr(key)
    
    # Set expiry to end of day
    if count == 1:
        r.expireat(key, int(datetime.now().replace(hour=23, minute=59, second=59).timestamp()))
    
    return count <= quota

# Usage
if not check_daily_quota(user_id, quota=1000):
    return {"error": "Daily quota exceeded"}, 429
```

### 6.2 Tiered Pricing

```python
PLANS = {
    "free": {"daily": 100, "rate": "10/minute"},
    "pro": {"daily": 10000, "rate": "100/minute"},
    "enterprise": {"daily": 1000000, "rate": "1000/minute"}
}

def check_limits(user_id, plan):
    config = PLANS[plan]
    
    # Check daily quota
    if not check_daily_quota(user_id, config["daily"]):
        return False, "Daily quota exceeded"
    
    # Check rate limit
    if not rate_limit(user_id, extract_limit(config["rate"])):
        return False, "Rate limit exceeded"
    
    return True, None
```

---

## 7. DDoS Protection

### 7.1 Challenge-Response

```python
from captcha.image import ImageCaptcha

def requires_captcha(user_id):
    # If too many requests → require CAPTCHA
    key = f"failed_attempts:{user_id}"
    failed_attempts = r.get(key) or 0
    
    return int(failed_attempts) > 5

@app.post("/api/login")
def login(username, password, captcha_response=None):
    if requires_captcha(username):
        if not verify_captcha(captcha_response):
            return {"error": "Invalid CAPTCHA"}, 400
    
    # Proceed with login
    ...
```

### 7.2 IP Blacklisting

```python
def check_blacklist(ip):
    return r.sismember("blacklist", ip)

@app.before_request
def block_blacklisted():
    ip = request.remote_addr
    
    if check_blacklist(ip):
        abort(403, "IP blacklisted")
```

---

## 8. Production Patterns

### 8.1 Multi-Tier Limits

```python
def multi_tier_rate_limit(user_id, ip):
    # Tier 1: Per-user limit
    if not rate_limit(f"user:{user_id}", limit=100, window=60):
        return False, "User rate limit exceeded"
    
    # Tier 2: Per-IP limit
    if not rate_limit(f"ip:{ip}", limit=1000, window=60):
        return False, "IP rate limit exceeded"
    
    # Tier 3: Global limit
    if not rate_limit("global", limit=100000, window=60):
        return False, "System overloaded"
    
    return True, None
```

### 8.2 Graceful Degradation

```python
@app.get("/api/users/{user_id}")
def get_user(user_id, detailed: bool = True):
    if not rate_limit(user_id, limit=100):
        # Exceeded limit → return cached/simplified data
        return get_cached_user(user_id)
    
    # Under limit → return full data
    if detailed:
        return get_detailed_user(user_id)
    else:
        return get_basic_user(user_id)
```

---

## 9. HTTP Headers

### 9.1 Rate Limit Headers

```python
@app.get("/api/users")
def get_users(request: Request, response: Response):
    user_id = get_user_id(request)
    limit = 100
    window = 60
    
    # Check rate limit
    key = f"rate_limit:{user_id}"
    count = r.incr(key)
    
    if count == 1:
        r.expire(key, window)
    
    # Add headers
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)
    
    if count > limit:
        response.status_code = 429
        retry_after = window - (int(time.time()) % window)
        response.headers["Retry-After"] = str(retry_after)
        return {"error": "Rate limit exceeded"}
    
    return {"users": [...]}
```

**Example Response**:
```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1609459260

HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1609459260
Retry-After: 42
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **Token Bucket** - Allows bursts, good for APIs
2. ✅ **Sliding Window** - Accurate, no boundary issues
3. ✅ **Distributed** - Redis for multi-server consistency
4. ✅ **Adaptive** - Adjust limits based on load
5. ✅ **Multi-Tier** - Per-user, per-IP, global limits
6. ✅ **HTTP Headers** - Communicate limits to clients
7. ✅ **Quotas** - Daily/monthly limits

### 10.2 Algorithm Comparison

| Algorithm | Accuracy | Memory | Bursts | Complexity |
|:----------|:---------|:-------|:-------|:-----------|
| **Token Bucket** | Good | Low | Yes | Low |
| **Leaky Bucket** | Good | Low | No | Low |
| **Fixed Window** | Poor | Low | Yes (boundary) | Very Low |
| **Sliding Window** | Excellent | High | No | Medium |
| **Sliding Counter** | Good | Medium | No | Medium |

### 10.3 Tomorrow (Day 28): Vector Databases in Production

- **Vector embeddings**: Generating & storing
- **Similarity search**: Cosine, euclidean, dot product
- **Indexing**: HNSW, IVF, PQ
- **Qdrant production**: Clustering, backups
- **RAG optimization**: Chunking strategies
- **Hybrid search**: Combining keyword + vector

See you tomorrow! 🚀

---

**File Statistics**: ~950 lines | Rate Limiting & Throttling mastered ✅
