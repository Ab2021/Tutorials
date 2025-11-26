# Lab: Day 39 - Caching Strategies

## Goal
Speed up a slow API using Redis and HTTP Headers.

## Prerequisites
- Redis running.
- `pip install flask redis`

## Step 1: The Slow App (`app.py`)

```python
from flask import Flask, make_response
import time
import redis
import json

app = Flask(__name__)
r = redis.Redis(decode_responses=True)

# Simulate DB Call
def get_user_from_db(user_id):
    time.sleep(2) # Slow!
    return {"id": user_id, "name": "Alice"}

@app.route("/user/<user_id>")
def get_user(user_id):
    # 1. Check Cache
    cache_key = f"user:{user_id}"
    cached = r.get(cache_key)
    
    if cached:
        print("⚡ Cache Hit")
        data = json.loads(cached)
        source = "Redis"
    else:
        print("🐢 Cache Miss")
        data = get_user_from_db(user_id)
        # 2. Write Cache (TTL 10s)
        r.setex(cache_key, 10, json.dumps(data))
        source = "DB"
    
    # 3. HTTP Cache Header
    response = make_response({"data": data, "source": source})
    response.headers['Cache-Control'] = 'public, max-age=5' # Browser cache 5s
    return response

if __name__ == "__main__":
    app.run(port=5000)
```

## Step 2: Run It
`python app.py`

## Step 3: Test
1.  **First Request**: `curl -v localhost:5000/user/1`.
    *   Takes 2s. Source: DB.
2.  **Second Request**: `curl -v localhost:5000/user/1`.
    *   Takes 5ms. Source: Redis.
3.  **Browser Test**: Open Chrome DevTools -> Network.
    *   Visit URL. Refresh immediately.
    *   Status: `200 (from disk cache)`. (Browser didn't even call Python!).

## Challenge
Implement **Cache Stampede Protection**.
Use `redis_lock` to ensure only one thread queries the DB for a missing key, while others wait.
