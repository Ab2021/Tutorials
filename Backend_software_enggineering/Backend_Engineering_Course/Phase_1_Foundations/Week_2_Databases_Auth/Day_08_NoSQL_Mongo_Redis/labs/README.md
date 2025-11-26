# Lab: Day 8 - NoSQL Playground

## Goal
Get hands-on with MongoDB and Redis. You will perform CRUD operations in Mongo and implement a Cache + Leaderboard in Redis.

## Prerequisites
- Docker.
- Python + `pymongo` + `redis` libraries.

## Directory Structure
```
day08/
├── docker-compose.yml
├── mongo_lab.py
└── redis_lab.py
```

## Step 1: Docker Compose

```yaml
version: '3.8'
services:
  mongo:
    image: mongo:6.0
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## Step 2: MongoDB Lab (`mongo_lab.py`)

```python
from pymongo import MongoClient

# Connect
client = MongoClient("mongodb://admin:password@localhost:27017/")
db = client["school"]
students = db["students"]

# 1. Create (Insert with Embedding)
alice = {
    "name": "Alice",
    "age": 20,
    "grades": [
        {"subject": "Math", "score": 90},
        {"subject": "History", "score": 85}
    ],
    "address": {"city": "New York", "zip": "10001"}
}
students.insert_one(alice)
print("✅ Inserted Alice")

# 2. Read (Find)
res = students.find_one({"name": "Alice"})
print(f"🔍 Found: {res['name']} from {res['address']['city']}")

# 3. Update (Add a grade)
students.update_one(
    {"name": "Alice"},
    {"$push": {"grades": {"subject": "Physics", "score": 92}}}
)
print("✅ Added Physics grade")

# 4. Aggregation (Average Score)
pipeline = [
    {"$match": {"name": "Alice"}},
    {"$unwind": "$grades"},
    {"$group": {"_id": "$name", "avgScore": {"$avg": "$grades.score"}}}
]
avg = list(students.aggregate(pipeline))
print(f"📊 Average Score: {avg[0]['avgScore']}")
```

## Step 3: Redis Lab (`redis_lab.py`)

```python
import redis
import time

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 1. Caching (Strings)
def get_user(user_id):
    # Check Cache
    cached = r.get(f"user:{user_id}")
    if cached:
        print("⚡ Cache Hit!")
        return cached
    
    # Simulate DB Call
    print("🐢 DB Miss... Fetching...")
    time.sleep(1)
    data = "User Data Payload"
    
    # Set Cache with TTL
    r.set(f"user:{user_id}", data, ex=5)
    return data

print(get_user(1)) # Miss
print(get_user(1)) # Hit
time.sleep(6)
print(get_user(1)) # Miss (Expired)

# 2. Leaderboard (Sorted Sets)
print("\n🏆 Leaderboard Demo")
r.zadd("game_scores", {"Alice": 100, "Bob": 150, "Charlie": 120})

# Get Top 3
top = r.zrevrange("game_scores", 0, 2, withscores=True)
for rank, (name, score) in enumerate(top, 1):
    print(f"#{rank}: {name} - {score}")
```

## Step 4: Run It
1.  `docker-compose up -d`
2.  `pip install pymongo redis`
3.  `python mongo_lab.py`
4.  `python redis_lab.py`
