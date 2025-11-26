# Lab: Day 14 - Pub/Sub with Redis

## Goal
Implement an Event-Driven flow. We will use Redis Pub/Sub (simpler than setting up Kafka) to decouple an Order Service from an Email Service.

## Directory Structure
```
day14/
├── publisher.py (Order Service)
├── subscriber.py (Email Service)
└── requirements.txt
```

## Step 1: Requirements
```text
redis
```

## Step 2: The Subscriber (Email Service)
Run this first. It listens for events.

```python
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
pubsub = r.pubsub()

# Subscribe to topic
TOPIC = "orders"
pubsub.subscribe(TOPIC)

print(f"📧 Email Service listening on '{TOPIC}'...")

for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        print(f"✅ Received Event: Order #{data['id']} for {data['email']}")
        print(f"   -> Sending confirmation email to {data['email']}...")
```

## Step 3: The Publisher (Order Service)
Run this to generate events.

```python
import redis
import json
import time

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
TOPIC = "orders"

def create_order(order_id, user_email):
    order = {"id": order_id, "email": user_email, "total": 100}
    
    # 1. Save to DB (Mock)
    print(f"💾 Order #{order_id} saved to DB.")
    
    # 2. Publish Event
    r.publish(TOPIC, json.dumps(order))
    print(f"📢 Event published to '{TOPIC}'")

if __name__ == "__main__":
    # Simulate traffic
    create_order(101, "alice@example.com")
    time.sleep(1)
    create_order(102, "bob@example.com")
```

## Step 4: Run It

1.  **Start Redis**: `docker run -d -p 6379:6379 redis:alpine`
2.  **Start Subscriber**: `python subscriber.py`
3.  **Run Publisher**: `python publisher.py`

## Challenge
Redis Pub/Sub is "Fire and Forget". If the Subscriber is down, the message is lost.
*   **Task**: Switch to **Redis Streams** (`XADD`, `XREADGROUP`). This provides persistence and consumer groups, similar to Kafka.
