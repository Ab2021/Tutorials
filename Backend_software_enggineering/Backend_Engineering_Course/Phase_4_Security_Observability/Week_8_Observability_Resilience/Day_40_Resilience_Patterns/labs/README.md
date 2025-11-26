# Lab: Day 40 - Circuit Breaker

## Goal
Implement a Circuit Breaker to protect your app.

## Prerequisites
- `pip install pybreaker`

## Step 1: The Code (`breaker.py`)

```python
import pybreaker
import time
import random

# 1. Configure Breaker
# Trip if 3 errors occur. Reset after 5 seconds.
breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=5)

class ServiceError(Exception):
    pass

@breaker
def call_flaky_service():
    # Simulate 60% failure rate
    if random.random() < 0.6:
        print("❌ Service Failed!")
        raise ServiceError("Boom")
    print("✅ Service Success!")
    return "OK"

# 2. Simulation
print("--- Starting Requests ---")
for i in range(20):
    try:
        print(f"Request {i+1}: ", end="")
        call_flaky_service()
        time.sleep(0.5)
    except pybreaker.CircuitBreakerError:
        print("⛔ Circuit OPEN! (Fast Fail)")
        time.sleep(0.5)
    except ServiceError:
        pass # Handled by breaker counting
```

## Step 2: Run It
`python breaker.py`

*   **Observe**:
    1.  You will see a few "Service Failed".
    2.  Then "Circuit OPEN".
    3.  Requests will fail *immediately* without calling the service.
    4.  After 5s, it will try again (Half-Open).

## Challenge
Implement **Exponential Backoff Retry**.
Write a decorator `@retry(attempts=3, delay=1, backoff=2)` that retries a function if it raises an exception.
Combine it with the Circuit Breaker.
