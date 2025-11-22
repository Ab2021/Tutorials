# Lab 05: Token Bucket Rate Limiter

## Difficulty
🟡 Medium

## Problem Statement
Implement the Token Bucket algorithm for API rate limiting.
- The bucket has a `capacity`.
- Tokens are added at a `refill_rate`.
- Each request consumes 1 token.
- If bucket empty, reject request.

## Starter Code
```python
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()

    def allow_request(self):
        # TODO: Refill tokens based on time elapsed
        # TODO: Check if token available
        pass
```
