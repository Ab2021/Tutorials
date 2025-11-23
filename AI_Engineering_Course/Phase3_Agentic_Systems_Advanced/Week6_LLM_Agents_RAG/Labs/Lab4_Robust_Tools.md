# Lab 4: Robust Tool Calling

## Objective
Tools fail (API down, bad input). Agents must handle this.
We will implement a **Retry Decorator**.

## 1. The Decorator (`robust.py`)

```python
import time
import random

def retry(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1} failed: {e}")
                    time.sleep(1)
            return "Error: Tool failed after retries."
        return wrapper
    return decorator

@retry()
def flaky_api():
    if random.random() < 0.7:
        raise ValueError("Network Error")
    return "Success!"

print(flaky_api())
```

## 2. Analysis
The agent sees "Success!" or "Error...". It does not crash.
This allows the agent to decide what to do next (e.g., try a different tool).

## 3. Submission
Submit the output log showing retries.
