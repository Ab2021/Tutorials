# Lab 08: Idempotent Sink

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Understand Idempotency.

## Problem Statement
Write a Sink that writes to a Python Dictionary (simulating a KV store).
Ensure that writing `(Key, Val)` twice results in the same state.

## Starter Code
```python
store = {}
def sink(k, v):
    store[k] = v
```

## Hints
<details>
<summary>Hint 1</summary>
A dictionary assignment is naturally idempotent. `list.append` is NOT.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class IdempotentSink(SinkFunction):
    def __init__(self):
        self.store = {}

    def invoke(self, value, context):
        # Idempotent: Overwrite
        self.store[value[0]] = value[1]
        print(self.store)
```
</details>
