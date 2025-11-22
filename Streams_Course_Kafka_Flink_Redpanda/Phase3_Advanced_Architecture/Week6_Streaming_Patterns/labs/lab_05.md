# Lab 05: Stream Enrichment (Async)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Simulate Async Enrichment.

## Problem Statement
Stream: `UserIDs`.
Enrichment: Call `fake_api(user_id)` which sleeps 0.1s and returns Name.
Use `AsyncDataStream` (or simulate with ThreadPool in Map if Async not available).

## Starter Code
```python
# PyFlink Async support requires specific setup.
# We will simulate the latency impact.
```

## Hints
<details>
<summary>Hint 1</summary>
Compare throughput of blocking `map` vs `async`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Conceptual Async Function
class AsyncEnricher(AsyncFunction):
    def async_invoke(self, input, result_future):
        # Call external API in thread
        val = call_api(input)
        result_future.complete([val])
```
</details>
