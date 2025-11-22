# Lab 11: Throttling

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement a Throttler.

## Problem Statement
Limit the stream to 1 element per second.
(Useful for Kappa backfill to protect DB).

## Starter Code
```python
time.sleep(1)
```

## Hints
<details>
<summary>Hint 1</summary>
`time.sleep` in a MapFunction blocks the thread. In Flink, this effectively throttles the source (backpressure).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import time

ds.map(lambda x: (time.sleep(1), x)[1]).print()
```
</details>
