# Lab 06: Sliding Windows

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement Sliding Windows.
-   Understand overlap.

## Problem Statement
Calculate the moving average of numbers over the last **1 minute**, updated every **10 seconds**.
Input: Stream of integers.

## Starter Code
```python
from pyflink.datastream.window import SlidingProcessingTimeWindows

# window(SlidingProcessingTimeWindows.of(Time.minutes(1), Time.seconds(10)))
```

## Hints
<details>
<summary>Hint 1</summary>
You might need a custom `AggregateFunction` or `ProcessWindowFunction` to calculate the average (sum/count).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Using Reduce for simplicity (Sum)
ds.key_by(lambda x: "key")   .window(SlidingProcessingTimeWindows.of(Time.minutes(1), Time.seconds(10)))   .reduce(lambda a, b: a + b)   .print()
```
</details>
