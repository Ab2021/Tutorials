# Lab 10: Pandas UDF (Vectorized)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Write a Vectorized UDF for performance.

## Problem Statement
Implement `add_one(series)` using Pandas. It should take a `pd.Series` and return a `pd.Series`.

## Starter Code
```python
@udf(result_type=Types.INT(), func_type="pandas")
def add_one(i):
    return i + 1
```

## Hints
<details>
<summary>Hint 1</summary>
Requires `pandas` and `pyarrow` installed.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
@udf(result_type=Types.INT(), func_type="pandas")
def add_one(i):
    return i + 1

# Usage is same as scalar UDF
```
</details>
