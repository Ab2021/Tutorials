# Lab 13: UID Assignment

## Difficulty
🟡 Medium

## Estimated Time
30 mins

## Learning Objectives
-   Understand `uid()`.
-   Prevent state loss during topology changes.

## Problem Statement
1.  Write a job with `ds.map(...).keyBy(...).sum(...)`.
2.  Assign UIDs: `ds.map(...).uid("my-map")...`.
3.  Change the chain (insert a filter).
4.  Verify that state can still be restored because UIDs match.

## Starter Code
```python
ds.map(MyMap()).uid("mapper-1")
```

## Hints
<details>
<summary>Hint 1</summary>
If you don't assign UIDs, Flink generates them based on the graph structure. Changing the graph changes the IDs, making state unrecoverable.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
    ds.map(lambda x: x).uid("source-map")       .key_by(...)       .map(StatefulMap()).uid("stateful-op")       .print()
```
Always assign UIDs to stateful operators in production!
</details>
