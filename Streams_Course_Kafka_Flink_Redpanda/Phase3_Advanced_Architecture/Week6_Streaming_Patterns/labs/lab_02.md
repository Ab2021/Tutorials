# Lab 02: CQRS Projection

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Separate Write and Read models.

## Problem Statement
Stream: `OrderCreated(id, user, amount)`.
Projection: Maintain a `UserSpent` table (User -> Total Amount).
Implement the projection using a Flink Map/Reduce job.

## Starter Code
```python
ds.key_by(lambda x: x['user']).reduce(...)
```

## Hints
<details>
<summary>Hint 1</summary>
The "Read Model" here is the state of the Reduce function.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
ds.key_by(lambda x: x['user'])   .reduce(lambda a, b: {'user': a['user'], 'amount': a['amount'] + b['amount']})   .print()
```
</details>
