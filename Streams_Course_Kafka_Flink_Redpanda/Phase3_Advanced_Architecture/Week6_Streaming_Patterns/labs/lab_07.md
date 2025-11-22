# Lab 07: Temporal Table Join (SQL)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use SQL Temporal Join.

## Problem Statement
Join `Orders` with `Rates` using `FOR SYSTEM_TIME AS OF`.
(Requires setting up versioned table).

## Starter Code
```sql
SELECT * FROM Orders o
JOIN Rates FOR SYSTEM_TIME AS OF o.ts r
ON o.curr = r.curr
```

## Hints
<details>
<summary>Hint 1</summary>
Rates table must have a primary key and watermark.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
t_env.execute_sql("""
    SELECT o.id, o.amount * r.rate
    FROM Orders o
    JOIN Rates FOR SYSTEM_TIME AS OF o.ts r
    ON o.currency = r.currency
""").print()
```
</details>
