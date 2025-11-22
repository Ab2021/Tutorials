# Lab 02: SQL Window Aggregation

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `TUMBLE` window in SQL.

## Problem Statement
Input: `(user, amount, rowtime)`.
Query: Calculate sum of amount per user every 10 seconds (Tumbling Window).

## Starter Code
```sql
SELECT user, SUM(amount)
FROM Orders
GROUP BY user, TUMBLE(rowtime, INTERVAL '10' SECOND)
```

## Hints
<details>
<summary>Hint 1</summary>
You need to define a watermark strategy on the table for `rowtime` to work.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# DDL
t_env.execute_sql("""
    CREATE TABLE Orders (
        user_name STRING,
        amount INT,
        ts TIMESTAMP(3),
        WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'datagen'
    )
""")

# Query
t_env.execute_sql("""
    SELECT user_name, SUM(amount), TUMBLE_END(ts, INTERVAL '10' SECOND)
    FROM Orders
    GROUP BY user_name, TUMBLE(ts, INTERVAL '10' SECOND)
""").print()
```
</details>
