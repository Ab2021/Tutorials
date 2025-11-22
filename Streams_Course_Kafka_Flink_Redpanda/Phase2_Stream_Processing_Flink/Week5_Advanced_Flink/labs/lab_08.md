# Lab 08: Upsert Kafka SQL

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `upsert-kafka` connector.
-   Write a changelog stream.

## Problem Statement
Read from `Orders`, aggregate `SUM(amount)` per user, and write to `UserStats` (Kafka Upsert topic).

## Starter Code
```sql
CREATE TABLE UserStats (
  user_name STRING PRIMARY KEY NOT ENFORCED,
  total_amount INT
) WITH (
  'connector' = 'upsert-kafka',
  ...
)
```

## Hints
<details>
<summary>Hint 1</summary>
Upsert Kafka requires a Primary Key.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
t_env.execute_sql("""
    CREATE TABLE UserStats (
      user_name STRING PRIMARY KEY NOT ENFORCED,
      total_amount INT
    ) WITH (
      'connector' = 'upsert-kafka',
      'topic' = 'user_stats',
      'properties.bootstrap.servers' = 'localhost:9092',
      'key.format' = 'json',
      'value.format' = 'json'
    )
""")

t_env.execute_sql("""
    INSERT INTO UserStats
    SELECT user_name, SUM(amount)
    FROM Orders
    GROUP BY user_name
""")
```
</details>
