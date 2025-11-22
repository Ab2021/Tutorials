# Lab 07: Kafka SQL Connector

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Define a Kafka table using DDL.

## Problem Statement
Create a table `KafkaTable` backed by topic `input`. Read from it using SQL.

## Starter Code
```sql
CREATE TABLE KafkaTable (
  `user` STRING,
  `age` INT
) WITH (
  'connector' = 'kafka',
  'topic' = 'input',
  'properties.bootstrap.servers' = 'localhost:9092',
  'format' = 'json'
)
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure you have the kafka-sql-connector JAR.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
t_env.execute_sql("""
    CREATE TABLE KafkaTable (
      `user` STRING,
      `age` INT
    ) WITH (
      'connector' = 'kafka',
      'topic' = 'input',
      'properties.bootstrap.servers' = 'localhost:9092',
      'properties.group.id' = 'testGroup',
      'scan.startup.mode' = 'earliest-offset',
      'format' = 'json'
    )
""")

t_env.execute_sql("SELECT * FROM KafkaTable").print()
```
</details>
