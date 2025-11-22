# Lab 15: Flink SQL Basics

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use the Table API / SQL.
-   Convert DataStream to Table.

## Problem Statement
1.  Create a DataStream of `(name, age)`.
2.  Convert to Table.
3.  Run SQL: `SELECT name FROM table WHERE age > 18`.
4.  Print result.

## Starter Code
```python
t_env = StreamTableEnvironment.create(env)
table = t_env.from_data_stream(ds)
result = t_env.sql_query("...")
```

## Hints
<details>
<summary>Hint 1</summary>
You need `flink-table` dependencies.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(env)

    ds = env.from_collection([("Alice", 25), ("Bob", 10)], 
                             type_info=Types.ROW([Types.STRING(), Types.INT()]))

    table = t_env.from_data_stream(ds, ["name", "age"])
    
    result = t_env.sql_query("SELECT name FROM %s WHERE age > 18" % table)
    
    result.execute().print()

if __name__ == '__main__':
    run()
```
</details>
