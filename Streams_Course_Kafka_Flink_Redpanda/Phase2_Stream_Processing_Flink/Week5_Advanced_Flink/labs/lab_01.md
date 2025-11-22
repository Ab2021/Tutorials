# Lab 01: Flink SQL Basics

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Create a Table Environment.
-   Execute a simple SQL query.

## Problem Statement
1.  Create a DataStream of `(name, age)`.
2.  Register it as a view `People`.
3.  Run SQL: `SELECT name, age + 1 FROM People`.
4.  Print results.

## Starter Code
```python
t_env.create_temporary_view("People", ds)
result = t_env.sql_query("...")
```

## Hints
<details>
<summary>Hint 1</summary>
Use `StreamTableEnvironment`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.table import StreamTableEnvironment

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(env)

    ds = env.from_collection([("Alice", 25), ("Bob", 30)], 
                             type_info=Types.ROW([Types.STRING(), Types.INT()]))
    
    t_env.create_temporary_view("People", ds, ["name", "age"])
    
    result = t_env.sql_query("SELECT name, age + 1 FROM People")
    result.execute().print()
```
</details>
