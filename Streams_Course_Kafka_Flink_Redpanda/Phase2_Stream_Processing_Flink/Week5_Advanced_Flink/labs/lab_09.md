# Lab 09: PyFlink UDF (Scalar)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Write a Python UDF.
-   Register it in SQL.

## Problem Statement
Write a UDF `to_upper(s)` that converts string to uppercase. Use it in SQL.

## Starter Code
```python
@udf(result_type=Types.STRING())
def to_upper(s):
    return s.upper()
```

## Hints
<details>
<summary>Hint 1</summary>
Register with `t_env.create_temporary_system_function`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.table.udf import udf

@udf(result_type=Types.STRING())
def to_upper(s):
    return s.upper()

t_env.create_temporary_system_function("to_upper", to_upper)

t_env.sql_query("SELECT to_upper(name) FROM People").execute().print()
```
</details>
