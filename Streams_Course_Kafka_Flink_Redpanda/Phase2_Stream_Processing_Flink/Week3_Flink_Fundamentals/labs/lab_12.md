# Lab 12: Rich Functions (Lifecycle)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `open()` and `close()` in a RichMapFunction.
-   Simulate a database connection.

## Problem Statement
Create a `RichMapFunction` that:
1.  In `open()`, prints "Opening DB Connection".
2.  In `map()`, appends " processed" to the input.
3.  In `close()`, prints "Closing DB Connection".

## Starter Code
```python
class MyMapper(RichMapFunction):
    def open(self, ctx):
        pass
    def map(self, value):
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
`open()` is called once per parallel task, not per element.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.functions import RichMapFunction

class DBMapper(RichMapFunction):
    def open(self, runtime_context):
        print("Opening DB Connection...")

    def map(self, value):
        return value + " processed"

    def close(self):
        print("Closing DB Connection...")

ds.map(DBMapper()).print()
```
</details>
