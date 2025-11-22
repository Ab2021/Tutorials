# Lab 09: ProcessWindowFunction

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `ProcessWindowFunction` to access window metadata (start/end time).

## Problem Statement
For a 10s tumbling window, output a string: `"Window [Start-End]: Sum = X"`.
You need `ProcessWindowFunction` to get the `Context`.

## Starter Code
```python
class MyProcessWindowFunction(ProcessWindowFunction):
    def process(self, key, context, elements):
        # context.window().start
        # context.window().end
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
In PyFlink, this is often a simple python function if using the functional API, or a class inheriting from `ProcessWindowFunction`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.functions import ProcessWindowFunction

class WindowLogger(ProcessWindowFunction):
    def process(self, key, context, elements):
        total = sum([e[1] for e in elements])
        start = context.window().start
        end = context.window().end
        yield f"Window [{start}-{end}]: Sum = {total}"

ds.key_by(...)   .window(...)   .process(WindowLogger())   .print()
```
</details>
