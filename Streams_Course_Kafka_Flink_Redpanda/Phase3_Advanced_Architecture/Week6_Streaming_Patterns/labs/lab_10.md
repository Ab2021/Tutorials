# Lab 10: Dead Letter Queue (DLQ)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Handle poison pills.

## Problem Statement
Process a stream of integers. If input is not an integer, send to DLQ (Side Output).
Do not fail the job.

## Starter Code
```python
try:
    val = int(s)
except:
    # side output
```

## Hints
<details>
<summary>Hint 1</summary>
Use `OutputTag`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
dlq_tag = OutputTag("dlq", Types.STRING())

class SafeMap(RichProcessFunction):
    def process_element(self, value, ctx, out):
        try:
            out.collect(int(value))
        except ValueError:
            ctx.output(dlq_tag, value)

main = ds.process(SafeMap())
main.get_side_output(dlq_tag).print()
```
</details>
