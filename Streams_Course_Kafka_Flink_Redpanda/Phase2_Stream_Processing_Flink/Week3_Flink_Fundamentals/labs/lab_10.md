# Lab 10: Side Outputs (Late Data)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Handle late data using Side Outputs.

## Problem Statement
1.  Use Event Time window (10s).
2.  Set Allowed Lateness to 0.
3.  Send late data to a Side Output tag `late-data`.
4.  Print the main stream and the side output stream separately.

## Starter Code
```python
late_tag = OutputTag("late-data")

result = ds.window(...)     .side_output_late_data(late_tag)     .sum(...)

late_stream = result.get_side_output(late_tag)
```

## Hints
<details>
<summary>Hint 1</summary>
You need to simulate late data by sending a timestamp older than the current watermark.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import OutputTag

late_tag = OutputTag("late-data", Types.TUPLE([Types.STRING(), Types.INT()]))

main_stream = ds.window(...)     .side_output_late_data(late_tag)     .sum(1)

main_stream.print()
main_stream.get_side_output(late_tag).print_to_err() # Print late data to stderr
```
</details>
