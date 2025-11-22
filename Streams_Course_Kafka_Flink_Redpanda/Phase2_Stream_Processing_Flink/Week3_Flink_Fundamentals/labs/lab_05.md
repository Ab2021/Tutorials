# Lab 05: Tumbling Windows

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement a Tumbling Processing Time Window.
-   Aggregate data per window.

## Problem Statement
Count words arriving from a socket, but aggregate them in **10-second tumbling windows**.
Output: `(word, count)` every 10 seconds.

## Starter Code
```python
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.common import Time

# ds.key_by(...).window(TumblingProcessingTimeWindows.of(Time.seconds(10))).sum(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Processing Time is easier for testing than Event Time (no watermarks needed).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
    ds.key_by(lambda x: x[0])       .window(TumblingProcessingTimeWindows.of(Time.seconds(10)))       .sum(1)       .print()
```
</details>
