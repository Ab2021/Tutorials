# Lab 08: Session Windows

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement Session Windows.
-   Understand the "Gap".

## Problem Statement
Group user clicks into sessions. A session ends if the user is idle for **5 seconds**.
Count clicks per session.

## Starter Code
```python
from pyflink.datastream.window import ProcessingTimeSessionWindows

# window(ProcessingTimeSessionWindows.with_gap(Time.seconds(5)))
```

## Hints
<details>
<summary>Hint 1</summary>
Session windows merge. The key is the user ID.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
ds.key_by(lambda x: x['user_id'])   .window(ProcessingTimeSessionWindows.with_gap(Time.seconds(5)))   .sum('clicks')   .print()
```
</details>
