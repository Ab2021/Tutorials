# Lab 07: Event Time & Watermarks

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Assign timestamps and watermarks.
-   Use Event Time windows.

## Problem Statement
Read a stream of CSVs: `timestamp, value`.
1.  Assign timestamps from the first column.
2.  Generate watermarks (BoundedOutOfOrderness = 5 seconds).
3.  Window by Event Time (10s tumbling).

## Starter Code
```python
class MyTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value, record_timestamp):
        return int(value[0])

watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5))     .with_timestamp_assigner(MyTimestampAssigner())
```

## Hints
<details>
<summary>Hint 1</summary>
Timestamps must be in milliseconds.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.common import WatermarkStrategy, Duration
from pyflink.datastream.window import TumblingEventTimeWindows

# Define Strategy
watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5))     .with_timestamp_assigner(lambda event, timestamp: int(event.split(',')[0]))

# Apply
ds = env.from_collection([...])     .assign_timestamps_and_watermarks(watermark_strategy)     .key_by(...)     .window(TumblingEventTimeWindows.of(Time.seconds(10)))     .sum(...)
```
</details>
