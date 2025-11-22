# Lab 05: State TTL

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Configure State Time-To-Live.
-   Prevent state leaks.

## Problem Statement
Modify Lab 01 (Running Sum) so that the state expires after **1 minute** of inactivity (no updates for that key).

## Starter Code
```python
ttl_config = StateTtlConfig.new_builder(Time.minutes(1))     .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)     .build()

descriptor.enable_time_to_live(ttl_config)
```

## Hints
<details>
<summary>Hint 1</summary>
TTL is configured on the `StateDescriptor`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.state import StateTtlConfig

    def open(self, ctx):
        ttl_config = StateTtlConfig.new_builder(Time.minutes(1))             .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)             .cleanup_full_snapshot()             .build()

        descriptor = ValueStateDescriptor("sum", Types.INT())
        descriptor.enable_time_to_live(ttl_config)
        
        self.sum_state = ctx.get_state(descriptor)
```
</details>
