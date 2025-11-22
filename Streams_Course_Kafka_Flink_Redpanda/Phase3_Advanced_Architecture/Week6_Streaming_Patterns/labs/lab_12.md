# Lab 12: Deduplication

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Deduplicate a stream using State.

## Problem Statement
Filter out duplicate events (based on ID) seen within the last 10 minutes.

## Starter Code
```python
state_desc = ValueStateDescriptor("seen", Types.BOOLEAN())
ttl = StateTtlConfig...
```

## Hints
<details>
<summary>Hint 1</summary>
Use Keyed State with TTL.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class Dedupe(RichFlatMapFunction):
    def open(self, ctx):
        desc = ValueStateDescriptor("seen", Types.BOOLEAN())
        desc.enable_time_to_live(ttl_config) # 10 mins
        self.seen = ctx.get_state(desc)

    def flat_map(self, value, out):
        if self.seen.value() is None:
            self.seen.update(True)
            out.collect(value)
```
</details>
