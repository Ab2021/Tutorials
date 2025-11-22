# Lab 04: ReducingState

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `ReducingState`.
-   Optimize aggregation.

## Problem Statement
Keep the "Max" value seen so far for each key. Use `ReducingState` instead of `ValueState` (it's more efficient because it merges on write).

## Starter Code
```python
class MaxReducer(ReduceFunction):
    def reduce(self, value1, value2):
        return max(value1, value2)

# In open():
# ctx.get_reducing_state(ReducingStateDescriptor("max", MaxReducer(), Types.INT()))
```

## Hints
<details>
<summary>Hint 1</summary>
`ReducingState.add(value)` automatically merges the new value with the existing one using your reducer.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class MaxFunction(RichFlatMapFunction):
    def open(self, ctx):
        self.max_state = ctx.get_reducing_state(
            ReducingStateDescriptor("max", lambda a, b: max(a, b), Types.INT())
        )

    def flat_map(self, value, out):
        self.max_state.add(value[1])
        out.collect((value[0], self.max_state.get()))
```
</details>
