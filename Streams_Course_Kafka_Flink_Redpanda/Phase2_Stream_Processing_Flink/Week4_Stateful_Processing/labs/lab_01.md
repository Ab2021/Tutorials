# Lab 01: ValueState

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use `ValueState`.
-   Understand state lifecycle.

## Problem Statement
Implement a `RichFlatMapFunction` that keeps a running sum of integers per key.
Input: `(key, value)`.
Output: `(key, current_sum)`.

## Starter Code
```python
class SumFunction(RichFlatMapFunction):
    def open(self, ctx):
        state_desc = ValueStateDescriptor("sum", Types.INT())
        self.sum_state = ctx.get_state(state_desc)

    def flat_map(self, value, out):
        # current = self.sum_state.value()
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
`value()` returns `None` if state is empty. Handle that case.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RichFlatMapFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common import Types

class RunningSum(RichFlatMapFunction):
    def open(self, runtime_context):
        descriptor = ValueStateDescriptor("sum", Types.INT())
        self.sum_state = runtime_context.get_state(descriptor)

    def flat_map(self, value, out):
        current_sum = self.sum_state.value()
        if current_sum is None:
            current_sum = 0
        
        new_sum = current_sum + value[1]
        self.sum_state.update(new_sum)
        out.collect((value[0], new_sum))

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    ds = env.from_collection([("A", 1), ("A", 2), ("B", 5)])
    ds.key_by(lambda x: x[0]).flat_map(RunningSum()).print()
    env.execute()

if __name__ == '__main__':
    run()
```
</details>
