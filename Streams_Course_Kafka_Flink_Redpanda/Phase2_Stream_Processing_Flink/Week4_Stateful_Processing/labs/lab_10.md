# Lab 10: Operator State (List)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement `CheckpointedFunction`.
-   Use Operator ListState.

## Problem Statement
Implement a Sink that buffers 10 elements and then prints them. It must persist the buffer in Operator State so that no data is lost on crash.

## Starter Code
```python
class BufferingSink(SinkFunction, CheckpointedFunction):
    def snapshot_state(self, context):
        self.checkpointed_state.clear()
        for element in self.buffer:
            self.checkpointed_state.add(element)

    def initialize_state(self, context):
        self.checkpointed_state = context.get_operator_state_store().get_list_state(...)
```

## Hints
<details>
<summary>Hint 1</summary>
`snapshot_state` is called on checkpoint. `initialize_state` is called on start/restore.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class BufferingSink(SinkFunction, CheckpointedFunction):
    def __init__(self):
        self.buffer = []
        self.checkpointed_state = None

    def invoke(self, value, context):
        self.buffer.append(value)
        if len(self.buffer) >= 10:
            print(self.buffer)
            self.buffer.clear()

    def snapshot_state(self, context):
        self.checkpointed_state.clear()
        for element in self.buffer:
            self.checkpointed_state.add(element)

    def initialize_state(self, context):
        descriptor = ListStateDescriptor("buffered-elements", Types.INT())
        self.checkpointed_state = context.get_operator_state_store().get_list_state(descriptor)
        
        if context.is_restored():
            for element in self.checkpointed_state.get():
                self.buffer.append(element)
```
</details>
