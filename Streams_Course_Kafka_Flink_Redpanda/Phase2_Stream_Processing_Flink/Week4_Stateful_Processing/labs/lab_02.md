# Lab 02: ListState

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `ListState`.
-   Buffer elements.

## Problem Statement
Buffer elements for a key until you have 5 elements, then emit the average and clear the buffer.

## Starter Code
```python
class BufferAverage(RichFlatMapFunction):
    def open(self, ctx):
        self.list_state = ctx.get_list_state(...)
```

## Hints
<details>
<summary>Hint 1</summary>
`list_state.get()` returns an iterable. `list_state.clear()` empties it.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class BufferAverage(RichFlatMapFunction):
    def open(self, ctx):
        self.buffer = ctx.get_list_state(ListStateDescriptor("buffer", Types.INT()))

    def flat_map(self, value, out):
        self.buffer.add(value[1])
        
        # Check size (inefficient for large lists, but okay for 5)
        elements = list(self.buffer.get())
        if len(elements) >= 5:
            avg = sum(elements) / len(elements)
            out.collect((value[0], avg))
            self.buffer.clear()
```
</details>
