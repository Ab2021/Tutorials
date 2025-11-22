# Lab 12: Queryable State

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand Queryable State (Deprecated but useful concept).
-   Alternative: Expose state via Side Output to a DB.

## Problem Statement
Since Queryable State is deprecated, implement the modern equivalent:
Write a `RichFlatMapFunction` that updates state AND sends the update to a "Query" stream (Side Output) that writes to Redis.

## Starter Code
```python
# Side Output
ctx.output(query_tag, (key, new_value))
```

## Hints
<details>
<summary>Hint 1</summary>
This is the "CQRS" pattern in streaming.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
    def flat_map(self, value, out):
        # Update State
        self.state.update(value)
        
        # Emit to Main Stream
        out.collect(value)
        
        # Emit to Query Stream (Side Output)
        # In reality, you might write directly to Redis here in async mode, 
        # or use a Sink on the side output.
        pass 
```
</details>
