# Lab 03: MapState

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `MapState`.
-   Manage a dictionary per key.

## Problem Statement
Input: `(User, URL)`.
Keep a count of visits *per URL* for each User.
Output: `(User, URL, NewCount)`.

## Starter Code
```python
class UrlCounter(RichFlatMapFunction):
    def open(self, ctx):
        self.map_state = ctx.get_map_state(MapStateDescriptor("counts", Types.STRING(), Types.INT()))
```

## Hints
<details>
<summary>Hint 1</summary>
`map_state.get(key)` returns None if not found.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class UrlCounter(RichFlatMapFunction):
    def open(self, ctx):
        self.counts = ctx.get_map_state(MapStateDescriptor("counts", Types.STRING(), Types.INT()))

    def flat_map(self, value, out):
        user, url = value
        current = self.counts.get(url)
        if current is None:
            current = 0
        
        new_count = current + 1
        self.counts.put(url, new_count)
        out.collect((user, url, new_count))
```
</details>
