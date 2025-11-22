# Lab 04: CEP Pattern (Next)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Define a strict sequence pattern.

## Problem Statement
Detect pattern: `Start` event followed immediately by `End` event for the same ID.
Input: `(id, type)`.

## Starter Code
```python
pattern = Pattern.begin("start").where(...)     .next("end").where(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Use `SimpleCondition`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
pattern = Pattern.begin("start").where(
    SimpleCondition(lambda x: x['type'] == 'Start')
).next("end").where(
    SimpleCondition(lambda x: x['type'] == 'End')
)

CEP.pattern(ds.key_by(lambda x: x['id']), pattern)    .select(lambda map: f"Matched: {map['start'][0]} -> {map['end'][0]}")    .print()
```
</details>
