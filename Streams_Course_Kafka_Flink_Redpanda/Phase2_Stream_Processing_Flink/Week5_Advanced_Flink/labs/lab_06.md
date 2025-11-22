# Lab 06: CEP Looping Pattern

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `.times()` or `.oneOrMore()`.

## Problem Statement
Detect: 3 consecutive failed logins.

## Starter Code
```python
pattern = Pattern.begin("fail").where(...).times(3).consecutive()
```

## Hints
<details>
<summary>Hint 1</summary>
`consecutive()` ensures they are adjacent.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
pattern = Pattern.begin("fail").where(
    SimpleCondition(lambda x: x['type'] == 'Fail')
).times(3).consecutive()
```
</details>
