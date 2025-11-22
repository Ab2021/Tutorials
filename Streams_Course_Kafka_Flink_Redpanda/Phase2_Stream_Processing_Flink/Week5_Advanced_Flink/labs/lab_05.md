# Lab 05: CEP Pattern (FollowedBy)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Define a relaxed sequence pattern.

## Problem Statement
Detect: `Login` followed by `Purchase` within 1 hour. (Other events can happen in between).

## Starter Code
```python
pattern = Pattern.begin("login")...followed_by("purchase")...within(Time.hours(1))
```

## Hints
<details>
<summary>Hint 1</summary>
Don't forget `.within()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
pattern = Pattern.begin("login").where(
    SimpleCondition(lambda x: x['type'] == 'Login')
).followed_by("purchase").where(
    SimpleCondition(lambda x: x['type'] == 'Purchase')
).within(Time.hours(1))
```
</details>
