# Lab 13: Event Time Join

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Join two streams on Event Time.

## Problem Statement
Join `Clicks` and `Views` on `ad_id` where Click is within 10 mins of View.

## Starter Code
```python
ds1.join(ds2).where(...).equal_to(...).window(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Use `IntervalJoin` (KeyedStream.intervalJoin) for relative time.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Java/Scala API has intervalJoin. 
# In PyFlink, use SQL or Window Join.
t_env.execute_sql("""
    SELECT * FROM Clicks c, Views v
    WHERE c.ad_id = v.ad_id
    AND c.ts BETWEEN v.ts AND v.ts + INTERVAL '10' MINUTE
""")
```
</details>
