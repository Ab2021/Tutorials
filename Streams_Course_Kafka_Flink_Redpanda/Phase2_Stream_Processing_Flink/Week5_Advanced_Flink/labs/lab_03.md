# Lab 03: SQL Join (Interval)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Perform an Interval Join in SQL.

## Problem Statement
Join `Orders` and `Shipments` where shipment happens within 1 hour of order.
`Orders(id, ts)`, `Shipments(order_id, ts)`.

## Starter Code
```sql
SELECT *
FROM Orders o, Shipments s
WHERE o.id = s.order_id
AND s.ts BETWEEN o.ts AND o.ts + INTERVAL '1' HOUR
```

## Hints
<details>
<summary>Hint 1</summary>
This is a standard SQL join with time bounds.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Assuming tables exist
sql = """
SELECT o.id, o.ts, s.ts
FROM Orders o
JOIN Shipments s ON o.id = s.order_id
WHERE s.ts BETWEEN o.ts AND o.ts + INTERVAL '1' HOUR
"""
t_env.execute_sql(sql).print()
```
</details>
