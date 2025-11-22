# Lab 04: Debezium JSON Parsing

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Parse Debezium JSON format.

## Problem Statement
Input: Debezium JSON string `{"before": null, "after": {"id": 1, "val": "A"}, "op": "c"}`.
Task: Extract the "after" state. Filter out deletes (`op="d"`).

## Starter Code
```python
import json
# ds.map(json.loads).filter(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Check `op` field. `c`=create, `u`=update, `d`=delete, `r`=read (snapshot).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import json

def parse_cdc(record):
    data = json.loads(record)
    if data['op'] != 'd':
        return data['after']
    return None

ds.map(parse_cdc).filter(lambda x: x is not None).print()
```
</details>
