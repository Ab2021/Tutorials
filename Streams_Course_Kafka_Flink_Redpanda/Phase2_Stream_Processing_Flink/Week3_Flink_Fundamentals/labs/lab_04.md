# Lab 04: Filter & Map

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use basic transformations.
-   Parse JSON strings.

## Problem Statement
Read a stream of JSON strings `{"user": "A", "age": 25}`.
1.  Parse JSON.
2.  Filter out users under 18.
3.  Map to `Name: A`.

## Starter Code
```python
import json
# ds.map(json.loads).filter(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Handle JSON parsing errors gracefully (try/except) or the job will fail.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import json
from pyflink.common import Types

def parse_and_filter(ds):
    return ds         .map(lambda x: json.loads(x), output_type=Types.MAP(Types.STRING(), Types.STRING()))         .filter(lambda x: int(x['age']) >= 18)         .map(lambda x: f"Name: {x['user']}", output_type=Types.STRING())
```
</details>
