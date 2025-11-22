# Lab 08: Savepoints

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Trigger a Savepoint.
-   Stop and Resume a job.

## Problem Statement
1.  Run the `RunningSum` job (Lab 01).
2.  Trigger a savepoint via CLI: `flink savepoint <job_id> /tmp/savepoints`.
3.  Cancel the job.
4.  Resume from savepoint: `flink run -s /tmp/savepoints/savepoint-xxx ...`.

## Starter Code
```bash
# CLI commands
```

## Hints
<details>
<summary>Hint 1</summary>
Use `flink list` to find the Job ID.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
# 1. List jobs
flink list

# 2. Savepoint
flink savepoint <job_id> /tmp/savepoints

# 3. Cancel
flink cancel <job_id>

# 4. Resume
flink run -s /tmp/savepoints/savepoint-<id> -py job.py
```
</details>
