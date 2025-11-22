# Lab 09: Transactional Sink (2PC)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement `TwoPhaseCommitSinkFunction`.

## Problem Statement
(Reuse Week 4 Lab 15 logic).
Implement a sink that writes to a temporary file and renames it on commit.

## Starter Code
```python
# See Week 4 Lab 15
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure `pre_commit` flushes data.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Refer to Week 4 Lab 15 for the 2PC implementation.
# Key takeaway: The file is not visible until the checkpoint completes.
```
</details>
