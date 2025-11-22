# Lab 15: Two-Phase Commit (Sink)

## Difficulty
🔴 Hard

## Estimated Time
90 mins

## Learning Objectives
-   Understand `TwoPhaseCommitSinkFunction`.
-   Implement Exactly-Once to external systems.

## Problem Statement
Implement a dummy `TwoPhaseCommitSinkFunction` that simulates writing to a transactional file system.
-   `beginTransaction`: Create temp file.
-   `preCommit`: Flush to temp file.
-   `commit`: Rename temp to final.
-   `abort`: Delete temp.

## Starter Code
```python
class File2PCSink(TwoPhaseCommitSinkFunction):
    # Implement methods
    pass
```

## Hints
<details>
<summary>Hint 1</summary>
This is complex. Focus on the lifecycle logs.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Conceptual Python implementation (requires Java wrapper usually)
class My2PC(TwoPhaseCommitSinkFunction):
    def begin_transaction(self):
        return create_temp_file()

    def invoke(self, transaction, value, context):
        transaction.write(value)

    def pre_commit(self, transaction):
        transaction.flush()

    def commit(self, transaction):
        transaction.move_to_final()

    def abort(self, transaction):
        transaction.delete()
```
</details>
