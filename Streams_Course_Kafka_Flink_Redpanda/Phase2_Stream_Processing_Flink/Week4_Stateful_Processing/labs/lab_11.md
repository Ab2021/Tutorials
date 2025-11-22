# Lab 11: Schema Evolution (Avro)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Simulate schema evolution.

## Problem Statement
1.  Define an Avro schema `User(name)`.
2.  Run a job using this state. Take a savepoint.
3.  Update schema to `User(name, age=0)`.
4.  Resume job. Verify it works.

## Starter Code
```json
// user_v1.avsc
{"type":"record", "name":"User", "fields":[{"name":"name", "type":"string"}]}
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure you use `AvroSerializer`. Flink handles the mapping if the new schema has a default value for the new field.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

*Conceptual Steps*:
1.  Compile `User` class from V1 schema.
2.  Run job, populate state.
3.  Stop with Savepoint.
4.  Compile `User` class from V2 schema (with default for `age`).
5.  Update job code to use new class.
6.  Restore. Flink detects the schema change and adapts.
</details>
