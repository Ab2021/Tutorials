# Lab 14: Speculative Execution (Concept)

## Difficulty
🔴 Hard

## Estimated Time
30 mins

## Learning Objectives
-   Understand Speculative Execution.

## Problem Statement
*Conceptual Lab*.
Flink does not support Speculative Execution (running duplicate tasks) because of State.
Explain why.

## Starter Code
```text
Write a short paragraph.
```

## Hints
<details>
<summary>Hint 1</summary>
If you run two copies of a stateful task, which one updates the state? Which one writes to the sink?
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Answer**: Flink tasks are stateful. If you run two copies, they would both try to update the state (concurrency issues) or produce duplicate side effects (writing to sink twice). Speculative execution works for stateless batch (MapReduce/Spark) but not for stateful streaming.
</details>
