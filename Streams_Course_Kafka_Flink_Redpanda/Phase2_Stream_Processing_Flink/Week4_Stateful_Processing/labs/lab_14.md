# Lab 14: State Processor API (Read)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Read a Savepoint offline.

## Problem Statement
Write a batch job using the State Processor API to read a Savepoint from Lab 01 and print the total sum of all keys.

## Starter Code
```java
// Java only (Python support is limited/experimental)
ExecutionEnvironment bEnv = ExecutionEnvironment.getExecutionEnvironment();
ExistingSavepoint savepoint = Savepoint.load(bEnv, "file:///tmp/savepoint", new MemoryStateBackend());
```

## Hints
<details>
<summary>Hint 1</summary>
You need to define a `KeyedStateReaderFunction`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```java
DataSet<Integer> sums = savepoint.readKeyedState(
    "stateful-op", 
    new ReaderFunction());

sums.sum(0).print();
```
</details>
