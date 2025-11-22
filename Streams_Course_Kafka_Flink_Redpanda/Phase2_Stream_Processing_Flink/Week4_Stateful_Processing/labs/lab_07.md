# Lab 07: RocksDB Backend

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Switch State Backend to RocksDB.
-   Understand dependencies.

## Problem Statement
Configure the job to use `EmbeddedRocksDBStateBackend`.
*Note: You need the `flink-statebackend-rocksdb` JAR.*

## Starter Code
```python
env.set_state_backend(EmbeddedRocksDBStateBackend())
```

## Hints
<details>
<summary>Hint 1</summary>
In PyFlink, you might need to add the JAR via `env.add_jars()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.state import EmbeddedRocksDBStateBackend

# Ensure JAR is loaded
env.add_jars("file:///path/to/flink-statebackend-rocksdb.jar")

env.set_state_backend(EmbeddedRocksDBStateBackend())
```
</details>
