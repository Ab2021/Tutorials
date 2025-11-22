# Lab 06: Checkpointing Config

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Enable Checkpointing.
-   Configure interval and storage.

## Problem Statement
Configure the environment to:
1.  Checkpoint every 10 seconds.
2.  Use `ExactlyOnce` mode.
3.  Store checkpoints in `file:///tmp/flink-checkpoints`.

## Starter Code
```python
env.enable_checkpointing(10000)
# env.get_checkpoint_config()...
```

## Hints
<details>
<summary>Hint 1</summary>
Use `CheckpointingMode.EXACTLY_ONCE`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import CheckpointingMode

env.enable_checkpointing(10000)
config = env.get_checkpoint_config()
config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
config.set_checkpoint_storage("file:///tmp/flink-checkpoints")
```
</details>
