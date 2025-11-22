# Lab 03: Kappa Backfill

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Simulate a Kappa Architecture backfill.

## Problem Statement
1.  Write a job that reads from a file (simulating Kafka history).
2.  Process all records.
3.  Switch to reading from a socket (simulating real-time).
*Note: In Flink, you can chain sources or use `HybridSource`.*

## Starter Code
```python
# HybridSource is complex in PyFlink.
# Simulate by reading file first, then socket.
```

## Hints
<details>
<summary>Hint 1</summary>
For this lab, just write a job that reads a file. The concept is that the *same code* runs on the file as on the stream.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Same logic for both
def process(ds):
    ds.map(lambda x: x.upper()).print()

# Backfill Job
env = StreamExecutionEnvironment.get_execution_environment()
ds = env.read_text_file("history.txt")
process(ds)
env.execute("Backfill")

# Realtime Job
# ds = env.socket_text_stream(...)
# process(ds)
```
</details>
