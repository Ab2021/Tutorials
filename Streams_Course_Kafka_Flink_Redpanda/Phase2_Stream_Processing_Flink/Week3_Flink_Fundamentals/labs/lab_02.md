# Lab 02: DataStream WordCount

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
-   Write a basic Flink DataStream application in Java/Python.
-   Use `socketTextStream`.
-   Use `flatMap` and `keyBy`.

## Problem Statement
Write a Flink job that reads text from a socket (port 9999), splits lines into words, counts them, and prints the result.

## Starter Code
```python
from pyflink.datastream import StreamExecutionEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
ds = env.socket_text_stream("localhost", 9999)

# ds.flat_map(...).key_by(...).sum(...)

env.execute("WordCount")
```

## Hints
<details>
<summary>Hint 1</summary>
You need `nc -lk 9999` running in a terminal to provide input.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment

def split(line):
    return line.split()

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    
    # Source
    text = env.socket_text_stream("host.docker.internal", 9999)
    
    # Transformation
    counts = text         .flat_map(lambda x: x.split(), output_type=Types.STRING())         .map(lambda i: (i, 1), output_type=Types.TUPLE([Types.STRING(), Types.INT()]))         .key_by(lambda i: i[0])         .sum(1)
        
    # Sink
    counts.print()
    
    env.execute("Socket WordCount")

if __name__ == '__main__':
    run()
```
</details>
